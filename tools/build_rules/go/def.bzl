# Copyright 2014 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""These are bare-bones Go rules.

In order of priority:

- No support for build tags

- BUILD file must be written by hand.

- No C++ interop (SWIG, cgo).

- No test sharding or test XML.

"""

_DEFAULT_LIB = "go_default_library"

_VENDOR_PREFIX = "/vendor/"

go_filetype = FileType([".go"])

################

# In Go, imports are always fully qualified with a URL,
# eg. github.com/user/project. Hence, a label //foo:bar from within a
# Bazel workspace must be referred to as
# "github.com/user/project/foo/bar". To make this work, each rule must
# know the repository's URL. This is achieved, by having all go rules
# depend on a globally unique target that has a "go_prefix" transitive
# info provider.

def _go_prefix_impl(ctx):
  """go_prefix_impl provides the go prefix to use as a transitive info provider."""
  return struct(go_prefix = ctx.attr.prefix)

def _go_prefix(ctx):
  """slash terminated go-prefix"""
  prefix = ctx.attr.go_prefix.go_prefix
  if prefix != "" and not prefix.endswith("/"):
    prefix = prefix + "/"
  return prefix

_go_prefix_rule = rule(
    _go_prefix_impl,
    attrs = {
        "prefix": attr.string(mandatory = True),
    },
)

def go_prefix(prefix):
  """go_prefix sets the Go import name to be used for this workspace."""
  _go_prefix_rule(name = "go_prefix",
    prefix = prefix,
    visibility = ["//visibility:public" ]
  )

################

# TODO(bazel-team): it would be nice if Bazel had this built-in.
def symlink_tree_commands(dest_dir, artifact_dict):
  """Symlink_tree_commands returns a list of commands to create the
  dest_dir, and populate it according to the given dict.

  Args:
    dest_dir: The destination directory, a string.
    artifact_dict: The mapping of exec-path => path in the dest_dir.

  Returns:
    A list of commands that will setup the symlink tree.
  """
  cmds = [
    "rm -rf " + dest_dir,
    "mkdir -p " + dest_dir,
  ]

  for old_path, new_path in artifact_dict.items():
    new_dir = new_path[:new_path.rfind('/')]
    up = (new_dir.count('/') + 1 +
          dest_dir.count('/') + 1)
    cmds += [
      "mkdir -p %s/%s" % (dest_dir, new_dir),
      "ln -s %s%s %s/%s" % ('../' * up, old_path, dest_dir, new_path),
    ]
  return cmds

def go_environment_vars(ctx):
  """Return a map of environment variables for use with actions, based on
  the arguments. Uses the ctx.fragments.cpp.cpu attribute, if present,
  and picks a default of target_os="linux" and target_arch="amd64"
  otherwise.

  Args:
    The skylark Context.

  Returns:
    A dict of environment variables for running Go tool commands that build for
    the target OS and architecture.
  """
  bazel_to_go_toolchain = {"k8": {"GOOS": "linux",
                                  "GOARCH": "amd64"},
                           "piii": {"GOOS": "linux",
                                    "GOARCH": "386"},
                           "darwin": {"GOOS": "darwin",
                                      "GOARCH": "amd64"},
                           "freebsd": {"GOOS": "freebsd",
                                       "GOARCH": "amd64"},
                           "armeabi-v7a": {"GOOS": "linux",
                                           "GOARCH": "arm"},
                           "arm": {"GOOS": "linux",
                                   "GOARCH": "arm"}}
  return bazel_to_go_toolchain.get(ctx.fragments.cpp.cpu,
                                   {"GOOS": "linux",
                                    "GOARCH": "amd64"})

def emit_go_compile_action(ctx, sources, deps, out_lib):
  """Construct the command line for compiling Go code.
  Constructs a symlink tree to accomodate for workspace name.

  Args:
    ctx: The skylark Context.
    sources: an iterable of source code artifacts (or CTs? or labels?)
    deps: an iterable of dependencies. Each dependency d should have an
      artifact in d.go_library_object representing an imported library.
    out_lib: the artifact (configured target?) that should be produced
  """
  config_strip = len(ctx.configuration.bin_dir.path) + 1

  out_dir = out_lib.path + ".dir"
  out_depth = out_dir.count('/') + 1
  tree_layout = {}
  inputs = []
  prefix = _go_prefix(ctx)
  import_map = {}
  for d in deps:
    library_artifact_path = d.go_library_object.path[config_strip:]
    tree_layout[d.go_library_object.path] = prefix + library_artifact_path
    inputs += [d.go_library_object]

    source_import = prefix + d.label.package + "/" + d.label.name
    actual_import = prefix + d.label.package + "/" + d.label.name
    if d.label.name == _DEFAULT_LIB:
      source_import = prefix + d.label.package

    if source_import.rfind(_VENDOR_PREFIX) != -1:
      source_import = source_import[len(_VENDOR_PREFIX) + source_import.rfind(_VENDOR_PREFIX):]

    if source_import != actual_import:
      if source_import in import_map:
        fail("duplicate import %s: adding %s and have %s"
             % (source_import, actual_import, import_map[source_import]))
      import_map[source_import] = actual_import

  inputs += list(sources)
  for s in sources:
    tree_layout[s.path] = prefix + s.path

  cmds = symlink_tree_commands(out_dir, tree_layout)
  args = [
      "cd ", out_dir, "&&",
      ('../' * out_depth) + ctx.file.go_tool.path,
      "tool", "compile",
      "-o", ('../' * out_depth) + out_lib.path, "-pack",

      # Import path.
      "-I", "."] + [
        "-importmap=%s=%s" % (k,v) for k, v in import_map.items()
      ]

  # Set -p to the import path of the library, ie.
  # (ctx.label.package + "/" ctx.label.name) for now.
  cmds += [ "export GOROOT=$(pwd)/" + ctx.file.go_tool.dirname + "/..",
    ' '.join(args + cmd_helper.template(sources, prefix + "%{path}"))]

  ctx.action(
      inputs = inputs + ctx.files.toolchain,
      outputs = [out_lib],
      mnemonic = "GoCompile",
      command =  " && ".join(cmds),
      env = go_environment_vars(ctx))

def go_library_impl(ctx):
  """Implements the go_library() rule."""

  sources = set(ctx.files.srcs)
  deps = ctx.attr.deps
  if ctx.attr.library:
    sources += ctx.attr.library.go_sources
    deps += ctx.attr.library.direct_deps

  if not sources:
    fail("may not be empty", "srcs")

  out_lib = ctx.outputs.lib
  emit_go_compile_action(ctx, set(sources), deps, out_lib)

  transitive_libs = set([out_lib])
  for dep in ctx.attr.deps:
     transitive_libs += dep.transitive_go_library_object

  runfiles = ctx.runfiles(collect_data = True)
  return struct(
    label = ctx.label,
    files = set([out_lib]),
    direct_deps = deps,
    runfiles = runfiles,
    go_sources = sources,
    go_library_object = out_lib,
    transitive_go_library_object = transitive_libs)

def emit_go_link_action(ctx, transitive_libs, lib, executable):
  """Sets up a symlink tree to libraries to link together."""
  out_dir = executable.path + ".dir"
  out_depth = out_dir.count('/') + 1
  tree_layout = {}

  config_strip = len(ctx.configuration.bin_dir.path) + 1
  prefix = _go_prefix(ctx)

  for l in transitive_libs:
    library_artifact_path = l.path[config_strip:]
    tree_layout[l.path] = prefix + library_artifact_path

  tree_layout[lib.path] = prefix + lib.path[config_strip:]
  tree_layout[executable.path] = prefix + executable.path[config_strip:]

  cmds = symlink_tree_commands(out_dir, tree_layout)
  cmds += [
    "export GOROOT=$(pwd)/" + ctx.file.go_tool.dirname + "/..",
    "cd " + out_dir,
    ' '.join([
      ('../' * out_depth) + ctx.file.go_tool.path,
      "tool", "link", "-L", ".",
      "-o", prefix + executable.path[config_strip:],
      prefix + lib.path[config_strip:]])]

  ctx.action(
      inputs = list(transitive_libs) + [lib] + ctx.files.toolchain,
      outputs = [executable],
      command = ' && '.join(cmds),
      mnemonic = "GoLink",
      env = go_environment_vars(ctx))

def go_binary_impl(ctx):
  """go_binary_impl emits actions for compiling and linking a go executable."""
  lib_result = go_library_impl(ctx)
  executable = ctx.outputs.executable
  lib_out = ctx.outputs.lib

  emit_go_link_action(
    ctx, lib_result.transitive_go_library_object, lib_out, executable)

  runfiles = ctx.runfiles(collect_data = True,
                          files = ctx.files.data)
  return struct(files = set([executable]) + lib_result.files,
                runfiles = runfiles)

def go_test_impl(ctx):
  """go_test_impl implements go testing.

  It emits an action to run the test generator, and then compiles the
  test into a binary."""

  lib_result = go_library_impl(ctx)
  main_go = ctx.outputs.main_go
  prefix = _go_prefix(ctx)

  go_import = prefix + ctx.label.package + "/" + ctx.label.name

  args = (["--package", go_import, "--output", ctx.outputs.main_go.path] +
          cmd_helper.template(lib_result.go_sources, "%{path}"))

  inputs = list(lib_result.go_sources) + list(ctx.files.toolchain)
  ctx.action(
      inputs = inputs,
      executable = ctx.executable.test_generator,
      outputs = [main_go],
      mnemonic = "GoTestGenTest",
      arguments = args,
      env = dict(go_environment_vars(ctx), RUNDIR=ctx.label.package))

  emit_go_compile_action(
    ctx, set([main_go]), ctx.attr.deps + [lib_result], ctx.outputs.main_lib)

  emit_go_link_action(
    ctx, lib_result.transitive_go_library_object,
    ctx.outputs.main_lib, ctx.outputs.executable)

  # TODO(bazel-team): the Go tests should do a chdir to the directory
  # holding the data files, so open-source go tests continue to work
  # without code changes.
  runfiles = ctx.runfiles(collect_data = True,
                          files = ctx.files.data + [ctx.outputs.executable])
  return struct(runfiles=runfiles)

go_library_attrs = {
    "data": attr.label_list(
        allow_files = True,
        cfg = DATA_CFG,
    ),
    "srcs": attr.label_list(allow_files = go_filetype),
    "deps": attr.label_list(
        providers = [
            "direct_deps",
            "go_library_object",
            "transitive_go_library_object",
        ],
    ),
    "toolchain": attr.label(
        default = Label("@//tools/build_rules/go/toolchain:toolchain"),
        allow_files = True,
        cfg = HOST_CFG,
    ),
    "go_tool": attr.label(
        default = Label("@//tools/build_rules/go/toolchain:go_tool"),
        single_file = True,
        allow_files = True,
        cfg = HOST_CFG,
    ),
    "library": attr.label(
        providers = ["go_sources"],
    ),
    "go_prefix": attr.label(
        providers = ["go_prefix"],
        default = Label("//:go_prefix"),
        allow_files = False,
        cfg = HOST_CFG,
    ),
}

go_library_outputs = {
    "lib": "%{name}.a",
}

go_library = rule(
    go_library_impl,
    attrs = go_library_attrs,
    fragments = ["cpp"],
    outputs = go_library_outputs,
)

go_binary = rule(
    go_binary_impl,
    attrs = go_library_attrs + {
        "stamp": attr.bool(default = False),
    },
    executable = True,
    fragments = ["cpp"],
    outputs = go_library_outputs,
)

go_test = rule(
    go_test_impl,
    attrs = go_library_attrs + {
        "test_generator": attr.label(
            executable = True,
            default = Label("//tools/build_rules/go/tools:generate_test_main"),
            cfg = HOST_CFG,
        ),
    },
    executable = True,
    fragments = ["cpp"],
    outputs = {
        "lib": "%{name}.a",
        "main_lib": "%{name}_main_test.a",
        "main_go": "%{name}_main_test.go",
    },
    test = True,
)

GO_TOOLCHAIN_BUILD_FILE = """
package(
  default_visibility = [ "//visibility:public" ])

filegroup(
  name = "toolchain",
  srcs = glob(["go/bin/*", "go/pkg/**", ]),
)

filegroup(
  name = "go_tool",
  srcs = [ "go/bin/go" ],
)
"""

def go_repositories():
  native.new_http_archive(
    name=  "golang_linux_amd64",
    url = "https://storage.googleapis.com/golang/go1.5.1.linux-amd64.tar.gz",
    build_file_content = GO_TOOLCHAIN_BUILD_FILE,
    sha256 = "2593132ca490b9ee17509d65ee2cd078441ff544899f6afb97a03d08c25524e7"
  )

  native.new_http_archive(
    name=  "golang_darwin_amd64",
    url = "https://storage.googleapis.com/golang/go1.5.1.darwin-amd64.tar.gz",
    build_file_content = GO_TOOLCHAIN_BUILD_FILE,
    sha256 = "e94487b8cd2e0239f27dc51e6c6464383b10acb491f753584605e9b28abf48fb"
  )
