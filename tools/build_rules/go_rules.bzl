# Copyright 2014 Google Inc. All rights reserved.
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

Several issues:

- Currently hardcoded to 6g from the GC suite. We should be able to
  extract the CPU type from ctx.configuration instead.

- It would be nice to be able to create a full-fledged Go
  configuration in Skylark.

- Almost supports checked in compilers (should use a filegroup under
  tools/go instead of symlink).

- No C++ interop.

- deps must be populated by hand.

- no test sharding or test XML.

"""

go_filetype = FileType([".go"])

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

  for item in artifact_dict.items():
    old_path = item[0]
    new_path = item[1]
    new_dir = new_path[:new_path.rfind('/')]
    up = (new_dir.count('/') + 1 +
          dest_dir.count('/') + 1)
    cmds += [
      "mkdir -p %s/%s" % (dest_dir, new_dir),
      "ln -s %s%s %s/%s" % ('../' * up, old_path, dest_dir, new_path),
    ]
  return cmds


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
  prefix = ""
  if ctx.workspace_name:
    ctx.workspace_name + "/"

  tree_layout = {}
  inputs = []
  for d in deps:
    library_artifact_path = d.go_library_object.path[config_strip:]
    tree_layout[d.go_library_object.path] = prefix + library_artifact_path
    inputs += [d.go_library_object]

  inputs += list(sources)
  for s in sources:
    tree_layout[s.path] = s.path

  cmds = symlink_tree_commands(out_dir, tree_layout)
  args = [
      "cd ", out_dir, "&&",
      ('../' * out_depth) + ctx.files.go_root[0].path + "/bin/go",
      "tool", "6g",
      "-o", ('../' * out_depth) + out_lib.path, "-pack",

      # Import path.
      "-I", "."]

  # Set -p to the import path of the library, ie.
  # (ctx.label.package + "/" ctx.label.name) for now.
  cmds += [ "export GOROOT=$(pwd)/" + ctx.files.go_root[0].path,
    ' '.join(args + cmd_helper.template(sources, "%{path}"))]

  ctx.action(
      inputs = inputs,
      outputs = [out_lib],
      mnemonic = "GoCompile",
      command =  " && ".join(cmds))


def go_library_impl(ctx):
  """Implements the go_library() rule."""

  sources = set(ctx.files.srcs)
  deps = ctx.targets.deps
  if ctx.targets.library:
    sources += ctx.target.library.go_sources
    deps += ctx.target.library.direct_deps

  if not sources:
    fail("may not be empty", "srcs")

  out_lib = ctx.outputs.lib
  emit_go_compile_action(ctx, set(sources), deps, out_lib)

  transitive_libs = set([out_lib])
  for dep in ctx.targets.deps:
     transitive_libs += dep.transitive_go_library_object

  return struct(
    files = set([out_lib]),
    direct_deps = deps,
    go_sources = sources,
    go_library_object = out_lib,
    transitive_go_library_object = transitive_libs)


def emit_go_link_action(ctx, transitive_libs, lib, executable):
  """Sets up a symlink tree to libraries to link together."""
  out_dir = executable.path + ".dir"
  out_depth = out_dir.count('/') + 1
  tree_layout = {}

  config_strip = len(ctx.configuration.bin_dir.path) + 1
  prefix = ""
  if ctx.workspace_name:
    prefix = ctx.workspace_name + "/"

  for l in transitive_libs:
    library_artifact_path = l.path[config_strip:]
    tree_layout[l.path] = prefix + library_artifact_path

  tree_layout[lib.path] = prefix + lib.path[config_strip:]
  tree_layout[executable.path] = prefix + executable.path[config_strip:]

  cmds = symlink_tree_commands(out_dir, tree_layout)
  cmds += [
    "export GOROOT=$(pwd)/" + ctx.files.go_root[0].path,
    "cd " + out_dir,
    ' '.join([
      ('../' * out_depth) + ctx.files.go_root[0].path + "/bin/go",
      "tool", "6l", "-L", ".",
      "-o", prefix + executable.path[config_strip:],
      prefix + lib.path[config_strip:]])]

  ctx.action(
      inputs = list(transitive_libs) + [lib],
      outputs = [executable],
      command = ' && '.join(cmds),
      mnemonic = "GoLink")


def go_binary_impl(ctx):
  """go_binary_impl emits the link action for a go executable."""
  lib_result = go_library_impl(ctx)
  executable = ctx.outputs.executable
  lib_out = ctx.outputs.lib

  emit_go_link_action(
    ctx, lib_result.transitive_go_library_object, lib_out, executable)
  return struct(files = set([executable]) + lib_result.files)


def go_test_impl(ctx):
  """go_test_impl implements go testing.

  It emits an action to run the test generator, and then compile the
  test."""

  lib_result = go_library_impl(ctx)
  main_go = ctx.outputs.main_go
  prefix = ""
  if ctx.workspace_name:
    prefix = ctx.workspace_name + "/"

  go_import = prefix + ctx.label.package + "/" + ctx.label.name

  args = (["--package", go_import, "--output", ctx.outputs.main_go.path] +
          cmd_helper.template(lib_result.go_sources, "%{path}"))
  ctx.action(
      inputs = list(lib_result.go_sources),
      executable = ctx.executable.test_generator,
      outputs = [main_go],
      mnemonic = "GoTestGenTest",
      arguments = args)

  emit_go_compile_action(
    ctx, set([main_go]), ctx.targets.deps + [lib_result], ctx.outputs.main_lib)

  emit_go_link_action(
    ctx, lib_result.transitive_go_library_object,
    ctx.outputs.main_lib, ctx.outputs.executable)

  # TODO(bazel-team): the Go tests should do a chdir to the directory
  # holding the data files, so open-source go tests continue to work
  # without code changes.
  runfiles = ctx.runfiles(collect_data = True, files = [ctx.outputs.executable])
  return struct(runfiles=runfiles)


go_library_attrs = {
    "data": attr.label_list(allow_files=True, cfg=DATA_CFG),
    "srcs": attr.label_list(allow_files=go_filetype),
    "deps": attr.label_list(
        providers=[
          "direct_deps",
          "go_library_object",
          "transitive_go_library_object",
        ]),
    "go_root": attr.label(
        default=Label("//tools/go:go_root"),
        allow_files=True,
        cfg=HOST_CFG),
    "library": attr.label(
        providers=["go_sources"]),
    }

go_library_outputs = {
  "lib": "%{name}.a",
    }

go_library = rule(
    go_library_impl,
    attrs = go_library_attrs,
    outputs =go_library_outputs)

go_binary = rule(
    go_binary_impl,
    executable = True,
    attrs = go_library_attrs + {
        "stamp": attr.bool(default=False),
        },
    outputs = go_library_outputs)

go_test = rule(
    go_test_impl,
    executable = True,
    test = True,
    attrs = go_library_attrs + {
      "test_generator": attr.label(
          default=Label("//tools/go:generate_test_main"),
          cfg=HOST_CFG, flags=["EXECUTABLE"]),
      # TODO(bazel-team): implement support for args and defines_main.
      "args": attr.string_list(),
      "defines_main": attr.bool(default=False),
      "stamp": attr.bool(default=False),
      },
    outputs =  {
      "lib" : "%{name}.a",
      "main_lib": "%{name}_main_test.a",
      "main_go": "%{name}_main_test.go",
})
