# Copyright 2015 The Bazel Authors. All rights reserved.
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

"""Jsonnet rules for Bazel."""

JSONNET_FILETYPE = FileType([".jsonnet"])

def _setup_deps(deps):
  """Collects source files and import flags of transitive dependencies.

  Args:
    deps: List of deps labels from ctx.attr.deps.

  Returns:
    Returns a struct containing the following fields:
      transitive_sources: List of Files containing sources of transitive
          dependencies
      imports: List of Strings containing import flags set by transitive
          dependency targets.
  """
  transitive_sources = set(order="compile")
  imports = set()
  for dep in deps:
    transitive_sources += dep.transitive_jsonnet_files
    imports += ["%s/%s" % (dep.label.package, im) for im in dep.imports]

  return struct(
      transitive_sources = transitive_sources,
      imports = imports)

def _jsonnet_library_impl(ctx):
  """Implementation of the jsonnet_library rule."""
  depinfo = _setup_deps(ctx.attr.deps)
  sources = depinfo.transitive_sources + ctx.files.srcs
  imports = depinfo.imports + ctx.attr.imports
  return struct(files = set(),
                transitive_jsonnet_files = sources,
                imports = imports)

def _jsonnet_toolchain(ctx):
  return struct(
      jsonnet_path = ctx.file._jsonnet.path,
      imports = ["-J %s" % ctx.file._std.dirname])

def _jsonnet_to_json_impl(ctx):
  """Implementation of the jsonnet_to_json rule."""
  depinfo = _setup_deps(ctx.attr.deps)
  toolchain = _jsonnet_toolchain(ctx)
  command = (
      [
          "set -e;",
          toolchain.jsonnet_path,
      ] +
      ["-J %s/%s" % (ctx.label.package, im) for im in ctx.attr.imports] +
      ["-J %s" % im for im in depinfo.imports] +
      toolchain.imports +
      ["-J ."])

  outputs = []
  # If multiple_outputs is set to true, then jsonnet will be invoked with the
  # -m flag for multiple outputs. Otherwise, jsonnet will write the resulting
  # JSON to stdout, which is redirected into a single JSON output file.
  if len(ctx.attr.outs) > 1 or ctx.attr.multiple_outputs:
    output_json_files = [ctx.new_file(ctx.configuration.bin_dir, out.name)
                         for out in ctx.attr.outs]
    outputs += output_json_files
    command += ["-m", ctx.file.src.path]
    # Currently, jsonnet -m creates the output files in the current working
    # directory. Append mv commands to move the output files into their
    # correct output directories.
    # TODO(dzc): Remove this hack when jsonnet supports a flag for setting
    # an output directory.
    for json_file in output_json_files:
      command += ["; mv %s %s" % (json_file.basename, json_file.path)]
  else:
    if len(ctx.attr.outs) > 1:
      fail("Only one file can be specified in outs if multiple_outputs is " +
           "not set.")

    compiled_json = ctx.new_file(ctx.configuration.bin_dir,
                                 ctx.attr.outs[0].name)
    outputs += [compiled_json]
    command += [ctx.file.src.path, "> %s" % compiled_json.path]

  compile_inputs = (
      [ctx.file.src, ctx.file._jsonnet, ctx.file._std] +
      list(depinfo.transitive_sources))

  ctx.action(
      inputs = compile_inputs,
      outputs = outputs,
      mnemonic = "Jsonnet",
      command = " ".join(command),
      use_default_shell_env = True,
      progress_message = "Compiling Jsonnet to JSON for " + ctx.label.name);

_jsonnet_common_attrs = {
    "deps": attr.label_list(providers = ["transitive_jsonnet_files"],
                            allow_files = False),
    "imports": attr.string_list(),
    "_jsonnet": attr.label(
        default = Label("//tools/build_defs/jsonnet:jsonnet"),
        executable = True,
        single_file = True),
    "_std": attr.label(default = Label("//tools/build_defs/jsonnet:std"),
                       single_file = True),
}

_jsonnet_library_attrs = {
    "srcs": attr.label_list(allow_files = JSONNET_FILETYPE),
}

jsonnet_library = rule(
    _jsonnet_library_impl,
    attrs = _jsonnet_library_attrs + _jsonnet_common_attrs,
)

_jsonnet_to_json_attrs = {
    "src": attr.label(allow_files = JSONNET_FILETYPE,
                      single_file = True),
    "outs": attr.output_list(mandatory = True),
    "multiple_outputs": attr.bool(),
}

jsonnet_to_json = rule(
    _jsonnet_to_json_impl,
    attrs = _jsonnet_to_json_attrs + _jsonnet_common_attrs,
)
