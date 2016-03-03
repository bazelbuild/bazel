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

_JSONNET_FILETYPE = FileType([".jsonnet"])

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
  jsonnet_vars = ctx.attr.vars
  jsonnet_code_vars = ctx.attr.code_vars
  command = (
      [
          "set -e;",
          toolchain.jsonnet_path,
      ] +
      ["-J %s/%s" % (ctx.label.package, im) for im in ctx.attr.imports] +
      ["-J %s" % im for im in depinfo.imports] +
      toolchain.imports +
      ["-J ."] +
      ["--var '%s'='%s'"
          % (var, jsonnet_vars[var]) for var in jsonnet_vars.keys()] +
      ["--code-var '%s'='%s'"
          % (var, jsonnet_code_vars[var]) for var in jsonnet_code_vars.keys()])

  outputs = []
  # If multiple_outputs is set to true, then jsonnet will be invoked with the
  # -m flag for multiple outputs. Otherwise, jsonnet will write the resulting
  # JSON to stdout, which is redirected into a single JSON output file.
  if len(ctx.attr.outs) > 1 or ctx.attr.multiple_outputs:
    output_json_files = [ctx.new_file(ctx.configuration.bin_dir, out.name)
                         for out in ctx.attr.outs]
    outputs += output_json_files
    command += ["-m", output_json_files[0].dirname, ctx.file.src.path]
  else:
    if len(ctx.attr.outs) > 1:
      fail("Only one file can be specified in outs if multiple_outputs is " +
           "not set.")

    compiled_json = ctx.new_file(ctx.configuration.bin_dir,
                                 ctx.attr.outs[0].name)
    outputs += [compiled_json]
    command += [ctx.file.src.path, "-o", compiled_json.path]

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

_EXIT_CODE_COMPARE_COMMAND = """
EXIT_CODE=$?
EXPECTED_EXIT_CODE=%d
if [ $EXIT_CODE -ne $EXPECTED_EXIT_CODE ] ; then
  echo "FAIL (exit code): %s"
  echo "Expected: $EXPECTED_EXIT_CODE"
  echo "Actual: $EXIT_CODE"
  echo "Output: $OUTPUT"
  exit 1
fi
"""

_DIFF_COMMAND = """
GOLDEN=$(cat %s)
if [ "$OUTPUT" != "$GOLDEN" ]; then
  echo "FAIL (output mismatch): %s"
  echo "Diff:"
  diff <(echo $GOLDEN) <(echo $OUTPUT)
  echo "Expected: $GOLDEN"
  echo "Actual: $OUTPUT"
  exit 1
fi
"""

_REGEX_DIFF_COMMAND = """
GOLDEN_REGEX=$(cat %s)
if [[ ! "$OUTPUT" =~ $GOLDEN_REGEX ]]; then
  echo "FAIL (regex mismatch): %s"
  echo "Output: $OUTPUT"
  exit 1
fi
"""

def _jsonnet_to_json_test_impl(ctx):
  """Implementation of the jsonnet_to_json_test rule."""
  depinfo = _setup_deps(ctx.attr.deps)
  toolchain = _jsonnet_toolchain(ctx)

  golden_files = []
  if ctx.file.golden:
    golden_files += [ctx.file.golden]
    if ctx.attr.regex:
      diff_command = _REGEX_DIFF_COMMAND % (ctx.file.golden.short_path,
                                           ctx.label.name)
    else:
      diff_command = _DIFF_COMMAND % (ctx.file.golden.short_path,
                                     ctx.label.name)

  jsonnet_vars = ctx.attr.vars
  jsonnet_code_vars = ctx.attr.code_vars
  jsonnet_command = " ".join(
      ["OUTPUT=$(%s" % ctx.file._jsonnet.short_path] +
      ["-J %s/%s" % (ctx.label.package, im) for im in ctx.attr.imports] +
      ["-J %s" % im for im in depinfo.imports] +
      toolchain.imports +
      ["-J ."] +
      ["--var %s=%s"
          % (var, jsonnet_vars[var]) for var in jsonnet_vars.keys()] +
      ["--code-var %s=%s"
          % (var, jsonnet_code_vars[var]) for var in jsonnet_code_vars.keys()] +
      [
          ctx.file.src.path,
          "2>&1)",
      ])

  command = "\n".join([
      "#!/bin/bash",
      jsonnet_command,
      _EXIT_CODE_COMPARE_COMMAND % (ctx.attr.error, ctx.label.name),
      diff_command])

  ctx.file_action(output = ctx.outputs.executable,
                  content = command,
                  executable = True);

  test_inputs = (
      [ctx.file.src, ctx.file._jsonnet, ctx.file._std] +
      golden_files +
      list(depinfo.transitive_sources))

  return struct(
      runfiles = ctx.runfiles(files = test_inputs, collect_data = True))

_jsonnet_common_attrs = {
    "deps": attr.label_list(providers = ["transitive_jsonnet_files"],
                            allow_files = False),
    "imports": attr.string_list(),
    "_jsonnet": attr.label(
        default = Label("@bazel_tools//tools/build_defs/jsonnet:jsonnet"),
        executable = True,
        single_file = True),
    "_std": attr.label(
        default = Label("@bazel_tools//tools/build_defs/jsonnet:std"),
        single_file = True),
}

_jsonnet_library_attrs = {
    "srcs": attr.label_list(allow_files = _JSONNET_FILETYPE),
}

jsonnet_library = rule(
    _jsonnet_library_impl,
    attrs = _jsonnet_library_attrs + _jsonnet_common_attrs,
)

_jsonnet_compile_attrs = {
    "src": attr.label(allow_files = _JSONNET_FILETYPE,
                      single_file = True),
    "vars": attr.string_dict(),
    "code_vars": attr.string_dict(),
}

_jsonnet_to_json_attrs = _jsonnet_compile_attrs + {
    "outs": attr.output_list(mandatory = True),
    "multiple_outputs": attr.bool(),
}

jsonnet_to_json = rule(
    _jsonnet_to_json_impl,
    attrs = _jsonnet_to_json_attrs + _jsonnet_common_attrs,
)

_jsonnet_to_json_test_attrs = _jsonnet_compile_attrs + {
    "golden": attr.label(allow_files = True, single_file = True),
    "error": attr.int(),
    "regex": attr.bool(),
}

jsonnet_to_json_test = rule(
    _jsonnet_to_json_test_impl,
    attrs = _jsonnet_to_json_test_attrs + _jsonnet_common_attrs,
    executable = True,
    test = True,
)

def jsonnet_repositories():
  native.git_repository(
      name = "jsonnet",
      remote = "https://github.com/google/jsonnet.git",
      tag = "v0.8.1",
  )
