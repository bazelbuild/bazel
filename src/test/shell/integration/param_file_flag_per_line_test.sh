#!/usr/bin/env bash
#
# Copyright 2019 The Bazel Authors. All rights reserved.
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
#
# Tests Starlark API pertaining to action inspection via aspect.

# --- begin runfiles.bash initialization ---
# Copy-pasted from Bazel's Bash runfiles library (tools/bash/runfiles/runfiles.bash).
set -euo pipefail
if [[ ! -d "${RUNFILES_DIR:-/dev/null}" && ! -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  if [[ -f "$0.runfiles_manifest" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles_manifest"
  elif [[ -f "$0.runfiles/MANIFEST" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles/MANIFEST"
  elif [[ -f "$0.runfiles/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
    export RUNFILES_DIR="$0.runfiles"
  fi
fi
if [[ -f "${RUNFILES_DIR:-/dev/null}/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
  source "${RUNFILES_DIR}/bazel_tools/tools/bash/runfiles/runfiles.bash"
elif [[ -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  source "$(grep -m1 "^bazel_tools/tools/bash/runfiles/runfiles.bash " \
            "$RUNFILES_MANIFEST_FILE" | cut -d ' ' -f 2-)"
else
  echo >&2 "ERROR: cannot find @bazel_tools//tools/bash/runfiles:runfiles.bash"
  exit 1
fi
# --- end runfiles.bash initialization ---

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

case "$(uname -s | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac


add_to_bazelrc "build --package_path=%workspace%"

function test_param_file_flag_per_line() {
  mkdir -p package

  # Python binary to copy positional args and param file to output
  # Expects param file name, output file name, then further args
  cat > package/flag_copy.py <<EOF
import sys


def main(argv):
  with open(argv[1], 'r') as args_in:
    args_file = args_in.read()
    with open(argv[2], 'w') as out:
      for arg in argv[3:]:
        out.write('%s\n' % arg)
      out.write('=== param_file\n')
      out.write(args_file)


if __name__ == '__main__':
  main(sys.argv)

EOF

  # Starlark rule to invoke the binary with specific positional
  # args and flags
  cat > package/lib.bzl <<EOF
def _flag_per_line_impl(ctx):
    args = ctx.actions.args()
    args.set_param_file_format("flag_per_line")
    args.use_param_file("%s", use_always = True)

    # Output path is a positional argument
    args.add(ctx.outputs.out.path)

    args.add("--foo", "bar")
    args.add("pos1")
    args.add("--nowoof")
    args.add_joined("--joined", ["a", "b", "c"], join_with=",")
    args.add("--baz", "boo")
    args.add("pos2")
    outputs = [ctx.outputs.out]
    ctx.actions.run(
        mnemonic = "CopyArgFile",
        outputs = outputs,
        executable = ctx.executable._flag_copy,
        arguments = [args],
    )
    return [DefaultInfo(files = depset(outputs))]

flag_per_line = rule(
    implementation = _flag_per_line_impl,
    attrs = {
        "out": attr.output(
            doc = """Output file.""",
            mandatory = True,
        ),
        "_flag_copy": attr.label(
            default = Label("//package:flag_copy"),
            executable = True,
            allow_files = True,
            cfg = "exec",
        ),
    },
)
EOF

  cat > package/BUILD <<EOF
load(":lib.bzl", "flag_per_line")
load("@rules_python//python:py_binary.bzl", "py_binary")

py_binary(
    name = "flag_copy",
    srcs = ["flag_copy.py"],
)

flag_per_line(
    name = "flag_per_line_gen",
    out = "flag_per_line_gen.out",
)
EOF

  bazel build package:flag_per_line_gen || fail "Unexpected build failure"
  BINS=$(bazel info $PRODUCT_NAME-bin)
  assert_equals "$(cat $BINS/package/flag_per_line_gen.out)" "pos1
pos2
=== param_file
--foo=bar
--nowoof
--joined=a,b,c
--baz=boo"
}


run_suite "Tests the 'flag_per_line' option of param file generation."
