#!/bin/bash
#
# Copyright 2020 The Bazel Authors. All rights reserved.
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
msys*)
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

if "$is_windows"; then
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

# ------------------------------------------------------------------------------
# TESTS
# ------------------------------------------------------------------------------

# Regression test for https://github.com/bazelbuild/bazel/issues/10621
function test_running_test_target_with_runfiles_disabled() {
  local -r pkg="pkg${LINENO}"
  mkdir $pkg
  cat >$pkg/BUILD <<eof
load(":my_test.bzl", "my_test")
my_test(name = "x")
eof
  if "$is_windows"; then
    local -r IsWindows=True
  else
    local -r IsWindows=False
  fi
  cat >$pkg/my_test.bzl <<eof
def _impl(ctx):
    # Extension needs to be ".bat" on Windows, so that Bazel can run the test's
    # binary (the script) as a direct child process.
    # Extension doesn't matter on other platforms.
    out = ctx.actions.declare_file("%s.bat" % ctx.label.name)
    ctx.actions.write(
        output = out,
        content = "@echo my_test" if $IsWindows else "#!/bin/bash\necho my_test\n",
        is_executable = True,
    )
    return [DefaultInfo(executable = out,
                        files = depset([out]),
                        default_runfiles = ctx.runfiles([out]))]

my_test = rule(
    implementation = _impl,
    test = True,
)
eof
  bazel run --enable_runfiles=no $pkg:x >&$TEST_log || fail "expected success"
  bazel run --enable_runfiles=yes $pkg:x >&$TEST_log || fail "expected success"
}

function test_windows_argument_escaping() {
  if ! "$is_windows"; then
    return # Run test only on Windows.
  fi

  local -r pkg="pkg${LINENO}"
  mkdir $pkg
  cat >$pkg/BUILD <<eof
sh_binary(
  name = "a",
  srcs = [":a.sh"],
)
eof
  cat >$pkg/a.sh <<eof
echo Hello World
eof
  disable_errexit
  # This test uses the content of the Bazel error message to test that Bazel correctly handles
  # paths to Bash which contain spaces - which is needed when using --run_under as it
  # unconditionally results in Bazel wrapping the executable in Bash.
  #
  # The expected error message starts with:
  # ```
  # FATAL: ExecuteProgram(C:\first_part second_part) failed: ERROR: s
  # rc/main/native/windows/process.cc(202): CreateProcessW("C:\first_
  # part second_part"
  # ```
  output="$(BAZEL_SH="C:/first_part second_part" bazel run --run_under=":;" $pkg:a 2>&1)"
  enable_errexit
  echo "$output" | grep --fixed-strings 'ExecuteProgram(C:\first_part second_part)' || fail "Expected error message to contain unquoted path"
}

run_suite "run_under_tests"
