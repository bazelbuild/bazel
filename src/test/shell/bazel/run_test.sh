#!/usr/bin/env bash
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

  add_rules_shell "MODULE.bazel"
  local -r pkg="pkg${LINENO}"
  mkdir $pkg
  cat >$pkg/BUILD <<eof
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")
sh_binary(
  name = "a",
  srcs = [":a.sh"],
)
eof
  cat >$pkg/a.sh <<eof
echo Hello World
eof
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
  output="$(BAZEL_SH="C:/first_part second_part" bazel run --run_under=":;" $pkg:a 2>&1 || true)"
  echo "$output" | grep --fixed-strings 'ExecuteProgram(C:\first_part second_part)' || fail "Expected error message to contain unquoted path"
}

function test_run_with_runfiles_env() {
  add_rules_shell "MODULE.bazel"
  mkdir -p b
  cat > b/BUILD <<'EOF'
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")
sh_binary(
  name = "binary",
  srcs = ["binary.sh"],
  deps = ["@bazel_tools//tools/bash/runfiles"],
)
EOF
  cat > b/binary.sh <<'EOF'
#!/usr/bin/env bash
# --- begin runfiles.bash initialization v3 ---
# Copy-pasted from the Bazel Bash runfiles library v3.
set -uo pipefail; set +e; f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
  source "$0.runfiles/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  { echo>&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v3 ---

own_path=$(rlocation _main/b/binary.sh)
echo "own path: $own_path"
test -f "$own_path"
EOF
  chmod +x b/binary.sh

  bazel run //b:binary --script_path=script.bat &>"$TEST_log" \
    || fail "Script generation should succeed"

  cat ./script.bat &>"$TEST_log"

  # Make it so that the runfiles variables point to an incorrect but valid
  # runfiles directory/manifest, simulating a left over one from a different
  # test to which RUNFILES_DIR and RUNFILES_MANIFEST_FILE point in the client
  # env.
  BOGUS_RUNFILES_DIR="$(pwd)/bogus_runfiles/bazel_tools/tools/bash/runfiles"
  mkdir -p "$BOGUS_RUNFILES_DIR"
  touch "$BOGUS_RUNFILES_DIR/runfiles.bash"
  BOGUS_RUNFILES_MANIFEST_FILE="$(pwd)/bogus_manifest"
  echo "bazel_tools/tools/bash/runfiles/runfiles.bash bogus/path" > "$BOGUS_RUNFILES_MANIFEST_FILE"

  RUNFILES_DIR="$BOGUS_RUNFILES_DIR" RUNFILES_MANIFEST_FILE="$BOGUS_RUNFILES_MANIFEST_FILE" \
     ./script.bat || fail "Run should succeed"
}

function test_run_test_exit_code() {
  add_rules_shell "MODULE.bazel"
  mkdir -p foo
  cat > foo/BUILD <<'EOF'
load("@rules_shell//shell:sh_test.bzl", "sh_test")

sh_test(
  name = "exit0",
  srcs = ["exit0.sh"],
)

sh_test(
  name = "exit1",
  srcs = ["exit1.sh"],
)
EOF

  cat > foo/exit0.sh <<'EOF'
set -x
exit 0
EOF
  chmod +x foo/exit0.sh
  bazel run //foo:exit0 &>"$TEST_log" \
    || fail "Expected exit code 0, received $?"

  cat > foo/exit1.sh <<'EOF'
set -x
exit 1
EOF
  chmod +x foo/exit1.sh
  bazel run //foo:exit1 &>"$TEST_log" \
    && fail "Expected exit code 1, received $?"

  # Avoid failing the test because of the last non-zero exit-code.
  true
}

run_suite "run_under_tests"
