#!/bin/bash
#
# Copyright 2018 The Bazel Authors. All rights reserved.
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
msys*|mingw*|cygwin*)
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

add_to_bazelrc "build --package_path=%workspace%"

function test_basic_aquery() {
  local pkg="${FUNCNAME[0]}"
  mkdir -p "$pkg" || fail "mkdir -p $pkg"
  cat > "$pkg/BUILD" <<'EOF'
genrule(
    name = "foo",
    srcs = [":bar"],
    outs = ["foo_out.txt"],
    cmd = "cat $(SRCS) > $(OUTS)",
)

genrule(
    name = "bar",
    srcs = ["dummy.txt"],
    outs = ["bar_out.txt"],
    cmd = "echo unused > $(OUTS)",
)
EOF
  echo "hello aquery" > "$pkg/in.txt"

  bazel aquery "//$pkg:foo" > output 2> "$TEST_log" || fail "Expected success"
  assert_contains "//$pkg:foo" output
  assert_not_contains "//$pkg:bar" output

  bazel aquery "deps(//$pkg:foo)" > output 2> "$TEST_log" \
    || fail "Expected success"
  assert_contains "//$pkg:foo" output
  assert_contains "//$pkg:bar" output
}

function test_aquery_text() {
  local pkg="${FUNCNAME[0]}"
  mkdir -p "$pkg" || fail "mkdir -p $pkg"
  cat > "$pkg/BUILD" <<'EOF'
genrule(
    name = "bar",
    srcs = ["dummy.txt"],
    outs = ["bar_out.txt"],
    cmd = "echo unused > $(OUTS)",
)
EOF
  echo "hello aquery" > "$pkg/in.txt"

  bazel aquery --output=text "//$pkg:bar" > output 2> "$TEST_log" \
    || fail "Expected success"
  cat output >> "$TEST_log"
  assert_contains "action 'Executing genrule //$pkg:bar'" output
  assert_contains "Mnemonic: Genrule" output
  assert_contains "Owner: //$pkg:bar" output
  assert_contains "Configuration: .*-fastbuild" output
  # Only check that the inputs/outputs/command line/environment exist, but not
  # their actual contents since that would be too much.
  assert_contains "Inputs: \[" output
  assert_contains "Outputs: \[" output
  if is_windows; then
    assert_contains "Command Line: .*bash\.exe" output
  else
    assert_contains "Command Line: (" output
  fi
  assert_contains "Environment: \[" output

}

run_suite "${PRODUCT_NAME} action graph query tests"
