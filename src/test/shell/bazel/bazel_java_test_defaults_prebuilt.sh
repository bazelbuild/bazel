#!/usr/bin/env bash
#
# Copyright 2023 The Bazel Authors. All rights reserved.
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
# Tests Java prebuilt toolchains.
#

set -euo pipefail

# --- begin runfiles.bash initialization ---
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


# PREBUILT_TOOLCHAIN_CONFIGURATION shall use prebuilt ijar and singlejar binaries.
function test_default_java_toolchain_prebuiltToolchain() {
  add_rules_java "MODULE.bazel"
  cat > BUILD <<EOF
load("@rules_java//toolchains:default_java_toolchain.bzl", "default_java_toolchain", "PREBUILT_TOOLCHAIN_CONFIGURATION")
default_java_toolchain(
  name = "prebuilt_toolchain",
  configuration = PREBUILT_TOOLCHAIN_CONFIGURATION,
)
EOF

  bazel build //:prebuilt_toolchain || fail "default_java_toolchain target failed to build"
  bazel cquery 'deps(//:prebuilt_toolchain)' >& $TEST_log || fail "failed to query //:prebuilt_toolchain"

  expect_log "ijar/ijar\(.exe\)\? "
  expect_log "singlejar/singlejar_local"
  expect_not_log "ijar/ijar.cc"
  expect_not_log "singlejar/singlejar_main.cc"
}

run_suite "Java prebuilt toolchains tests."
