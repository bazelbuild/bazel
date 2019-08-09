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

# For these tests to run do the following:
#
#   1. Install an Android SDK from https://developer.android.com
#   2. Set the $ANDROID_HOME environment variable
#   3. Uncomment the line in WORKSPACE containing android_sdk_repository
#
# Note that if the environment is not set up as above android_integration_test
# will silently be ignored and will be shown as passing.

# --- begin runfiles.bash initialization v2 ---
# Copy-pasted from the Bazel Bash runfiles library v2.
set -uo pipefail; f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
  source "$0.runfiles/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  { echo>&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v2 ---

cat $(rlocation MANIFEST) | grep integration_test_setup

source "$(rlocation io_bazel/src/test/shell/bazel/android/android_helper.sh)" \
  || { echo "android_helper.sh not found!" >&2; exit 1; }
fail_if_no_android_sdk

source "$(rlocation io_bazel/src/test/shell/integration_test_setup.sh)" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# TODO(#8169): Make this test compatible with Python toolchains. Blocked on the
# fact that there's no PY3 environment on our Mac workers
# (bazelbuild/continuous-integration#578).
add_to_bazelrc "build --incompatible_use_python_toolchains=false"

function test_build_with_aapt() {
  create_new_workspace
  setup_android_sdk_support
  create_android_binary

  assert_build //java/bazel:bin --android_aapt=aapt
}

function test_build_with_aapt2() {
  create_new_workspace
  setup_android_sdk_support
  create_android_binary

  assert_build //java/bazel:bin --android_aapt=aapt2
}

function test_build_with_aapt2_skip_parsing_action() {
  create_new_workspace
  setup_android_sdk_support
  create_android_binary

  assert_build //java/bazel:bin \
    --android_aapt=aapt2 \
    --experimental_skip_parsing_action
}

function tear_down() {
  # Saves 10 seconds per test on Windows by properly shutting down the server to release
  # a lock on a temp file for deletion. e.g. without bazel shutdown:
  # INFO[aapt_integration_test 2019-08-08 21:37:30 (-0400)] Cleaning up workspace
  # rm: cannot remove
  # 'C:/users/user/_bazel_user/wqlyjfgb/execroot/io_bazel/_tmp/778a66baaf866926175b5b5753acaf30/workspace.INxSD1eY':
  # Device or resource busy INFO[aapt_integration_test 2019-08-08 21:37:43
  # (-0400)] try_with_timeout(rm -fr
  # C:/users/user/_bazel_user/wqlyjfgb/execroot/io_bazel/_tmp/778a66baaf866926175b5b5753acaf30/workspace.INxSD1eY):
  # no success after 10 seconds (timeout in 110 seconds) rm: cannot remove
  # 'C:/users/user/_bazel_user/wqlyjfgb/execroot/io_bazel/_tmp/778a66baaf866926175b5b5753acaf30/workspace.INxSD1eY':
  # Device or resource busy

  bazel shutdown
}
run_suite "aapt/aapt2 integration tests"
