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

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${CURRENT_DIR}/android_helper.sh" \
  || { echo "android_helper.sh not found!" >&2; exit 1; }
fail_if_no_android_sdk

source "${CURRENT_DIR}/../../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

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

run_suite "aapt/aapt2 integration tests"
