#!/bin/bash

#
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

if [[ "$1" = '--with_platforms' ]]; then
  # TODO(b/161709111): With platforms, the below fails with
  # "no attribute `$android_sdk_toolchain_type`" on AspectAwareAttributeMapper.
  echo "android_local_test_integration_test.sh does not support --with_platforms!" >&2
  exit 0
fi

function setup_android_local_test_env() {
  mkdir -p java/com/bin/res/values
  mkdir -p javatests/com/bin

  # Targets for android_local_test to depend on
  cat > java/com/bin/BUILD <<EOF
android_library(
  name = 'lib',
  manifest = 'AndroidManifest.xml',
  exports_manifest = 0,
  srcs = ['Bar.java'],
  visibility = ["//visibility:public"],
)
EOF
  cat > java/com/bin/AndroidManifest.xml <<EOF
<?xml version="1.0" encoding="utf-8"?>
<manifest package='com.bin' xmlns:android="http://schemas.android.com/apk/res/android" />
EOF
  cat > java/com/bin/Bar.java <<EOF
package com.bin;
public class Bar { }
EOF
  cat > java/com/bin/res/values/values.xml <<EOF
<?xml version="1.0" encoding="utf-8"?>
<resources xmlns:android="http://schemas.android.com/apk/res/android">
</resources>
EOF

  # android_local_test targets
  cat > javatests/com/bin/robolectric-deps.properties <<EOF
EOF

  cat > javatests/com/bin/BUILD <<EOF
java_library(
  name = 'robolectric-deps',
  data = ['robolectric-deps.properties'],
)
android_local_test(
  name = 'test',
  srcs = ['BarTest.java'],
  manifest = 'AndroidManifest.xml',
  test_class = "com.bin.BarTest",
  deps = ['//java/com/bin:lib', ':robolectric-deps'],
)
EOF
  cat > javatests/com/bin/AndroidManifest.xml <<EOF
<?xml version="1.0" encoding="utf-8"?>
<manifest package='com.bin.test' xmlns:android='http://schemas.android.com/apk/res/android'>
</manifest>
EOF
  cat > javatests/com/bin/BarTest.java <<EOF
package com.bin;

import org.junit.Test;

public class BarTest {
  @Test
  public void testBar() {
    new Bar();
  }
}
EOF
}

# Asserts that the coverage file exists by looking at the current $TEST_log.
# The method fails if TEST_log does not contain any coverage report for a passed test.
function assert_coverage_file_exists() {
  local ending_part="$(sed -n -e '/PASSED/,$p' "$TEST_log")"
  local coverage_file_path=$(grep -Eo "/[/a-zA-Z0-9\.\_\-]+\.dat$" <<< "$ending_part")
  [[ -e "$coverage_file_path" ]] || fail "Coverage output file does not exist!"
}

function test_hello_world_android_local_test() {
  create_new_workspace
  setup_android_sdk_support
  setup_android_local_test_env

  bazel clean
  bazel test --test_output=all \
    //javatests/com/bin:test &>$TEST_log || fail "Tests for //javatests/com/bin:test failed"
}

function test_hello_world_android_local_test_with_coverage() {
  create_new_workspace
  setup_android_sdk_support
  setup_android_local_test_env

  bazel clean
  bazel coverage --test_output=all \
    //javatests/com/bin:test &>$TEST_log || fail "Test with coverage for //javatests/com/bin:test failed"

  assert_coverage_file_exists
}

run_suite "android_local_test integration tests"
