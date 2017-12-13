#!/bin/bash
#
# Copyright 2017 The Bazel Authors. All rights reserved.
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

function test_proguard() {
  create_new_workspace
  setup_android_sdk_support
  mkdir -p java/com/bin
  cat > java/com/bin/BUILD <<EOF
android_binary(
  name = 'bin',
  srcs = ['Bin.java', 'NotUsed.java'],
  manifest = 'AndroidManifest.xml',
  proguard_specs = ['proguard.config'],
  deps = [':lib'],
)
android_library(
  name = 'lib',
  srcs = ['Lib.java'],
)
EOF
  cat > java/com/bin/AndroidManifest.xml <<EOF
<manifest package='com.bin' />
EOF
  cat > java/com/bin/Bin.java <<EOF
package com.bin;
public class Bin {
  public Lib getLib() {
    return new Lib();
  }
}
EOF
  cat > java/com/bin/NotUsed.java <<EOF
package com.bin;
public class NotUsed {}
EOF
  cat > java/com/bin/Lib.java <<EOF
package com.bin;
public class Lib {}
EOF
  cat > java/com/bin/proguard.config <<EOF
-keep public class com.bin.Bin {
  public *;
}
EOF
  assert_build //java/com/bin
  output_classes=$(zipinfo -1 bazel-bin/java/com/bin/bin_proguard.jar)
  assert_equals 3 $(wc -w <<< $output_classes)
  assert_one_of $output_classes "META-INF/MANIFEST.MF"
  assert_one_of $output_classes "com/bin/Bin.class"
  # Not kept by proguard
  assert_not_one_of $output_classes "com/bin/Unused.class"
  # This is renamed by proguard to something else
  assert_not_one_of $output_classes "com/bin/Lib.class"
}

run_suite "Android integration tests for Proguard"
