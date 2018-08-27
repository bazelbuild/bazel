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

function setup_font_resources() {
  rm java/bazel/BUILD

  cat > java/bazel/BUILD <<EOF
package(default_visibility = ["//visibility:public"])
aar_import(
    name = "aar",
    aar = "sample.aar",
)
android_library(
    name = "lib",
    srcs = ["Lib.java"],
    deps = [":aar"],
)
android_binary(
    name = "bin",
    srcs = ["MainActivity.java"],
    resource_files = glob(["res/**"]),
    manifest = "AndroidManifest.xml",
    deps = [":lib"],
)
EOF
  mkdir -p java/bazel/res/font
  cp "$TEST_SRCDIR/io_bazel/src/test/shell/bazel/android/testdata/roboto.ttf" \
    java/bazel/res/font/

  mkdir -p java/bazel/res/values
  cat > java/bazel/res/values/styles.xml <<EOF
<?xml version="1.0" encoding="utf-8"?>
<resources>
  <style name="AppTheme">
      <item name="android:fontFamily">@font/roboto</item>
  </style>
</resources>
EOF

  cat > java/bazel/AndroidManifest.xml <<EOF
<manifest
    xmlns:android="http://schemas.android.com/apk/res/android"
    package="bazel.android">
    <application
        android:label="Bazel App"
        android:theme="@style/AppTheme" >
        <activity
            android:name="bazel.MainActivity"
            android:label="Bazel" />
    </application>
</manifest>
EOF
}

function test_font_support() {
  create_new_workspace
  setup_android_sdk_support
  create_android_binary
  setup_font_resources

  assert_build //java/bazel:bin
}

function test_persistent_resource_processor_aapt() {
  create_new_workspace
  setup_android_sdk_support
  create_android_binary
  setup_font_resources

  assert_build //java/bazel:bin --persistent_android_resource_processor
}

function test_persistent_resource_processor_aapt2() {
  create_new_workspace
  setup_android_sdk_support
  create_android_binary
  setup_font_resources

  assert_build //java/bazel:bin --persistent_android_resource_processor --android_aapt=aapt2
}

run_suite "Resource processing integration tests"
