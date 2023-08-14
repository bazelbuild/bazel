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

resolve_android_toolchains "$1"

# Regression test for https://github.com/bazelbuild/bazel/issues/1928.
function test_empty_tree_artifact_action_inputs_mount_empty_directories() {
  create_new_workspace
  setup_android_sdk_support
  cat > AndroidManifest.xml <<EOF
<manifest package="com.test"/>
EOF
  mkdir res
  zip test.aar AndroidManifest.xml res/
  cat > BUILD <<EOF
aar_import(
  name = "test",
  aar = "test.aar",
)
EOF
  # Building aar_import invokes the AndroidResourceProcessingAction with a
  # TreeArtifact of the AAR resources as the input. Since there are no
  # resources, the Bazel sandbox should create an empty directory. If the
  # directory is not created, the action thinks that its inputs do not exist and
  # crashes.
  bazel build :test
}

function test_nonempty_aar_resources_tree_artifact() {
  create_new_workspace
  setup_android_sdk_support
  cat > AndroidManifest.xml <<EOF
<manifest package="com.test"/>
EOF
  mkdir -p res/values
  cat > res/values/values.xml <<EOF
<?xml version="1.0" encoding="utf-8"?>
<resources xmlns:android="http://schemas.android.com/apk/res/android">
</resources>
EOF
  zip test.aar AndroidManifest.xml res/values/values.xml
  cat > BUILD <<EOF
aar_import(
  name = "test",
  aar = "test.aar",
)
EOF
  bazel build :test
}

function test_android_binary_depends_on_aar() {
  create_new_workspace
  setup_android_sdk_support
  cat > AndroidManifest.xml <<EOF
<manifest package="com.example"/>
EOF
  mkdir -p res/layout
  cat > res/layout/mylayout.xml <<EOF
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android" />
EOF
  mkdir assets
  echo "some asset" > assets/a
  zip example.aar AndroidManifest.xml res/layout/mylayout.xml assets/a
  cat > BUILD <<EOF
aar_import(
  name = "example",
  aar = "example.aar",
)
android_binary(
  name = "app",
  custom_package = "com.example",
  manifest = "AndroidManifest.xml",
  deps = [":example"],
)
EOF
  assert_build :app
  apk_contents="$(zipinfo -1 bazel-bin/app.apk)"
  assert_one_of $apk_contents "assets/a"
  assert_one_of $apk_contents "res/layout/mylayout.xml"
}

function test_android_binary_fat_apk_contains_all_shared_libraries() {
  create_new_workspace
  setup_android_sdk_support
  setup_android_ndk_support

  # TODO(b/161709111): enable platform-based toolchain resolution when
  # --fat_apk_cpu fully supports it. Now it sets a split transition that clears
  # out --platforms. The mapping in android_helper.sh re-enables a test Android
  # platform for ARM but not x86. Enabling it for x86 requires an
  # Android-compatible cc toolchain in tools/cpp/BUILD.tools.
  add_to_bazelrc "build --noincompatible_enable_android_toolchain_resolution"

  # sample.aar contains native shared libraries for x86 and armeabi-v7a
  cp "$(rlocation io_bazel/src/test/shell/bazel/android/sample.aar)" .
  cat > AndroidManifest.xml <<EOF
<manifest package="com.example"/>
EOF
  cat > BUILD <<EOF
aar_import(
  name = "sample",
  aar = "sample.aar",
)
android_binary(
  name = "app",
  custom_package = "com.example",
  manifest = "AndroidManifest.xml",
  deps = [":sample"],
)
EOF
  assert_build :app --fat_apk_cpu=x86,armeabi-v7a
  apk_contents="$(zipinfo -1 bazel-bin/app.apk)"
  assert_one_of $apk_contents "lib/x86/libapp.so"
  assert_one_of $apk_contents "lib/armeabi-v7a/libapp.so"
}

run_suite "aar_import integration tests"
