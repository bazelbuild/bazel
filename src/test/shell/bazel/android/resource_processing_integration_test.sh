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

source "$(rlocation io_bazel/src/test/shell/bazel/android/android_helper.sh)" \
  || { echo "android_helper.sh not found!" >&2; exit 1; }
fail_if_no_android_sdk

source "$(rlocation io_bazel/src/test/shell/integration_test_setup.sh)" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

resolve_android_toolchains "$1"

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
  cp "$(rlocation io_bazel/src/test/shell/bazel/android/testdata/roboto.ttf)" \
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
    xmlns:tools="http://schemas.android.com/tools"
    package="bazel.android">

    <!-- tools:replace is a deprecated attribute, and will cause the manifest merger action to log a warning -->
    <uses-feature android:name="android.hardware.location" android:required="false" tools:replace="android:required" />

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

function test_persistent_resource_processor() {
  create_new_workspace
  setup_android_sdk_support
  create_android_binary
  setup_font_resources

  assert_build //java/bazel:bin --persistent_android_resource_processor \
    --worker_verbose &> $TEST_log
  expect_log "Created new non-sandboxed AndroidResourceParser worker (id [0-9]\+, key hash -\?[0-9]\+)"
  expect_log "Created new non-sandboxed AndroidResourceCompiler worker (id [0-9]\+, key hash -\?[0-9]\+)"
  expect_log "Created new non-sandboxed AndroidCompiledResourceMerger worker (id [0-9]\+, key hash -\?[0-9]\+)"
  expect_log "Created new non-sandboxed AndroidAapt2 worker (id [0-9]\+, key hash -\?[0-9]\+)"
  expect_log "Created new non-sandboxed ManifestMerger worker (id [0-9]\+, key hash -\?[0-9]\+)"
}

function test_persistent_multiplex_resource_processor() {
  create_new_workspace
  setup_android_sdk_support
  create_android_binary
  setup_font_resources

  assert_build //java/bazel:bin --worker_multiplex \
    --persistent_multiplex_android_tools \
    --worker_verbose &> $TEST_log
  expect_log "Created new non-sandboxed AndroidResourceParser multiplex-worker (id [0-9]\+, key hash -\?[0-9]\+)"
  expect_log "Created new non-sandboxed AndroidResourceCompiler multiplex-worker (id [0-9]\+, key hash -\?[0-9]\+)"
  expect_log "Created new non-sandboxed AndroidCompiledResourceMerger multiplex-worker (id [0-9]\+, key hash -\?[0-9]\+)"
  expect_log "Created new non-sandboxed AndroidAapt2 multiplex-worker (id [0-9]\+, key hash -\?[0-9]\+)"
  expect_log "Created new non-sandboxed ManifestMerger multiplex-worker (id [0-9]\+, key hash -\?[0-9]\+)"
}

run_suite "Resource processing integration tests"
