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

# To run this test, ensure that //external:android_sdk_for_testing is set to
# the @androidsdk//:files filegroup created by the AndroidSdkRepositoryFunction.
# If this is not set, this test will silently pass so as to prevent compile.sh
# from failing for developers without an Android SDK. See the BUILD file for
# more details.

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

function create_java_8_android_binary() {
  mkdir -p java/bazel
  cat > java/bazel/BUILD <<EOF
android_binary(
    name = "bin",
    srcs = ["MainActivity.java"],
    manifest = "AndroidManifest.xml",
)
EOF

  cat > java/bazel/AndroidManifest.xml <<EOF
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="bazel.android"
    android:versionCode="1"
    android:versionName="1.0" >

    <uses-sdk
        android:minSdkVersion="21"
        android:targetSdkVersion="21" />

    <application
        android:label="Bazel Test App" >
        <activity
            android:name="bazel.MainActivity"
            android:label="Bazel" >
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>
EOF

  cat > java/bazel/MainActivity.java <<EOF
package bazel;
import android.app.Activity;
import java.util.stream.Stream;
import java.util.Arrays;

public class MainActivity extends Activity {
  interface A {
    int foo(int x, int y);
  }

  A bar() {
    return (a, b) -> a * b;
  }

  int someHashcode() {
    // JDK 8 language feature depending on primitives desugar
    return java.lang.Long.hashCode(42L);
  }

  int getSumOfInts() {
    // JDK 8 language feature depending on streams desugar
    return Arrays
      .asList("x1", "x2", "x3")
      .stream()
      .map(s -> s.substring(1))
      .mapToInt(Integer::parseInt)
      .sum();
  }
}
EOF
}

function test_java_8_android_binary() {
  create_new_workspace
  setup_android_sdk_support
  create_java_8_android_binary

  # Test desugar in sandboxed mode, or fallback to standalone for Windows.
  bazel build \
   --strategy=Desugar=sandboxed \
   --desugar_for_android //java/bazel:bin \
      || fail "build failed"
}

function test_java_8_android_binary_worker_strategy() {
  create_new_workspace
  setup_android_sdk_support
  create_java_8_android_binary

  assert_build //java/bazel:bin \
    --persistent_android_dex_desugar \
    --worker_verbose &> $TEST_log
  expect_log "Created new non-sandboxed Desugar worker (id [0-9]\+, key hash -\?[0-9]\+)"
  expect_log "Created new non-sandboxed DexBuilder worker (id [0-9]\+, key hash -\?[0-9]\+)"
}

function test_java_8_android_binary_multiplex_worker_strategy() {
  create_new_workspace
  setup_android_sdk_support
  create_java_8_android_binary

  assert_build //java/bazel:bin \
    --worker_multiplex \
    --persistent_multiplex_android_dex_desugar \
    --worker_verbose &> $TEST_log
  expect_log "Created new non-sandboxed Desugar multiplex-worker (id [0-9]\+, key hash -\?[0-9]\+)"
  expect_log "Created new non-sandboxed DexBuilder multiplex-worker (id [0-9]\+, key hash -\?[0-9]\+)"
}

run_suite "Android desugarer integration tests"
