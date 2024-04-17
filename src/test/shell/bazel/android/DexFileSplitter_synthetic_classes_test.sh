#!/bin/bash
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

source "${CURRENT_DIR}/../../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

source "${CURRENT_DIR}/android_helper.sh" \
  || { echo "android_helper.sh not found!" >&2; exit 1; }
fail_if_no_android_sdk

resolve_android_toolchains

function create_test_app() {

  inner_class_count="$1"
  lambda_count="$2"
  dex_shard_count="$3"

  mkdir -p java/com/testapp

  cat > java/com/testapp/AndroidManifest.xml <<EOF
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.testapp"
    android:versionCode="1"
    android:versionName="1.0" >

    <uses-sdk
        android:minSdkVersion="30"
        android:targetSdkVersion="30" />

    <application android:label="Test App" >
        <activity
            android:name="com.testapp.MainActivity"
            android:label="App"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>
EOF

  cat > java/com/testapp/MainActivity.java <<EOF
package com.testapp;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;

public class MainActivity extends Activity {
  @Override
  public void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    Log.i("tag", "info");
  }
}
EOF

  generate_java_file_with_many_synthetic_classes "$1" "$2" > java/com/testapp/BigLib.java

  cat > java/com/testapp/BUILD <<EOF
android_binary(
    name = "testapp",
    srcs = [
        "MainActivity.java",
        ":BigLib.java",
    ],
    dex_shards = $dex_shard_count,
    manifest = "AndroidManifest.xml",
)
EOF
}

function generate_java_file_with_many_synthetic_classes() {

  inner_class_count="$1"
  lambda_count="$2"

  echo "package com.testapp;"
  echo "public class BigLib {"

  # First generate enough inner classes to fill up most of the dex
  for (( i = 0; i < $inner_class_count; i++ )) do

    echo "  public static class Bar$i {"
    echo "    public int bar() {"
    echo "      return $i;"
    echo "    }"
    echo "  }"

  done

  # Then create enough synethetic classes via lambdas to fill up the rest of the
  # dex and into another dex file.
  echo "  public interface IntSupplier {"
  echo "    int supply();"
  echo "  }"

  echo "  public static class Foo {"
  echo "    public IntSupplier[] foo() {"
  echo "      return new IntSupplier[] {"

  for ((i = 0; i < $lambda_count; i++ )) do
    echo "        () -> $i,"
  done

  echo "      };"
  echo "    }"
  echo "    public IntSupplier[] bar() {"
  echo "      return new IntSupplier[] {"

  for ((i = 0; i < $lambda_count; i++ )) do
    echo "        () -> $i,"
  done

  echo "      };"
  echo "    }"
  echo "  }"
  echo "}"
}

function test_DexFileSplitter_synthetic_classes_crossing_dexfiles() {
  create_new_workspace
  setup_android_sdk_support

  # dex_shards default is 1
  create_test_app 21400 6000 1

  bazel build java/com/testapp || fail "Test app should have built succesfully"

  dex_file_count=$(unzip -l bazel-bin/java/com/testapp/testapp.apk | grep "classes[0-9]*.dex" | wc -l)
  if [[ ! "$dex_file_count" -ge "2" ]]; then
    echo "Expected at least 2 dexes in app, found: $dex_file_count"
    exit 1
  fi
}

function test_DexMapper_synthetic_classes_crossing_dexfiles() {
  create_new_workspace
  setup_android_sdk_support

  # 3 inner classes, 6 lambdas (and thus 6 synthetics from D8) and 5 dex_shards
  # is one magic combination to repro synthetics being separated from their
  # context / enclosing classes.
  create_test_app 3 6 5

  echo here2
  echo $TEST_TMPDIR/bazelrc
  cat $TEST_TMPDIR/bazelrc

  bazel build java/com/testapp || fail "Test app should have built succesfully"

  dex_file_count=$(unzip -l bazel-bin/java/com/testapp/testapp.apk | grep "classes[0-9]*.dex" | wc -l)
  if [[ ! "$dex_file_count" -eq "4" ]]; then
    echo "Expected 4 dexes in app, found: $dex_file_count"
    exit 1
  fi
}

run_suite "Tests for DexFileSplitter with synthetic classes crossing dexfiles"