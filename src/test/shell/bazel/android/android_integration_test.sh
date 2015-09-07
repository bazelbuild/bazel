#!/bin/bash
#
# Copyright 2015 Google Inc. All rights reserved.
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

# Load test environment
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../test-setup.sh \
  || { echo "test-setup.sh not found!" >&2; exit 1; }

function test_android_binary() {
  create_new_workspace
  setup_android_support

  mkdir -p java/bazel
  cat > java/bazel/BUILD <<EOF
android_library(
    name = "lib",
    srcs = ["Lib.java"],
)

android_binary(
    name = "bin",
    srcs = [
        "MainActivity.java",
        "Jni.java",
    ],
    legacy_native_support = 0,
    manifest = "AndroidManifest.xml",
    deps = [
        ":lib",
        ":jni"
    ],
)

cc_library(
    name = "jni",
    srcs = ["jni.cc"],
    deps = [":jni_dep"],
)

cc_library(
    name = "jni_dep",
    srcs = ["jni_dep.cc"],
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

  cat > java/bazel/Lib.java <<EOF
package bazel;

public class Lib {
  public static String message() {
    return "Hello Lib";
  }
}
EOF

  cat > java/bazel/Jni.java <<EOF
package bazel;

public class Jni {
  public static native String hello();
}

EOF
  cat > java/bazel/MainActivity.java <<EOF
package bazel;

import android.app.Activity;
import android.os.Bundle;

public class MainActivity extends Activity {
  @Override
  public void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
  }
}
EOF

  cat > java/bazel/jni_dep.h <<EOF
#pragma once

#include <jni.h>

jstring NewStringLatin1(JNIEnv *env, const char *str);
EOF

  cat > java/bazel/jni_dep.cc <<EOF
#include "java/bazel/jni_dep.h"

#include <stdlib.h>
#include <string.h>

jstring NewStringLatin1(JNIEnv *env, const char *str) {
  int len = strlen(str);
  jchar *str1;
  str1 = reinterpret_cast<jchar *>(malloc(len * sizeof(jchar)));

  for (int i = 0; i < len; i++) {
    str1[i] = (unsigned char)str[i];
  }
  jstring result = env->NewString(str1, len);
  free(str1);
  return result;
}
EOF

  cat > java/bazel/jni.cc <<EOF
#include <jni.h>
#include <string>

#include "java/bazel/jni_dep.h"

extern "C" JNIEXPORT jstring JNICALL
Java_bazel_Jni_hello(JNIEnv *env, jclass clazz) {
  std::string hello = "Hello";
  std::string jni = "JNI";
  return NewStringLatin1(env, (hello + " " + jni).c_str());
}
EOF

  bazel build -s //java/bazel:bin || fail "build failed"
}

if [[ ! -r "${TEST_SRCDIR}/external/androidndk/ndk/RELEASE.TXT" ]]; then
  echo "Not running Android tests due to lack of an Android NDK."
  exit 0
fi

if [[ ! -r "${TEST_SRCDIR}/external/androidsdk/SDK Readme.txt" ]]; then
  echo "Not running Android tests due to lack of an Android SDK."
  exit 0
fi

run_suite "Android integration tests"
