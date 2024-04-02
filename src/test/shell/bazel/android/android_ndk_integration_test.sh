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
#   1. Install an Android SDK and NDK from https://developer.android.com
#   2. Set the $ANDROID_HOME and $ANDROID_NDK_HOME environment variables
#   3. Uncomment the two lines in WORKSPACE containing android_sdk_repository
#      and android_ndk_repository
#
# Note that if the environment is not set up as above
# android_ndk_integration_test will silently be ignored and will be shown as
# passing.
#
# Due to clang version constraints, this test only runs with NDK 15 and above.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${CURRENT_DIR}/android_helper.sh" \
  || { echo "android_helper.sh not found!" >&2; exit 1; }
fail_if_no_android_sdk
fail_if_no_android_ndk

source "${CURRENT_DIR}/../../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

resolve_android_toolchains

add_to_bazelrc "build --extra_toolchains=@androidndk//:all"

function create_android_binary() {
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
    hdrs = ["jni_dep.h"],
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
}

function check_num_sos() {
  num_sos=$(unzip -Z1 bazel-bin/java/bazel/bin.apk '*.so' | wc -l | sed -e 's/[[:space:]]//g')
  assert_equals "4" "$num_sos"
}

function check_soname() {
  unzip -p bazel-bin/java/bazel/bin.apk lib/x86/libbin.so > libbin.so
  # For an android_binary with name foo, readelf output format is
  #  Tag        Type          Name/Value
  # 0x00000010 (SONAME)       Library soname: [libfoo]
  #
  # If -Wl,soname= is not set, then SONAME will not appear in the output.
  #
  # readelf is a Linux utility and not available on Mac by default. The NDK
  # includes readelf however the path is difference for Mac vs Linux, hence the
  # star.
  readelf="${TEST_SRCDIR}/androidndk/ndk/toolchains/arm-linux-androideabi-4.9/prebuilt/*/bin/arm-linux-androideabi-readelf"
  soname=$($readelf -d libbin.so \
    | grep SONAME \
    | awk '{print substr($5,2,length($5)-2)}')
  assert_equals "libbin" "$soname"
}

function test_android_binary() {
  create_new_workspace
  setup_android_sdk_support
  setup_android_ndk_support
  create_android_binary

  cpus="armeabi-v7a,arm64-v8a,x86,x86_64"

  bazel build -s //java/bazel:bin \
    --android_platforms=//test_android_platforms:x86,//test_android_platforms:x86_64,//test_android_platforms:armeabi-v7a,//test_android_platforms:arm64-v8a \
     || fail "build failed"
  check_num_sos
  check_soname
}

function test_android_binary_sibling_repository_layout() {
  create_new_workspace
  setup_android_sdk_support
  setup_android_ndk_support
  create_android_binary

  cpus="armeabi-v7a,arm64-v8a,x86,x86_64"

  bazel build --experimental_sibling_repository_layout -s \
      //java/bazel:bin \
      --android_platforms=//test_android_platforms:x86,//test_android_platforms:x86_64,//test_android_platforms:armeabi-v7a,//test_android_platforms:arm64-v8a \
      || fail "build failed"
  check_num_sos
  check_soname
}

function test_android_binary_clang() {
  create_new_workspace
  setup_android_sdk_support
  setup_android_ndk_support
  create_android_binary

  cpus="armeabi-v7a,arm64-v8a,x86,x86_64"

  bazel build -s //java/bazel:bin \
      --android_compiler=clang5.0.300080 \
      --android_platforms=//test_android_platforms:x86,//test_android_platforms:x86_64,//test_android_platforms:armeabi-v7a,//test_android_platforms:arm64-v8a \
      || fail "build failed"
  check_num_sos
  check_soname
}

# Regression test for https://github.com/bazelbuild/bazel/issues/2601.
function test_clang_include_paths() {
  create_new_workspace
  setup_android_sdk_support
  setup_android_ndk_support
  cat > BUILD <<EOF
cc_binary(
    name = "foo",
    srcs = ["foo.cc"],
    copts = ["-mfpu=neon"],
)
EOF
  cat > foo.cc <<EOF
#include <arm_neon.h>
int main() { return 0; }
EOF
  bazel build //:foo \
    --compiler=clang5.0.300080 \
    --cpu=armeabi-v7a \
    || fail "build failed"

  bazel build //:foo \
    --features=compiler_param_file \
    --compiler=clang5.0.300080 \
    --cpu=armeabi-v7a \
    || fail "build failed with --features=compiler_param_file"
}

function test_android_ndk_repository_path_from_environment() {
  create_new_workspace
  setup_android_sdk_support
  setup_android_ndk_support
  cat > WORKSPACE <<EOF
android_ndk_repository(
    name = "androidndk",
    api_level = 25,
)
EOF
  ANDROID_NDK_HOME=$ANDROID_NDK bazel build @androidndk//:files || fail \
    "android_ndk_repository failed to build with \$ANDROID_NDK_HOME instead " \
    "of path"
}

function test_android_ndk_repository_no_path_or_android_ndk_home() {
  create_new_workspace
  cat > WORKSPACE <<EOF
android_ndk_repository(
    name = "androidndk",
    api_level = 25,
)
EOF
  bazel build @androidndk//:files >& $TEST_log && fail "Should have failed"
  expect_log "Either the path attribute of android_ndk_repository"

  # Ensure that the prefix identifies the errant rule.
  # (All errors have this prefix, but we check only in this test.)
  expect_log "WORKSPACE:1:23: fetching android_ndk_repository rule //external:androidndk: Either..."
}

function test_android_ndk_repository_wrong_path() {
  create_new_workspace
  mkdir "$TEST_TMPDIR/some_dir"
  cat > WORKSPACE <<EOF
android_ndk_repository(
    name = "androidndk",
    api_level = 25,
    path = "$TEST_TMPDIR/some_dir",
)
EOF
  bazel build @androidndk//:files >& $TEST_log && fail "Should have failed"
  expect_log "Unable to read the Android NDK at $TEST_TMPDIR/some_dir, the path may be invalid." \
    " Is the path in android_ndk_repository() or \$ANDROID_NDK_HOME set correctly?"
}

function test_stripped_cc_binary() {
  create_new_workspace
  setup_android_sdk_support
  setup_android_ndk_support
  cat > BUILD <<EOF
cc_binary(
    name = "foo",
    srcs = ["foo.cc"],
)
EOF
  cat > foo.cc <<EOF
int main() { return 0; }
EOF
  bazel build //:foo.stripped \
    --cpu=armeabi-v7a \
    || fail "build failed"

  bazel build //:foo.stripped \
    --features=compiler_param_file \
    --cpu=armeabi-v7a \
    || fail "build failed with --features=compiler_param_file"
}

function test_stripped_cc_test() {
  create_new_workspace
  setup_android_sdk_support
  setup_android_ndk_support
  cat > BUILD <<EOF
cc_test(
    name = "foo",
    srcs = ["foo.cc"],
)
EOF
  cat > foo.cc <<EOF
int main() { return 0; }
EOF
  bazel build //:foo.stripped \
    --cpu=armeabi-v7a \
    || fail "build failed"

  bazel build //:foo.stripped \
    --features=compiler_param_file \
    --cpu=armeabi-v7a \
    || fail "build failed with --features=compiler_param_file"
}

function test_platforms_and_toolchains() {
  create_new_workspace
  setup_android_sdk_support
  setup_android_ndk_support
  cat > BUILD <<EOF
cc_binary(
    name = "foo",
    srcs = ["foo.cc"],
    linkopts = ["-ldl", "-lm"],
)
EOF
  cat > foo.cc <<EOF
#include <string>
#include <jni.h>
#include <android/log.h>
#include <cstdio>
#include <iostream>

using namespace std;
int main(){
  string foo = "foo";
  string bar = "bar";
  string foobar = foo + bar;
  return 0;
}
EOF
  assert_build //:foo \
    --cpu=armeabi-v7a

  assert_build //:foo \
    --features=compiler_param_file \
    --cpu=armeabi-v7a
}

run_suite "Android NDK integration tests"
