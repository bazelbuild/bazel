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

function fail_if_no_android_sdk() {
  if [[ ! -d "${TEST_SRCDIR}/androidsdk" ]]; then
    echo "Not running Android tests due to lack of an Android SDK."
    exit 1
  fi
}

function fail_if_no_android_ndk() {
  # ndk r10 and earlier
  if [[ ! -r "${TEST_SRCDIR}/androidndk/ndk/RELEASE.TXT" ]]; then
    # ndk r11 and later
    if [[ ! -r "${TEST_SRCDIR}/androidndk/ndk/source.properties" ]]; then
      echo "Not running Android NDK tests due to lack of an Android NDK."
      exit 1
    fi
  fi
}

function create_android_binary() {
  mkdir -p java/bazel
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
    manifest = "AndroidManifest.xml",
    deps = [":lib"],
)
EOF

  cp "${TEST_SRCDIR}/io_bazel/src/test/shell/bazel/android/sample.aar" \
    java/bazel/sample.aar
  cat > java/bazel/AndroidManifest.xml <<EOF
  <manifest package="bazel.android" />
EOF

  cat > java/bazel/Lib.java <<EOF
package bazel;
import com.sample.aar.Sample;
public class Lib {
  public static String message() {
  return "Hello Lib" + Sample.getZero();
  }
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
}
