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

function fail_if_no_android_sdk() {
  # Required for runfiles library on Windows, since $(rlocation) lookups
  # can't do directories. We use android-28's android.jar as the anchor
  # for the androidsdk location.
  android_sdk_anchor=$(rlocation androidsdk/platforms/android-28/android.jar)
  if [[ ! -r "$android_sdk_anchor" ]]; then
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

  cp "$(rlocation io_bazel/src/test/shell/bazel/android/sample.aar)" \
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

function setup_head_android_tools_if_exists() {
  local head_android_tools=$(rlocation io_bazel/tools/android/runtime_deps/android_tools.tar)
  if [[ -f $head_android_tools ]]; then
    HEAD_ANDROID_TOOLS_WS="$TEST_TMPDIR/head_android_tools"
    mkdir "$HEAD_ANDROID_TOOLS_WS"
    tar xvf $head_android_tools -C "$HEAD_ANDROID_TOOLS_WS"
    echo "common --override_repository=android_tools=$HEAD_ANDROID_TOOLS_WS" >> $TEST_TMPDIR/bazelrc
  fi
}

# Resolves Android toolchains with platforms.
function resolve_android_toolchains() {
  add_to_bazelrc "build --android_platforms=//test_android_platforms:simple"
}

setup_head_android_tools_if_exists
