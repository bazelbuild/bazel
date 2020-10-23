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

source "$(rlocation io_bazel/src/test/shell/bazel/android/android_helper.sh)" \
  || { echo "android_helper.sh not found!" >&2; exit 1; }
fail_if_no_android_sdk

source "$(rlocation io_bazel/src/test/shell/integration_test_setup.sh)" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function test_simple_idl_srcs() {
  create_new_workspace
  setup_android_sdk_support

  mkdir -p java/com/example
  cat > java/com/example/BUILD <<EOF
android_library(
    name = "lib",
    srcs = ["Lib.java"],
    idl_srcs = ["ILib.aidl"],
)
android_binary(
    name = "bin",
    manifest = "AndroidManifest.xml",
    srcs = ["Bin.java"],
    deps = [":lib"],
)
EOF
  cat > java/com/example/AndroidManifest.xml <<EOF
<manifest xmlns:android='http://schemas.android.com/apk/res/android' package='com.example' />
EOF
  cat > java/com/example/ILib.aidl <<EOF
package com.example;
interface ILib {}
EOF
  cat > java/com/example/Lib.java <<EOF
package com.example;
import android.os.IBinder;
class Lib implements ILib {
  @Override
  public IBinder asBinder() {
    return null;
  }
}
EOF
  cat > java/com/example/Bin.java <<EOF
package com.example;
public class Bin {
  ILib lib = new Lib();
}
EOF

  bazel build -s --verbose_failures //java/com/example:bin \
    || fail "build failed"
}

run_suite "Android IDL tests"
