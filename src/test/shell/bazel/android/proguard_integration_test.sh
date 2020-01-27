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

function test_proguard() {
  create_new_workspace
  setup_android_sdk_support
  mkdir -p java/com/bin
  cat > java/com/bin/BUILD <<EOF
android_binary(
  name = 'bin',
  srcs = ['Bin.java', 'NotUsed.java', 'Renamed.java', 'Filtered.java'],
  manifest = 'AndroidManifest.xml',
  proguard_specs = ['proguard.config'],
  deps = [':lib'],
)
android_library(
  name = 'lib',
  srcs = ['Lib.java', 'Filtered.java'],
  neverlink = 1,
)
EOF
  cat > java/com/bin/AndroidManifest.xml <<EOF
<manifest xmlns:android='http://schemas.android.com/apk/res/android' package='com.bin' />
EOF
  cat > java/com/bin/Bin.java <<EOF
package com.bin;
public class Bin {
  public Renamed getLib() {
    return new Renamed();
  }
}
EOF
  cat > java/com/bin/Filtered.java <<EOF
package com.bin;
public class Filtered {}
EOF
  cat > java/com/bin/NotUsed.java <<EOF
package com.bin;
public class NotUsed {}
EOF
  cat > java/com/bin/Renamed.java <<EOF
package com.bin;
public class Renamed {}
EOF
  cat > java/com/bin/Lib.java <<EOF
package com.bin;
public class Lib {}
EOF
  cat > java/com/bin/proguard.config <<EOF
-keep public class com.bin.Bin {
  public *;
}
EOF
  bazel build --experimental_filter_library_jar_with_program_jar -s --verbose_failures //java/com/bin || fail "Failed to build //java/com/bin"

  filtered_lib=$(zipinfo -1  bazel-bin/java/com/bin/proguard/bin/legacy_bin_combined_library_jars_filtered.jar)
  # Don't assert on the size of the library jar, because it contains the jdk runtime.
  assert_one_of $filtered_lib "com/bin/Lib.class"
  assert_not_one_of $filtered_lib "com/bin/Filtered.class"

  output_classes=$(zipinfo -1 bazel-bin/java/com/bin/bin_proguard.jar)
  assert_equals 3 $(wc -w <<< $output_classes)
  assert_one_of $output_classes "META-INF/MANIFEST.MF"
  assert_one_of $output_classes "com/bin/Bin.class"
  # Not kept by proguard
  assert_not_one_of $output_classes "com/bin/NotUsed.class"
  # This was in ProGuard's library_jars, not in the jar under optimization.
  assert_not_one_of $output_classes "com/bin/Lib.class"
  # This is renamed by proguard to something else
  assert_not_one_of $output_classes "com/bin/Renamed.class"
}

run_suite "Android integration tests for Proguard"
