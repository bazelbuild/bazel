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

resolve_android_toolchains

function test_library_desugar_lib_builds() {
  # TODO(b/299338002): Move this to the main desugarer test suite.
  create_new_workspace
  setup_android_sdk_support
  create_android_binary

  assert_build @bazel_tools//tools/android:desugar_java8_legacy_libs
}

function test_library_desugaring() {
  # TODO(b/299338002): Move this to the main desugarer test suite.
  create_new_workspace
  setup_android_sdk_support
  create_android_binary

  assert_build //java/bazel:bin --desugar_java8_libs --experimental_check_desugar_deps
}

run_suite "Android library desugaring integration tests"

