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
#
# Note that if the environment is not set up as above android_integration_test
# will silently be ignored and will be shown as passing.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${CURRENT_DIR}/android_helper.sh" \
  || { echo "android_helper.sh not found!" >&2; exit 1; }
fail_if_no_android_sdk

source "${CURRENT_DIR}/../../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

resolve_android_toolchains

function test_android_sdk_repository_path_from_environment() {
  create_new_workspace
  setup_android_sdk_support
  # Overwrite WORKSPACE that was created by setup_android_sdk_support with one
  # that does not set the path attribute of android_sdk_repository.
  rm WORKSPACE
  cat >> $(create_workspace_with_default_repos WORKSPACE) <<EOF
android_sdk_repository(
    name = "androidsdk",
)
EOF
  ANDROID_HOME=$ANDROID_SDK bazel build @androidsdk//:files || fail \
    "android_sdk_repository failed to build with \$ANDROID_HOME instead of " \
    "path"
}

function test_android_sdk_repository_no_path_or_android_home() {
  create_new_workspace
  setup_android_platforms
  cat >> $(create_workspace_with_default_repos WORKSPACE) <<EOF
android_sdk_repository(
    name = "androidsdk",
    api_level = 25,
)
EOF
  bazel build @androidsdk//:files >& $TEST_log && fail "Should have failed" || true
  expect_log "Either the path attribute of android_sdk_repository"
}

function test_android_sdk_repository_wrong_path() {
  create_new_workspace
  setup_android_platforms
  mkdir "$TEST_SRCDIR/some_dir"
  cat >> $(create_workspace_with_default_repos WORKSPACE) <<EOF
android_sdk_repository(
    name = "androidsdk",
    api_level = 25,
    path = "$TEST_SRCDIR/some_dir",
)
EOF
  bazel build @androidsdk//:files >& $TEST_log && fail "Should have failed" || true
  expect_log "Unable to read the Android SDK at $TEST_SRCDIR/some_dir, the path may be invalid." \
    " Is the path in android_sdk_repository() or \$ANDROID_SDK_HOME set correctly?"
}

# Regression test for https://github.com/bazelbuild/bazel/issues/2621.
function test_android_sdk_repository_returns_null_if_env_vars_missing() {
  create_new_workspace
  setup_android_sdk_support
  ANDROID_HOME=/does_not_exist_1 bazel build @androidsdk//:files || \
    fail "Build failed"
  sed -i -e 's/path =/#path =/g' WORKSPACE
  ANDROID_HOME=/does_not_exist_2 bazel build @androidsdk//:files && \
    fail "Build should have failed"
  ANDROID_HOME=$ANDROID_SDK bazel build @androidsdk//:files || fail "Build failed"
}

# Regression test for https://github.com/bazelbuild/bazel/issues/12069
function test_android_sdk_repository_present_not_set() {
  create_new_workspace
  setup_android_platforms
  cat >> WORKSPACE <<EOF
android_sdk_repository(
    name = "androidsdk",
)
EOF

  mkdir -p a
  cat > a/BUILD <<EOF
package(default_visibility = ["//visibility:public"])
sh_test(
  name = 'helper',
  srcs = [ 'helper.sh' ],
)
EOF

  cat > a/helper.sh <<EOF
#!/bin/sh
exit 0
EOF
  chmod +x a/helper.sh

  unset ANDROID_HOME
  # We should be able to build non-android targets without a valid SDK.
  bazel build //a:helper || fail "Build failed"
  # Trying to actually build android code repo should still fail.
  create_android_binary
  bazel build //java/bazel:bin && fail "Build should have failed" || true
}

run_suite "Android integration tests for SDK"
