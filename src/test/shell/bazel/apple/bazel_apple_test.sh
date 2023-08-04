#!/bin/bash
#
# Copyright 2016 The Bazel Authors. All rights reserved.
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
#
# Tests the examples provided in Bazel
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

if [ "${PLATFORM}" != "darwin" ]; then
  echo "This test suite requires running on OS X" >&2
  exit 0
fi

function set_up() {
  copy_examples
  setup_objc_test_support

  # Find the version number for an installed Xcode.
  XCODE_VERSION=$(xcodebuild -version | grep ^Xcode | cut -d' ' -f2)

  create_new_workspace
}

function test_host_xcodes() {
  XCODE_VERSION=$(env -i xcodebuild -version | grep "Xcode" \
      | sed -E "s/Xcode (([0-9]|.)+).*/\1/")
  XCODE_BUILD_VERSION=$(env -i xcodebuild -version | grep "Build version" \
      | sed -E "s/Build version (([0-9]|.)+).*/\1/")
  IOS_SDK=$(env -i xcodebuild -version -sdk | grep iphoneos \
      | sed -E "s/.*\(iphoneos(([0-9]|.)+)\).*/\1/")
  MACOSX_SDK=$(env -i xcodebuild -version -sdk | grep "(macosx" \
      | sed -E "s/.*\(macosx(([0-9]|.)+)\).*/\1/" | head -n 1)

  # Unfortunately xcodebuild -version doesn't always pad with trailing .0, so,
  # for example, may produce "6.4", which is bad for this test.
  if [[ ! $XCODE_VERSION =~ [0-9].[0-9].[0-9] ]]
  then
    XCODE_VERSION="${XCODE_VERSION}.0"
  fi

  XCODE_VERSION_FULL="${XCODE_VERSION}.${XCODE_BUILD_VERSION}"

  bazel build @local_config_xcode//:host_xcodes >"${TEST_log}" 2>&1 \
     || fail "Expected host_xcodes to build"

  bazel query "attr(version, $XCODE_VERSION_FULL, \
      attr(default_ios_sdk_version, $IOS_SDK, \
      attr(default_macos_sdk_version, $MACOSX_SDK, \
      labels('versions', '@local_config_xcode//:host_xcodes'))))" \
      > xcode_version_target

  assert_contains "local_config_xcode" xcode_version_target

  DEFAULT_LABEL=$(bazel query \
      "labels('default', '@local_config_xcode//:host_xcodes')")

  assert_equals $DEFAULT_LABEL $(cat xcode_version_target)
}

function test_host_available_xcodes() {

  XCODE_VERSION=$(env -i xcodebuild -version | grep "Xcode" \
      | sed -E "s/Xcode (([0-9]|.)+).*/\1/")
  IOS_SDK=$(env -i xcodebuild -version -sdk | grep iphoneos \
      | sed -E "s/.*\(iphoneos(([0-9]|.)+)\).*/\1/")
  MACOSX_SDK=$(env -i xcodebuild -version -sdk | grep "(macosx" \
      | sed -E "s/.*\(macosx(([0-9]|.)+)\).*/\1/" | head -n 1)

  # Unfortunately xcodebuild -version doesn't always pad with trailing .0, so,
  # for example, may produce "6.4", which is bad for this test.
  if [[ ! $XCODE_VERSION =~ [0-9].[0-9].[0-9] ]]
  then
    XCODE_VERSION="${XCODE_VERSION}.0"
  fi

  bazel build @local_config_xcode//:host_available_xcodes >"${TEST_log}" 2>&1 \
     || fail "Expected host_available_xcodes to build"

  bazel query "attr(version, $XCODE_VERSION, \
      attr(default_ios_sdk_version, $IOS_SDK, \
      attr(default_macos_sdk_version, $MACOSX_SDK, \
      labels('versions', '@local_config_xcode//:host_available_xcodes'))))" \
      > xcode_version_target

  assert_contains "local_config_xcode" xcode_version_target

  DEFAULT_LABEL=$(bazel query \
      "labels('default', '@local_config_xcode//:host_available_xcodes')")

  assert_equals "$DEFAULT_LABEL" "$(cat xcode_version_target)"
}

function test_xcode_config_select() {
  mkdir -p a
  cat > a/BUILD <<'EOF'
xcode_config(
    name = "xcodes",
    default = ":version10",
    versions = [ ":version10", ":version20", ":version30" ],
    visibility = ["//visibility:public"],
)

xcode_version(
    name = "version10",
    default_ios_sdk_version = "1.1",
    default_macos_sdk_version = "1.2",
    default_tvos_sdk_version = "1.3",
    default_watchos_sdk_version = "1.4",
    version = "1.0",
)

xcode_version(
    name = "version20",
    default_ios_sdk_version = "2.1",
    default_macos_sdk_version = "2.2",
    default_tvos_sdk_version = "2.3",
    default_watchos_sdk_version = "2.4",
    version = "2.0",
)

xcode_version(
    name = "version30",
    default_ios_sdk_version = "3.1",
    default_macos_sdk_version = "3.2",
    default_tvos_sdk_version = "3.3",
    default_watchos_sdk_version = "3.4",
    version = "3.0",
)

config_setting(
    name = "xcode10",
    flag_values = { "@bazel_tools//tools/osx:xcode_version_flag": "1.0" },
)

config_setting(
    name = "xcode20",
    flag_values = { "@bazel_tools//tools/osx:xcode_version_flag": "2.0" },
)

config_setting(
    name = "ios11",
    flag_values = { "@bazel_tools//tools/osx:ios_sdk_version_flag": "1.1" },
)

config_setting(
    name = "ios21",
    flag_values = { "@bazel_tools//tools/osx:ios_sdk_version_flag": "2.1" },
)

genrule(
    name = "xcode",
    srcs = [],
    outs = ["xcodeo"],
    cmd = "echo " + select({
      ":xcode10": "XCODE 1.0",
      ":xcode20": "XCODE 2.0",
      "//conditions:default": "XCODE UNKNOWN",
    }) + " >$@",)

genrule(
    name = "ios",
    srcs = [],
    outs = ["ioso"],
    cmd = "echo " + select({
      ":ios11": "IOS 1.1",
      ":ios21": "IOS 2.1",
      "//conditions:default": "IOS UNKNOWN",
    }) + " >$@",)

EOF

  bazel build //a:xcode //a:ios --xcode_version_config=//a:xcodes || fail "build failed"
  assert_contains "XCODE 1.0" bazel-genfiles/a/xcodeo
  assert_contains "IOS 1.1" bazel-genfiles/a/ioso

  bazel build //a:xcode //a:ios --xcode_version_config=//a:xcodes \
      --xcode_version=2.0 || fail "build failed"
  assert_contains "XCODE 2.0" bazel-genfiles/a/xcodeo
  assert_contains "IOS 2.1" bazel-genfiles/a/ioso

  bazel build //a:xcode //a:ios --xcode_version_config=//a:xcodes \
      --xcode_version=3.0 || fail "build failed"
  assert_contains "XCODE UNKNOWN" bazel-genfiles/a/xcodeo
  assert_contains "IOS UNKNOWN" bazel-genfiles/a/ioso
}

run_suite "apple_tests"
