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

function test_fat_binary_no_srcs() {
  mkdir -p package
  cat > package/BUILD <<EOF
objc_library(
    name = "lib_a",
    srcs = ["a.m"],
)
objc_library(
    name = "lib_b",
    srcs = ["b.m"],
)
apple_binary(
    name = "main_binary",
    deps = [":lib_a", ":lib_b"],
    platform_type = "ios",
    minimum_os_version = "10.0",
)
genrule(
  name = "lipo_run",
  srcs = [":main_binary_lipobin"],
  outs = ["lipo_out"],
  cmd =
      "set -e && " +
      "lipo -info \$(location :main_binary_lipobin) > \$(@)",
  tags = ["requires-darwin"],
)
EOF
  touch package/a.m
  cat > package/b.m <<EOF
int main() {
  return 0;
}
EOF

  bazel build --verbose_failures --xcode_version=$XCODE_VERSION \
      //package:lipo_out --ios_multi_cpus=i386,x86_64 \
      || fail "should build apple_binary and obtain info via lipo"

  cat bazel-genfiles/package/lipo_out | grep "i386 x86_64" \
    || fail "expected output binary to contain 2 architectures"
}

function test_additive_cpus_flag() {
  mkdir -p package
  cat > package/BUILD <<EOF
objc_library(
    name = "lib_a",
    srcs = ["a.m"],
)
objc_library(
    name = "lib_b",
    srcs = ["b.m"],
)
apple_binary(
    name = "main_binary",
    deps = [":lib_a", ":lib_b"],
    platform_type = "ios",
    minimum_os_version = "10.0",
)
genrule(
  name = "lipo_run",
  srcs = [":main_binary_lipobin"],
  outs = ["lipo_out"],
  cmd =
      "set -e && " +
      "lipo -info \$(location :main_binary_lipobin) > \$(@)",
  tags = ["requires-darwin"],
)
EOF
  touch package/a.m
  cat > package/b.m <<EOF
int main() {
  return 0;
}
EOF

  bazel build --verbose_failures --xcode_version=$XCODE_VERSION \
      //package:lipo_out \
      --ios_multi_cpus=i386 --ios_multi_cpus=x86_64 \
      || fail "should build apple_binary and obtain info via lipo"

  cat bazel-genfiles/package/lipo_out | grep "i386 x86_64" \
    || fail "expected output binary to contain 2 architectures"
}

function test_host_xcodes() {
  XCODE_VERSION=$(env -i xcodebuild -version | grep "Xcode" \
      | sed -E "s/Xcode (([0-9]|.)+).*/\1/")
  IOS_SDK=$(env -i xcodebuild -version -sdk | grep iphoneos \
      | sed -E "s/.*\(iphoneos(([0-9]|.)+)\).*/\1/")
  MACOSX_SDK=$(env -i xcodebuild -version -sdk | grep macosx \
      | sed -E "s/.*\(macosx(([0-9]|.)+)\).*/\1/" | head -n 1)

  # Unfortunately xcodebuild -version doesn't always pad with trailing .0, so,
  # for example, may produce "6.4", which is bad for this test.
  if [[ ! $XCODE_VERSION =~ [0-9].[0-9].[0-9] ]]
  then
    XCODE_VERSION="${XCODE_VERSION}.0"
  fi

  bazel build @local_config_xcode//:host_xcodes >"${TEST_log}" 2>&1 \
     || fail "Expected host_xcodes to build"

  bazel query "attr(version, $XCODE_VERSION, \
      attr(default_ios_sdk_version, $IOS_SDK, \
      attr(default_macos_sdk_version, $MACOSX_SDK, \
      labels('versions', '@local_config_xcode//:host_xcodes'))))" \
      > xcode_version_target

  assert_contains "local_config_xcode" xcode_version_target

  DEFAULT_LABEL=$(bazel query \
      "labels('default', '@local_config_xcode//:host_xcodes')")

  assert_equals $DEFAULT_LABEL $(cat xcode_version_target)
}

function test_apple_binary_crosstool_ios() {
  rm -rf package
  mkdir -p package
  cat > package/BUILD <<EOF
objc_library(
    name = "lib_a",
    srcs = ["a.m"],
)
objc_library(
    name = "lib_b",
    srcs = ["b.m"],
    deps = [":cc_lib"],
)
cc_library(
    name = "cc_lib",
    srcs = ["cc_lib.cc"],
)
apple_binary(
    name = "main_binary",
    deps = [":main_lib"],
    platform_type = "ios",
    minimum_os_version = "10.0",
)
objc_library(
    name = "main_lib",
    deps = [":lib_a", ":lib_b"],
    srcs = ["main.m"],
)
genrule(
  name = "lipo_run",
  srcs = [":main_binary_lipobin"],
  outs = ["lipo_out"],
  cmd =
      "set -e && " +
      "lipo -info \$(location :main_binary_lipobin) > \$(@)",
  tags = ["requires-darwin"],
)
EOF
  touch package/a.m
  touch package/b.m
  cat > package/main.m <<EOF
int main() {
  return 0;
}
EOF
  cat > package/cc_lib.cc << EOF
#include <string>

std::string GetString() { return "h3ll0"; }
EOF

  bazel build --verbose_failures //package:lipo_out \
    --ios_multi_cpus=i386,x86_64 \
    --xcode_version=$XCODE_VERSION \
    || fail "should build apple_binary and obtain info via lipo"

  cat bazel-genfiles/package/lipo_out | grep "i386 x86_64" \
    || fail "expected output binary to be for x86_64 architecture"
}

function test_apple_binary_crosstool_watchos() {
  rm -rf package
  mkdir -p package
  cat > package/BUILD <<EOF
genrule(
  name = "lipo_run",
  srcs = [":main_binary_lipobin"],
  outs = ["lipo_out"],
  cmd =
      "set -e && " +
      "lipo -info \$(location :main_binary_lipobin) > \$(@)",
  tags = ["requires-darwin"],
)

apple_binary(
    name = "main_binary",
    deps = [":main_lib"],
    platform_type = "watchos",
)
objc_library(
    name = "main_lib",
    srcs = ["main.m"],
    deps = [":lib_a"],
)
cc_library(
    name = "cc_lib",
    srcs = ["cc_lib.cc"],
)
# By depending on a library which requires it is built for watchos,
# this test verifies that dependencies of apple_binary are compiled
# for the specified platform_type.
objc_library(
    name = "lib_a",
    srcs = ["a.m"],
    deps = [":cc_lib"],
)
EOF
  cat > package/main.m <<EOF
#import <WatchKit/WatchKit.h>

// Note that WKExtensionDelegate is only available in Watch SDK.
@interface TestInterfaceMain : NSObject <WKExtensionDelegate>
@end

int main() {
  return 0;
}
EOF
  cat > package/a.m <<EOF
#import <WatchKit/WatchKit.h>

// Note that WKExtensionDelegate is only available in Watch SDK.
@interface TestInterfaceA : NSObject <WKExtensionDelegate>
@end

int aFunction() {
  return 0;
}
EOF
  cat > package/cc_lib.cc << EOF
#include <string>

std::string GetString() { return "h3ll0"; }
EOF

  bazel build --verbose_failures //package:lipo_out \
      --watchos_cpus=armv7k \
      --xcode_version=$XCODE_VERSION \
      || fail "should build watch binary"

  cat bazel-genfiles/package/lipo_out | grep "armv7k" \
      || fail "expected output binary to be for armv7k architecture"

  bazel build --verbose_failures //package:lipo_out \
      --watchos_cpus=i386 \
      --xcode_version=$XCODE_VERSION \
      || fail "should build watch binary"

  cat bazel-genfiles/package/lipo_out | grep "i386" \
      || fail "expected output binary to be for i386 architecture"
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

function test_apple_binary_dsym_builds() {
  rm -rf package
  mkdir -p package
  cat > package/BUILD <<EOF
apple_binary(
    name = "main_binary",
    deps = [":main_lib"],
    platform_type = "ios",
    minimum_os_version = "10.0",
)
objc_library(
    name = "main_lib",
    srcs = ["main.m"],
)
EOF
  cat > package/main.m <<EOF
int main() {
  return 0;
}
EOF

  bazel build --verbose_failures //package:main_binary \
      --ios_multi_cpus=i386,x86_64 \
      --xcode_version=$XCODE_VERSION \
      --apple_generate_dsym=true \
      || fail "should build apple_binary with dSYMs"
}

function test_apple_binary_spaces() {
  rm -rf package
  mkdir -p package
  cat > package/BUILD <<EOF
apple_binary(
    name = "main_binary",
    deps = [":main_lib"],
    platform_type = "ios",
    minimum_os_version = "10.0",
)
objc_library(
    name = "main_lib",
    srcs = ["the main.m"],
)
EOF
  cat > "package/the main.m" <<EOF
int main() {
  return 0;
}
EOF

  bazel build --verbose_failures //package:main_binary \
      --ios_multi_cpus=i386,x86_64 \
      --xcode_version=$XCODE_VERSION \
      --apple_generate_dsym=true \
      || fail "should build apple_binary with dSYMs"
}

function test_apple_static_library() {
  rm -rf package
  mkdir -p package
  cat > package/BUILD <<EOF
apple_static_library(
    name = "static_lib",
    deps = [":dummy_lib"],
    platform_type = "ios",
)
objc_library(
    name = "dummy_lib",
    srcs = ["dummy.m"],
)
EOF
  cat > "package/dummy.m" <<EOF
static int dummy __attribute__((unused,used)) = 0;
EOF

  bazel build --verbose_failures //package:static_lib \
      --ios_multi_cpus=i386,x86_64 \
      --ios_minimum_os=8.0 \
      --xcode_version=$XCODE_VERSION \
      || fail "should build apple_static_library"
}

run_suite "apple_tests"
