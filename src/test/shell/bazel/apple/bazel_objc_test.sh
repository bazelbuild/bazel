#!/bin/bash
#
# Copyright 2015 The Bazel Authors. All rights reserved.
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

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

if [ "${PLATFORM}" != "darwin" ]; then
  echo "This test suite requires running on OS X" >&2
  exit 0
fi

function make_lib() {
  rm -rf ios
  mkdir -p ios

  cat >ios/main.m <<EOF
#import <UIKit/UIKit.h>

int main(int argc, char *argv[]) {
  NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
  int retVal = UIApplicationMain(argc, argv, nil, nil);
  [pool release];
  return retVal;
}
EOF

  cat >ios/BUILD <<EOF
objc_library(name = "lib",
             non_arc_srcs = ['main.m'])
EOF
}

function test_build_app() {
  setup_objc_test_support
  make_lib

  bazel build --verbose_failures --apple_platform_type=ios \
      --ios_sdk_version=$IOS_SDK_VERSION \
      //ios:lib >$TEST_log 2>&1 || fail "should pass"
  ls bazel-out/apl-ios_x86_64-fastbuild/bin/ios/liblib.a \
      || fail "should generate lib.a"
}

function test_invalid_ios_sdk_version() {
  setup_objc_test_support
  make_lib

  ! bazel build --verbose_failures --apple_platform_type=ios \
      --ios_sdk_version=2.34 \
      //ios:lib >$TEST_log 2>&1 || fail "should fail"
  expect_log "SDK \"iphonesimulator2.34\" cannot be located."
}

function test_xcodelocator_embedded_tool() {
  rm -rf ios
  mkdir -p ios

  cat >ios/BUILD <<EOF
genrule(
    name = "invoke_tool",
    srcs = ["@bazel_tools//tools/osx:xcode-locator"],
    outs = ["tool_output"],
    cmd = "\$< > \$@",
    tags = ["requires-darwin"],
)
EOF

  bazel build --verbose_failures //ios:invoke_tool >$TEST_log 2>&1 \
      || fail "should be able to resolve xcode-locator"
}

# Verifies contents of .a files do not contain timestamps -- if they did, the
# results would not be hermetic.
function test_archive_timestamps() {
  setup_objc_test_support

  mkdir -p objclib
  cat > objclib/BUILD <<EOF
objc_library(
    name = "objclib",
    srcs = ["mysrc.m"],
)
EOF

  cat > objclib/mysrc.m <<EOF
int aFunction() {
  return 0;
}
EOF

  bazel build --verbose_failures --apple_platform_type=ios \
      --ios_sdk_version=$IOS_SDK_VERSION //objclib:objclib >"$TEST_log" 2>&1 \
      || fail "Should build objc_library"

  # Based on timezones, ar -tv may show the timestamp of the contents as either
  # Dec 31 1969 or Jan 1 1970 -- either is fine.
  # We would use 'date' here, but the format is slightly different (Jan 1 vs.
  # Jan 01).
  ar -tv bazel-out/apl-ios_x86_64-fastbuild/bin/objclib/libobjclib.a \
      | grep "mysrc" | grep "Dec 31" | grep "1969" \
      || ar -tv bazel-out/apl-ios_x86_64-fastbuild/bin/objclib/libobjclib.a \
      | grep "mysrc" | grep "Jan  1" | grep "1970" || \
      fail "Timestamp of contents of archive file should be zero"
}

function test_strip_symbols() {
  setup_objc_test_support

  rm -rf ios
  mkdir -p ios

  cat >ios/main.m <<EOF
#import <UIKit/UIKit.h>
/* function declaration */
int addOne(int num);
int addOne(int num) {
  return num + 1;
}
 int main(int argc, char *argv[]) {
  NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
  int retVal = UIApplicationMain(argc, argv, nil, nil);
  [pool release];
  return retVal;
}
EOF

  cat >ios/BUILD <<EOF
apple_binary(name = 'app',
             deps = [':main'],
             platform_type = 'ios')
objc_library(name = 'main',
             non_arc_srcs = ['main.m'])
EOF

  bazel build --verbose_failures \
      --apple_platform_type=ios \
      --ios_sdk_version=$IOS_SDK_VERSION \
      --objc_enable_binary_stripping=true \
      --compilation_mode=opt \
      //ios:app >$TEST_log 2>&1 || fail "should pass"
  ls bazel-out/apl-ios_x86_64-opt/bin/ios/app_lipobin \
    || fail "should generate lipobin (stripped binary)"
  ! nm bazel-out/apl-ios_x86_64-opt/bin/ios/app_lipobin | grep addOne \
    || fail "should fail to find symbol addOne"
}

run_suite "objc/ios test suite"
