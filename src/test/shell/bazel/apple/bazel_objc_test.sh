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

function make_app() {
  rm -rf ios
  mkdir -p ios

  cat >ios/app.m <<EOF
#import <UIKit/UIKit.h>

int main(int argc, char *argv[]) {
  NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
  int retVal = UIApplicationMain(argc, argv, nil, nil);
  [pool release];
  return retVal;
}
EOF

  cat >ios/App-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>app</string>
        <key>CFBundleName</key>
        <string>app</string>
        <key>CFBundleDisplayName</key>
        <string>app</string>
        <key>CFBundlePackageType</key>
        <string>APPL</string>
        <key>CFBundleIdentifier</key>
        <string>com.google.app</string>
        <key>CFBundleSignature</key>
        <string>????</string>
        <key>CFBundleVersion</key>
        <string>1.0</string>
        <key>LSRequiresIPhoneOS</key>
        <true/>
</dict>
</plist>
EOF

  cat >ios/PassTest-Info.plist <<EOF
<plist version="1.0">
<dict>
        <key>CFBundleExecutable</key>
        <string>PassingXcTest</string>
</dict>
</plist>
EOF

  cat >ios/passtest.m <<EOF
#import <XCTest/XCTest.h>

@interface PassingXcTest : XCTestCase

@end

@implementation PassingXcTest

- (void)testPass {
  XCTAssertEqual(1, 1, @"should pass");
}

@end
EOF

  cat >ios/BUILD <<EOF
objc_binary(name = "bin",
            non_arc_srcs = ['app.m'])
ios_application(name = "app",
                binary = ':bin',
                infoplist = 'App-Info.plist')
ios_test(name = 'PassingXcTest',
         srcs = ['passtest.m'],
         infoplist = "PassTest-Info.plist",
         xctest = True,
         xctest_app = ':app')
EOF
}

function test_build_app() {
  setup_objc_test_support
  make_app

  bazel build --verbose_failures --ios_sdk_version=$IOS_SDK_VERSION \
      //ios:app >$TEST_log 2>&1 || fail "should pass"
  ls bazel-bin/ios/app.xcodeproj || fail "should generate app.xcodeproj"
  ls bazel-bin/ios/app.ipa || fail "should generate app.ipa"
}

function test_ios_test() {
  setup_objc_test_support
  make_app

  bazel build --test_output=all --ios_sdk_version=$IOS_SDK_VERSION \
      //ios:PassingXcTest >$TEST_log 2>&1 || fail "should pass"
  ls bazel-bin/ios/PassingXcTest.xcodeproj \
      || fail "should generate PassingXcTest.xcodeproj"
  ls bazel-bin/ios/PassingXcTest.ipa \
      || fail "should generate PassingXcTest.ipa"
}

function test_valid_ios_sdk_version() {
  setup_objc_test_support
  make_app

  bazel build --verbose_failures --ios_sdk_version=$IOS_SDK_VERSION \
      //ios:app >$TEST_log 2>&1 || fail "should pass"
  ls bazel-bin/ios/app.xcodeproj || fail "should generate app.xcodeproj"
  ls bazel-bin/ios/app.ipa || fail "should generate app.ipa"
}

# Bazel caches mappings for ios sdk locations for local host execution.
# Verify that multiple invocations (with primed cache) work.
function test_xcrun_cache() {
  setup_objc_test_support
  make_app

  ! ls bazel-out/__xcruncache || fail "clean build should not have cache file"
  bazel build --verbose_failures --ios_sdk_version=$IOS_SDK_VERSION \
      //ios:bin >$TEST_log 2>&1 || fail "should pass"
  ls bazel-out/__xcruncache || fail "xcrun cache should be present"
  bazel build --verbose_failures --ios_sdk_version=$IOS_SDK_VERSION \
      //ios:app >$TEST_log 2>&1 || fail "should pass"
  ls bazel-bin/ios/app.xcodeproj || fail "should generate app.xcodeproj"
  ls bazel-bin/ios/app.ipa || fail "should generate app.ipa"
  ls bazel-out/__xcruncache || fail "xcrun cache should be present"

  bazel clean
  ! ls bazel-bin/__xcruncache || fail "xcrun cache should be removed on clean"
}

function test_invalid_ios_sdk_version() {
  setup_objc_test_support
  make_app

  ! bazel build --verbose_failures --ios_sdk_version=2.34 \
      //ios:app >$TEST_log 2>&1 || fail "should fail"
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

  bazel build --verbose_failures --ios_sdk_version=$IOS_SDK_VERSION \
      //objclib:objclib >"$TEST_log" 2>&1 || \
      fail "Should build objc_library"

  # Based on timezones, ar -tv may show the timestamp of the contents as either
  # Dec 31 1969 or Jan 1 1970 -- either is fine.
  # We would use 'date' here, but the format is slightly different (Jan 1 vs.
  # Jan 01).
  ar -tv bazel-bin/objclib/libobjclib.a \
      | grep "mysrc" | grep "Dec 31" | grep "1969" \
      || ar -tv bazel-bin/objclib/libobjclib.a \
      | grep "mysrc" | grep "Jan  1" | grep "1970" || \
      fail "Timestamp of contents of archive file should be zero"
}

run_suite "objc/ios test suite"
