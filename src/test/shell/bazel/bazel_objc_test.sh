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

# Load test environment
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test-setup.sh \
  || { echo "test-setup.sh not found!" >&2; exit 1; }

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

  bazel build --verbose_failures \
      //ios:app >$TEST_log 2>&1 || fail "should pass"
  ls bazel-bin/ios/app.xcodeproj || fail "should generate app.xcodeproj"
  ls bazel-bin/ios/app.ipa || fail "should generate app.ipa"
}

function test_ios_test() {
  setup_objc_test_support
  make_app

  bazel build --test_output=all //ios:PassingXcTest >$TEST_log 2>&1 \
      || fail "should pass"
  ls bazel-bin/ios/PassingXcTest.xcodeproj \
      || fail "should generate PassingXcTest.xcodeproj"
  ls bazel-bin/ios/PassingXcTest.ipa \
      || fail "should generate PassingXcTest.ipa"
}

run_suite "objc/ios test suite"
