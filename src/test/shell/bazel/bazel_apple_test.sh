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

# Load test environment
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test-setup.sh \
  || { echo "test-setup.sh not found!" >&2; exit 1; }

if [ "${PLATFORM}" != "darwin" ]; then
  echo "This test suite requires running on OS X" >&2
  exit 0
fi

function set_up() {
  copy_examples
  setup_objc_test_support

  # Allow access to //external:xcrunwrapper.
  rm WORKSPACE
  ln -sv ${workspace_file} WORKSPACE
}

function make_app() {
  rm -rf ios
  mkdir -p ios

  touch ios/dummy.swift

  cat >ios/app.swift <<EOF
import UIKit

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {
  var window: UIWindow?
  func application(application: UIApplication, didFinishLaunchingWithOptions launchOptions: [NSObject: AnyObject]?) -> Bool {
    NSLog("Hello, world")
    return true
  }
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

  cat >ios/BUILD <<EOF
load("//tools/build_defs/apple:swift.bzl", "swift_library")

swift_library(name = "SwiftMain",
              srcs = ["app.swift"])

objc_binary(name = "bin",
            # TODO(b/28723643): This dummy is only here to trigger the
            # USES_SWIFT flag on ObjcProvider and should not be necessary.
            srcs = ['dummy.swift'],
            deps = [":SwiftMain"])

ios_application(name = "app",
                binary = ':bin',
                infoplist = 'App-Info.plist')
EOF
}

function test_swift_library() {
  local swift_lib_pkg=examples/swift
  assert_build_output ./bazel-genfiles/${swift_lib_pkg}/swift_lib.a \
      ${swift_lib_pkg}:swift_lib --ios_sdk_version=$IOS_SDK_VERSION
  assert_build_output ./bazel-genfiles/${swift_lib_pkg}/swift_lib.swiftmodule \
      ${swift_lib_pkg}:swift_lib --ios_sdk_version=$IOS_SDK_VERSION
}

function test_build_app() {
  make_app

  bazel build --verbose_failures --ios_sdk_version=$IOS_SDK_VERSION \
      //ios:app >$TEST_log 2>&1 || fail "should pass"
  ls bazel-bin/ios/app.ipa || fail "should generate app.ipa"
}

function test_objc_depends_on_swift() {
  rm -rf ios
  mkdir -p ios

  touch ios/dummy.swift

  cat >ios/main.swift <<EOF
import Foundation

@objc public class Foo: NSObject {
  public func bar() -> Int { return 42; }
}
EOF

  cat >ios/app.m <<EOF
#import <UIKit/UIKit.h>
#import "ios/SwiftMain-Swift.h"

int main(int argc, char *argv[]) {
  @autoreleasepool {
    NSLog(@"%d", [[[Foo alloc] init] bar]);
    return UIApplicationMain(argc, argv, nil, nil);
  }
}
EOF

  cat >ios/BUILD <<EOF
load("//tools/build_defs/apple:swift.bzl", "swift_library")

swift_library(name = "SwiftMain",
              srcs = ["main.swift"])

objc_binary(name = "bin",
            # TODO(b/28723643): This dummy is only here to trigger the
            # USES_SWIFT flag on ObjcProvider and should not be necessary.
            srcs = ['app.m', 'dummy.swift'],
            deps = [":SwiftMain"])
EOF

  bazel build --verbose_failures --ios_sdk_version=$IOS_SDK_VERSION \
      //ios:bin >$TEST_log 2>&1 || fail "should build"
}

run_suite "apple_tests"
