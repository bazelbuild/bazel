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

  # Find where Xcode 7 (any sub-version will do) is located and get the iOS SDK
  # version it contains.
  # TODO(b/27267941): This is a hack until the bug is fixed.
  XCODE_LOCATOR="$(bazel info output_base)/external/bazel_tools/tools/objc/xcode-locator"
  XCODE_INFO=$($XCODE_LOCATOR -v | grep -m1 7)
  XCODE_DIR=$(echo $XCODE_INFO | cut -d ':' -f3)
  XCODE_VERSION=$(echo $XCODE_INFO | cut -d ':' -f1)
  IOS_SDK_VERSION=$(DEVELOPER_DIR=$XCODE_DIR xcodebuild -sdk -version \
      | grep iphonesimulator | cut -d ' '  -f6)

  # Allow access to //external:xcrunwrapper.
  rm WORKSPACE
  ln -sv ${workspace_file} WORKSPACE
}

function make_app() {
  rm -rf ios
  mkdir -p ios

  cat >ios/app.swift <<EOF
import UIKit

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {
  var window: UIWindow?
  func application(
      application: UIApplication,
      didFinishLaunchingWithOptions launchOptions: [NSObject: AnyObject]?)
      -> Bool {
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
            srcs = ["//tools/objc:dummy.c"],
            deps = [":SwiftMain"])

ios_application(name = "app",
                binary = ':bin',
                infoplist = 'App-Info.plist')
EOF
}

function test_swift_library() {
  local swift_lib_pkg=examples/swift
  assert_build_output ./bazel-genfiles/${swift_lib_pkg}/examples_swift_swift_lib.a \
      ${swift_lib_pkg}:swift_lib --ios_sdk_version=$IOS_SDK_VERSION --xcode_version=$XCODE_VERSION
  assert_build_output ./bazel-genfiles/${swift_lib_pkg}/examples_swift_swift_lib.swiftmodule \
      ${swift_lib_pkg}:swift_lib --ios_sdk_version=$IOS_SDK_VERSION --xcode_version=$XCODE_VERSION
}

function test_build_app() {
  make_app

  bazel build --verbose_failures --ios_sdk_version=$IOS_SDK_VERSION \
      --xcode_version=$XCODE_VERSION \
      //ios:app >$TEST_log 2>&1 || fail "should pass"
  ls bazel-bin/ios/app.ipa || fail "should generate app.ipa"
}

function test_objc_depends_on_swift() {
  rm -rf ios
  mkdir -p ios

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
            srcs = ['app.m',],
            deps = [":SwiftMain"])
EOF

  bazel build --verbose_failures --ios_sdk_version=$IOS_SDK_VERSION \
      --xcode_version=$XCODE_VERSION \
      //ios:bin >$TEST_log 2>&1 || fail "should build"
}

function test_swift_imports_objc() {
  rm -rf ios
  mkdir -p ios

  cat >ios/main.swift <<EOF
import Foundation
import ios_ObjcLib

public class SwiftClass {
  public func bar() -> String {
    return ObjcClass().foo()
  }
}
EOF

  cat >ios/ObjcClass.h <<EOF
#import <Foundation/Foundation.h>

#if !DEFINE_FOO
#error "Define is not passed in"
#endif

#if !COPTS_FOO
#error "Copt is not passed in
#endif

@interface ObjcClass : NSObject
- (NSString *)foo;
@end
EOF

  cat >ios/ObjcClass.m <<EOF
#import "ObjcClass.h"
@implementation ObjcClass
- (NSString *)foo { return @"Hello ObjcClass"; }
@end
EOF

  cat >ios/BUILD <<EOF
load("//tools/build_defs/apple:swift.bzl", "swift_library")

swift_library(name = "swift_lib",
              srcs = ["main.swift"],
              deps = [":ObjcLib"])

objc_library(name = "ObjcLib",
             hdrs = ['ObjcClass.h'],
             srcs = ['ObjcClass.m'],
             defines = ["DEFINE_FOO=1"])
EOF

  bazel build --verbose_failures --ios_sdk_version=$IOS_SDK_VERSION \
      --objccopt=-DCOPTS_FOO=1 -s \
      --xcode_version=$XCODE_VERSION \
      //ios:swift_lib >$TEST_log 2>&1 || fail "should build"
  expect_log "-module-cache-path bazel-out/local-fastbuild/genfiles/_objc_module_cache"
}

function test_swift_import_objc_framework() {
  rm -rf ios
  mkdir -p ios

  # Copy the prebuilt framework into app's directory.
  cp -RL "${BAZEL_RUNFILES}/tools/build_defs/apple/test/testdata/BlazeFramework.framework" ios

  cat >ios/main.swift <<EOF
import UIKit

import BlazeFramework

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {
  var window: UIWindow?
  func application(
      application: UIApplication,
      didFinishLaunchingWithOptions launchOptions: [NSObject: AnyObject]?)
        -> Bool {
          NSLog("\(Multiplier().foo())")
          return true
  }
}
EOF

  cat >ios/BUILD <<EOF
load("//tools/build_defs/apple:swift.bzl", "swift_library")

objc_binary(name = "bin",
            srcs = ["//tools/objc:dummy.c"],
            deps = [":swift_lib"])

swift_library(name = "swift_lib",
              srcs = ["main.swift"],
              deps = [":dylib"])

objc_framework(name = "dylib",
               framework_imports = glob(["BlazeFramework.framework/**"]),
               is_dynamic = 1)
EOF

  bazel build --verbose_failures --ios_sdk_version=$IOS_SDK_VERSION \
      --ios_minimum_os=8.0 --xcode_version=$XCODE_VERSION \
      //ios:swift_lib >$TEST_log 2>&1 || fail "should build"
}

function test_fat_apple_binary() {
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
)
apple_binary(
    name = "main_binary",
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

  bazel build --verbose_failures //package:lipo_out  \
    --ios_multi_cpus=i386,x86_64 \
    --xcode_version=$XCODE_VERSION \
    --ios_sdk_version=$IOS_SDK_VERSION \
    || fail "should build apple_binary and obtain info via lipo"

  cat bazel-genfiles/package/lipo_out | grep "i386 x86_64" \
    || fail "expected output binary to contain 2 architectures"
}

function test_apple_binary_lipo_archive() {
  rm -rf package
  mkdir -p package
  cat > package/BUILD <<EOF
apple_binary(
    name = "main_binary",
    srcs = ["a.m"],
)
genrule(
  name = "extract_archives",
  srcs = [":main_binary_lipo.a"],
  outs = ["info_x86_64", "info_i386"],
  cmd =
      "set -e && " +
      "lipo -extract x86_64 \$(location :main_binary_lipo.a) " +
      "-output archive_x86_64.a && " +
      "lipo -extract i386 \$(location :main_binary_lipo.a) " +
      "-output archive_i386.a && " +
      "file archive_x86_64.a > \$(location :info_x86_64) && " +
      "file archive_i386.a > \$(location :info_i386)",
  tags = ["requires-darwin"],
)
EOF
  cat > package/a.m <<EOF
int main() {
  return 0;
}
EOF


  bazel build --verbose_failures //package:extract_archives  \
    --ios_multi_cpus=i386,x86_64 \
    --xcode_version=$XCODE_VERSION \
    --ios_sdk_version=$IOS_SDK_VERSION \
    || fail "should build multi-architecture archive"

  assert_contains "x86_64.*archive" bazel-genfiles/package/info_x86_64
  assert_contains "i386.*archive" bazel-genfiles/package/info_i386
}

function test_swift_imports_swift() {
  rm -rf ios
  mkdir -p ios

  cat >ios/main.swift <<EOF
import Foundation
import ios_util

public class SwiftClass {
  public func bar() -> String {
    return Utility().foo()
  }
}
EOF

  cat >ios/Utility.swift <<EOF
public class Utility {
  public init() {}
  public func foo() -> String { return "foo" }
}
EOF

  cat >ios/BUILD <<EOF
load("//tools/build_defs/apple:swift.bzl", "swift_library")

swift_library(name = "swift_lib",
              srcs = ["main.swift"],
              deps = [":util"])

swift_library(name = "util",
              srcs = ['Utility.swift'])
EOF

  bazel build --verbose_failures --ios_sdk_version=$IOS_SDK_VERSION \
      --xcode_version=$XCODE_VERSION \
      //ios:swift_lib >$TEST_log 2>&1 || fail "should build"
}

function test_swift_tests() {
  make_app

  cat >ios/internal.swift <<EOF
internal class InternalClass {
  func foo() -> String { return "bar" }
}
EOF

  cat >ios/tests.swift <<EOF
  import XCTest
  @testable import ios_SwiftMain

class FooTest: XCTestCase {
  func testFoo() { XCTAssertEqual(2, 3) }
  func testInternalClass() { XCTAssertEqual(InternalClass().foo(), "bar") }
}
EOF

  cat >ios/BUILD <<EOF
load("//tools/build_defs/apple:swift.bzl", "swift_library")

swift_library(name = "SwiftMain",
              srcs = ["app.swift", "internal.swift"])

objc_binary(name = "bin",
            srcs = ["//tools/objc:dummy.c"],
            deps = [":SwiftMain"])

ios_application(name = "app",
                binary = ':bin',
                infoplist = 'App-Info.plist')

swift_library(name = "SwiftTest",
              srcs = ["tests.swift"],
              deps = [":SwiftMain"])

ios_test(name = "app_test",
         srcs = ["//tools/objc:dummy.c"],
         deps = [":SwiftTest"],
         xctest_app = "app")
EOF

  bazel build --verbose_failures --ios_sdk_version=$IOS_SDK_VERSION \
      --xcode_version=$XCODE_VERSION \
      //ios:app_test >$TEST_log 2>&1 || fail "should build"

  otool -lv bazel-bin/ios/app_test_bin \
      | grep @executable_path/Frameworks -sq \
      || fail "expected test binary to contain @executable_path in LC_RPATH"

  otool -lv bazel-bin/ios/app_test_bin \
      | grep @loader_path/Frameworks -sq \
      || fail "expected test binary to contain @loader_path in LC_RPATH"

}

function test_swift_compilation_mode_flags() {
  rm -rf ios
  mkdir -p ios

  cat >ios/debug.swift <<EOF
// A trick to break compilation when DEBUG is not set.
func foo() {
  #if DEBUG
  var x: Int
  #endif
  x = 3
}
EOF

  cat >ios/BUILD <<EOF
load("//tools/build_defs/apple:swift.bzl", "swift_library")

swift_library(name = "swift_lib",
              srcs = ["debug.swift"])
EOF

  ! bazel build --verbose_failures --ios_sdk_version=$IOS_SDK_VERSION -c opt \
      --xcode_version=$XCODE_VERSION \
      //ios:swift_lib >$TEST_log 2>&1 || fail "should not build"
  expect_log "error: use of unresolved identifier 'x'"

  bazel build --verbose_failures --ios_sdk_version=$IOS_SDK_VERSION -c dbg \
      --xcode_version=$XCODE_VERSION \
      //ios:swift_lib >$TEST_log 2>&1 || fail "should build"
}

run_suite "apple_tests"
