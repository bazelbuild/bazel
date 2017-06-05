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

  # Find the version number for an installed 7-series or 8-series Xcode
  # (any sub-version will do)
  bazel query "labels('versions', '@local_config_xcode//:host_xcodes')" \
      --output xml  | grep 'name="version"' \
      | sed -E 's/.*(value=\"(([0-9]|.)+))\".*/\2/' > xcode_versions

  XCODE_VERSION=$(cat xcode_versions | grep -m1 '7\|8')

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
  assert_build_output ./bazel-genfiles/${swift_lib_pkg}/swift_lib/_objs/examples_swift_swift_lib.a \
      ${swift_lib_pkg}:swift_lib --xcode_version=$XCODE_VERSION
  assert_build_output ./bazel-genfiles/${swift_lib_pkg}/swift_lib/_objs/examples_swift_swift_lib.swiftmodule \
      ${swift_lib_pkg}:swift_lib --xcode_version=$XCODE_VERSION
}

function test_build_app() {
  make_app

  bazel build --verbose_failures --xcode_version=$XCODE_VERSION \
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

  bazel build --verbose_failures --xcode_version=$XCODE_VERSION \
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

  bazel build --verbose_failures --objccopt=-DCOPTS_FOO=1 -s \
      --xcode_version=$XCODE_VERSION \
      //ios:swift_lib >$TEST_log 2>&1 || fail "should build"
  expect_log "-module-cache-path bazel-out/darwin_x86_64-fastbuild/genfiles/_objc_module_cache"
}

function test_swift_import_objc_framework() {
  rm -rf ios
  mkdir -p ios

  # Copy the prebuilt framework into app's directory.
  cp -RL "${BAZEL_RUNFILES}/tools/build_defs/apple/test/testdata/BlazeFramework.framework" ios

  cat >ios/app.swift <<EOF
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
              srcs = ["app.swift"],
              deps = [":dylib"])

objc_framework(name = "dylib",
               framework_imports = glob(["BlazeFramework.framework/**"]),
               is_dynamic = 1)
EOF

  bazel build --verbose_failures --ios_minimum_os=8.0 \
      --xcode_version=$XCODE_VERSION \
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
    || fail "should build apple_binary and obtain info via lipo"

  cat bazel-genfiles/package/lipo_out | grep "i386 x86_64" \
    || fail "expected output binary to contain 2 architectures"
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

  bazel build --verbose_failures --xcode_version=$XCODE_VERSION \
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

  bazel build --verbose_failures --xcode_version=$XCODE_VERSION \
      --ios_minimum_os=8.0 \
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

  ! bazel build --verbose_failures -c opt \
      --xcode_version=$XCODE_VERSION \
      //ios:swift_lib >$TEST_log 2>&1 || fail "should not build"
  expect_log "error: use of unresolved identifier 'x'"

  bazel build --verbose_failures -c dbg \
      --xcode_version=$XCODE_VERSION \
      //ios:swift_lib >$TEST_log 2>&1 || fail "should build"
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

function test_swift_defines() {
  rm -rf ios
  mkdir -p ios
  touch ios/dummy.swift

  cat >ios/main.swift <<EOF
import Foundation

public class SwiftClass {
  public func bar() {
    #if !FLAG
    let x: String = 1 // Invalid statement, should throw compiler error when FLAG is not set
    #endif

    #if !DEP_FLAG
    let x: String = 2 // Invalid statement, should throw compiler error when DEP_FLAG is not set
    #endif
  }
}
EOF

  cat >ios/BUILD <<EOF
load("//tools/build_defs/apple:swift.bzl", "swift_library")

swift_library(name = "dep_lib",
              srcs = ["dummy.swift"],
              defines = ["DEP_FLAG"])

swift_library(name = "swift_lib",
              srcs = ["main.swift"],
              defines = ["FLAG"],
              deps = [":dep_lib"])
EOF

  bazel build --verbose_failures --xcode_version=$XCODE_VERSION \
      //ios:swift_lib >$TEST_log 2>&1 || fail "should build"
}

function test_apple_watch_with_swift() {
  make_app

  cat >ios/watchapp.swift <<EOF
  import WatchKit
  class ExtensionDelegate: NSObject, WKExtensionDelegate {
    func applicationDidFinishLaunching() {}
  }
EOF

  cat >ios/BUILD <<EOF
load("//tools/build_defs/apple:swift.bzl", "swift_library")

swift_library(name = "WatchModule",
              srcs = ["watchapp.swift"])

apple_binary(name = "bin",
             deps = [":WatchModule"],
             platform_type = "watchos")

apple_watch2_extension(
    name = "WatchExtension",
    app_bundle_id = "com.google.app.watchkit",
    app_name = "WatchApp",
    binary = ":bin",
    ext_bundle_id = "com.google.app.extension",
)
EOF

  bazel build --verbose_failures --xcode_version=$XCODE_VERSION \
      //ios:WatchExtension >$TEST_log 2>&1 || fail "should build"
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
      attr(default_macosx_sdk_version, $MACOSX_SDK, \
      labels('versions', '@local_config_xcode//:host_xcodes'))))" \
      > xcode_version_target

  assert_contains "local_config_xcode" xcode_version_target

  DEFAULT_LABEL=$(bazel query \
      "labels('default', '@local_config_xcode//:host_xcodes')")

  assert_equals $DEFAULT_LABEL $(cat xcode_version_target)
}

function test_no_object_file_collisions() {
  rm -rf ios
  mkdir -p ios

  touch ios/foo.swift

  cat >ios/BUILD <<EOF
load("//tools/build_defs/apple:swift.bzl", "swift_library")

swift_library(name = "Foo",
              srcs = ["foo.swift"])
swift_library(name = "Bar",
              srcs = ["foo.swift"])
EOF

  bazel build --verbose_failures --xcode_version=$XCODE_VERSION \
      //ios:{Foo,Bar} >$TEST_log 2>&1 || fail "should build"
}

function test_minimum_os() {
  rm -rf ios
  mkdir -p ios

  touch ios/foo.swift

  cat >ios/BUILD <<EOF
load("//tools/build_defs/apple:swift.bzl", "swift_library")

swift_library(name = "foo",
              srcs = ["foo.swift"])
EOF

  bazel build --verbose_failures -s --announce_rc \
      --xcode_version=$XCODE_VERSION --ios_minimum_os=9.0 \
      //ios:foo >$TEST_log 2>&1 || fail "should build"

  # Get the min OS version encoded as "version" argument of
  # LC_VERSION_MIN_IPHONEOS load command in Mach-O
  MIN_OS=$(otool -l bazel-genfiles/ios/foo/_objs/ios_foo.a | \
      grep -A 3 LC_VERSION_MIN_IPHONEOS | grep version | cut -d " " -f4)
  assert_equals $MIN_OS "9.0"
}

function test_swift_copts() {
  rm -rf ios
  mkdir -p ios

  cat >ios/main.swift <<EOF
import Foundation

public class SwiftClass {
  public func bar() {
    #if !FLAG
    let x: String = 1 // Invalid statement, should throw compiler error when FLAG is not set
    #endif

    #if !CMD_FLAG
    let y: String = 1 // Invalid statement, should throw compiler error when CMD_FLAG is not set
    #endif
  }
}
EOF

cat >ios/BUILD <<EOF
load("//tools/build_defs/apple:swift.bzl", "swift_library")

swift_library(name = "swift_lib",
              srcs = ["main.swift"],
              copts = ["-DFLAG"])
EOF

  bazel build --verbose_failures --xcode_version=$XCODE_VERSION \
      --swiftcopt=-DCMD_FLAG \
      //ios:swift_lib >$TEST_log 2>&1 || fail "should build"
}

function test_swift_bitcode() {
  rm -rf ios
  mkdir -p ios

cat >ios/main.swift <<EOF
func f() {}
EOF

cat >ios/BUILD <<EOF
load("//tools/build_defs/apple:swift.bzl", "swift_library")

swift_library(name = "swift_lib",
              srcs = ["main.swift"])
EOF

  ARCHIVE=bazel-genfiles/ios/swift_lib/_objs/ios_swift_lib.a

  # No bitcode
  bazel build --verbose_failures --xcode_version=$XCODE_VERSION --ios_multi_cpus=arm64 \
      //ios:swift_lib >$TEST_log 2>&1 || fail "should build"
  ! otool -l $ARCHIVE | grep __bitcode -sq \
      || fail "expected a.o to not contain bitcode"

  # Bitcode marker
  bazel build --verbose_failures \
      --xcode_version=$XCODE_VERSION --apple_bitcode=embedded_markers --ios_multi_cpus=arm64 \
      //ios:swift_lib >$TEST_log 2>&1 || fail "should build"
  # Bitcode marker has a length of 1.
  assert_equals $(size -m $ARCHIVE | grep __bitcode | cut -d: -f2 | tr -d ' ') "1"

  # Full bitcode
  bazel build --verbose_failures \
      --xcode_version=$XCODE_VERSION --apple_bitcode=embedded --ios_multi_cpus=arm64 \
      //ios:swift_lib >$TEST_log 2>&1 || fail "should build"
  otool -l $ARCHIVE | grep __bitcode -sq \
      || fail "expected a.o to contain bitcode"

  # Bitcode disabled because of simulator architecture
  bazel build --verbose_failures \
      --xcode_version=$XCODE_VERSION --apple_bitcode=embedded --ios_multi_cpus=x86_64 \
      //ios:swift_lib >$TEST_log 2>&1 || fail "should build"
  ! otool -l $ARCHIVE | grep __bitcode -sq \
      || fail "expected a.o to not contain bitcode"
}

function test_swift_name_validation() {
  rm -rf ios
  mkdir -p ios

  touch ios/main.swift
  touch ios/main.m

cat >ios/BUILD <<EOF
load("//tools/build_defs/apple:swift.bzl", "swift_library")

swift_library(name = "swift-lib",
              srcs = ["main.swift"])
EOF

  ! bazel build --verbose_failures \
      --xcode_version=$XCODE_VERSION \
      //ios:swift-lib >$TEST_log 2>&1 || fail "should fail"
  expect_log "Error in target '//ios:swift-lib'"

  cat >ios/BUILD <<EOF
load("//tools/build_defs/apple:swift.bzl", "swift_library")

objc_library(name = "bad-dep", srcs = ["main.m"])

swift_library(name = "swift_lib",
              srcs = ["main.swift"], deps=[":bad-dep"])
EOF

  ! bazel build --verbose_failures \
      --xcode_version=$XCODE_VERSION \
      //ios:swift_lib >$TEST_log 2>&1 || fail "should fail"
  expect_log "Error in target '//ios:bad-dep'"
}

function test_swift_ast_is_recorded() {
  rm -rf ios
  mkdir -p ios

  touch ios/main.swift
  cat >ios/dep.swift <<EOF
import UIKit
// Add dummy code so that Swift symbols are exported into final binary, which
// will cause runtime libraries to be packaged into the IPA
class X: UIViewController {}
EOF

  cat >ios/main.m <<EOF
#import <UIKit/UIKit.h>

int main(int argc, char *argv[]) {
  @autoreleasepool {
    return UIApplicationMain(argc, argv, nil, nil);
  }
}
EOF

  cat >ios/BUILD <<EOF
load("//tools/build_defs/apple:swift.bzl", "swift_library")

swift_library(name = "dep",
              srcs = ["dep.swift"])

swift_library(name = "swift_lib",
              srcs = ["main.swift"],
              deps = [":dep"])
objc_binary(name = "bin",
            srcs = ["main.m"],
            deps = [":swift_lib"])
EOF

  bazel build --verbose_failures --xcode_version=$XCODE_VERSION -s \
      //ios:bin >$TEST_log 2>&1 || fail "should build"
  expect_log "-Xlinker -add_ast_path -Xlinker bazel-out/darwin_x86_64-fastbuild/genfiles/ios/dep/_objs/ios_dep.swiftmodule"
  expect_log "-Xlinker -add_ast_path -Xlinker bazel-out/darwin_x86_64-fastbuild/genfiles/ios/swift_lib/_objs/ios_swift_lib.swiftmodule"
}

function test_swiftc_script_mode() {
  rm -rf ios
  mkdir -p ios
  touch ios/foo.swift

  cat >ios/top.swift <<EOF
print() // Top level expression outside of main.swift, should fail.
EOF

  cat >ios/main.swift <<EOF
import UIKit

class AppDelegate: UIResponder, UIApplicationDelegate {}

#if swift(>=3)
UIApplicationMain(
  CommandLine.argc,
  UnsafeMutableRawPointer(CommandLine.unsafeArgv)
    .bindMemory(
      to: UnsafeMutablePointer<Int8>.self,
      capacity: Int(CommandLine.argc)),
  nil,
  NSStringFromClass(AppDelegate.self)
)
#else
UIApplicationMain(
  Process.argc, UnsafeMutablePointer<UnsafeMutablePointer<CChar>>(Process.unsafeArgv),
  nil, NSStringFromClass(AppDelegate)
)
#endif
EOF

cat >ios/BUILD <<EOF
load("//tools/build_defs/apple:swift.bzl", "swift_library")

swift_library(name = "main_should_compile_as_script",
              srcs = ["main.swift", "foo.swift"])
swift_library(name = "top_should_not_compile_as_script",
              srcs = ["top.swift"])
swift_library(name = "single_source_should_compile_as_library",
              srcs = ["foo.swift"])
EOF

  bazel build --verbose_failures --xcode_version=$XCODE_VERSION \
      //ios:single_source_should_compile_as_library \
      //ios:main_should_compile_as_script >$TEST_log 2>&1 || fail "should build"

  ! bazel build --verbose_failures --xcode_version=$XCODE_VERSION \
      //ios:top_should_not_compile_as_script >$TEST_log 2>&1 || fail "should not build"
  expect_log "ios/top.swift:1:1: error: expressions are not allowed at the top level"
}

# Test that it's possible to import Clang module of a target that contains private headers.
function test_import_module_with_private_hdrs() {
  rm -rf ios
  mkdir -p ios
  touch ios/Foo.h ios/Foo_Private.h

cat >ios/main.swift <<EOF
import ios_lib
EOF

cat >ios/BUILD <<EOF
load("//tools/build_defs/apple:swift.bzl", "swift_library")

objc_library(name = "lib",
             srcs = ["Foo_Private.h"],
             hdrs = ["Foo.h"])

swift_library(name = "swiftmodule",
              srcs = ["main.swift"],
              deps = [":lib"])
EOF
  bazel build --verbose_failures --xcode_version=$XCODE_VERSION \
      //ios:swiftmodule >$TEST_log 2>&1 || fail "should build"
}

function test_swift_whole_module_optimization() {
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
              deps = [":util"],
              copts = ["-wmo"])

swift_library(name = "util",
              srcs = ['Utility.swift'],
              copts = ["-whole-module-optimization"])
EOF

  bazel build --verbose_failures --xcode_version=$XCODE_VERSION \
      //ios:swift_lib >$TEST_log 2>&1 || fail "should build"
}

function test_swift_dsym() {
  rm -rf ios
  mkdir -p ios

  cat >ios/main.swift <<EOF
import Foundation

public class SwiftClass {
  public func bar() -> String { return "foo" } }
EOF

cat >ios/BUILD <<EOF
load("//tools/build_defs/apple:swift.bzl", "swift_library")

swift_library(name = "swift_lib",
              srcs = ["main.swift"])
EOF

  bazel build -c opt --apple_generate_dsym \
      --verbose_failures --xcode_version=$XCODE_VERSION \
      //ios:swift_lib >$TEST_log 2>&1 || fail "should build"

  # Verify that debug info is present.
  dwarfdump -R bazel-genfiles/ios/swift_lib/_objs/ios_swift_lib.a \
      | grep -sq "__DWARF" \
      || fail "should contain DWARF data"
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
    --experimental_objc_crosstool=all \
    --apple_crosstool_transition \
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
    srcs = ["main.m"],
    deps = [":lib_a"],
    platform_type = "watchos",
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
      --experimental_objc_crosstool=library \
      --apple_crosstool_transition \
      --watchos_cpus=armv7k \
      --xcode_version=$XCODE_VERSION \
      || fail "should build watch binary"

  cat bazel-genfiles/package/lipo_out | grep "armv7k" \
    || fail "expected output binary to be for armv7k architecture"
}

run_suite "apple_tests"
