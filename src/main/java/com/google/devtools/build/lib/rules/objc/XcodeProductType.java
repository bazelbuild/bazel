// Copyright 2014 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.rules.objc;

/**
 * Possible values that {@code objc_*} rules care about for what Xcode project files refer to as
 * "product type."
 */
enum XcodeProductType {
  LIBRARY_STATIC("com.apple.product-type.library.static"),
  BUNDLE("com.apple.product-type.bundle"),
  APPLICATION("com.apple.product-type.application"),
  UNIT_TEST("com.apple.product-type.bundle.unit-test"),
  EXTENSION("com.apple.product-type.app-extension"),
  FRAMEWORK("com.apple.product-type.framework"),
  WATCH_OS1_APPLICATION("com.apple.product-type.application.watchapp"),
  WATCH_OS2_APPLICATION("com.apple.product-type.application.watchapp2"),
  WATCH_OS1_EXTENSION("com.apple.product-type.watchkit-extension"),
  WATCH_OS2_EXTENSION("com.apple.product-type.watchkit2-extension");

  private final String identifier;

  XcodeProductType(String identifier) {
    this.identifier = identifier;
  }

  /**
   * Returns the string used to identify this product type in the {@code productType} field of
   * {@code PBXNativeTarget} objects in Xcode project files.
   */
  public String getIdentifier() {
    return identifier;
  }
}
