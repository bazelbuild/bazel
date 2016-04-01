// Copyright 2016 The Bazel Authors. All rights reserved.
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
 * Used as a descriptor of the expected type of the dSYM bundle that is output when building a rule.
 */
enum DsymOutputType {
  /**
   * Specifies that the dSYM bundle should have an 'xctest' suffix, which is the expected type when
   * Xcode runs the tests.
   */
  TEST(".xctest.dSYM"),

  /**
   * Specifies that the dSYM bundle should have an 'app' suffix, which is the default type when
   * generating the bundle for debugging or for crash symbolication.
   */
  APP(".app.dSYM");

  private final String suffix;

  private DsymOutputType(String suffix) {
    this.suffix = suffix;
  }

  /**
   * Returns the suffix to be used by the dSYM bundle output.
   */
  String getSuffix() {
    return suffix;
  }
}
