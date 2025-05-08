// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.python;

import static com.google.devtools.build.lib.testutil.TestConstants.RULES_PYTHON_PACKAGE_ROOT;
import static org.junit.Assume.assumeTrue;

import com.google.devtools.build.lib.testutil.TestConstants;

/** Helpers for Python tests. */
public class PythonTestUtils {

  // Static utilities class.
  private PythonTestUtils() {}

  /**
   * Skips the test if the product isn't bazel. This is mostly to skip tests for py2 support that
   * the Google implementation would otherwise fail on.
   */
  public static void assumeIsBazel() {
    assumeTrue(TestConstants.PRODUCT_NAME.equals("bazel")); // Google has py2 disabled.
  }

  /**
   * Stub method that is used to annotate that the calling test case assumes the default Python
   * version is PY2.
   *
   * <p>Marking test cases that depend on the default Python version helps to diagnose failures. It
   * also helps guard against accidentally making the test spuriously pass, e.g. if the expected
   * value becomes the same as the default value..
   */
  public static void assumesDefaultIsPY2() {
    // No-op.
  }

  /** Same as {@link #assumesDefaultIsPY2}, but for PY3. */
  public static void assumesDefaultIsPY3() {
    // No-op.
  }

  public static String getPyLoad(String symbolName) {
    if (RULES_PYTHON_PACKAGE_ROOT.isEmpty()) {
      return "";
    }
    String bzlFilename;
    switch (symbolName) {
      case "PyInfo":
        bzlFilename = "py_info.bzl";
        break;
      case "PyRuntimeInfo":
        bzlFilename = "py_runtime_info.bzl";
        break;
      default:
        bzlFilename = symbolName + ".bzl";
    }
    return String.format(
        "load('%s/python:%s', '%s')", RULES_PYTHON_PACKAGE_ROOT, bzlFilename, symbolName);
  }
}
