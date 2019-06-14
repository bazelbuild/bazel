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



/** Helpers for Python tests. */
public class PythonTestUtils {

  // Static utilities class.
  private PythonTestUtils() {}

  /**
   * Stub method that is used to annotate that the calling test case assumes the default Python
   * version is PY2.
   *
   * <p>Marking test cases that depend on the default Python version helps to diagnose failures. It
   * also helps guard against accidentally making the test spuriously pass, e.g. if the expected
   * value becomes the same as the default value..
   *
   * <p>Although the hard-coded default in {@link PythonOptions} has been flipped to PY3, we
   * override this back to PY2 in our analysis-time tests and some of our integration tests. These
   * tests will need to be ported in the future.
   */
  public static void assumesDefaultIsPY2() {
    // No-op.
  }

  /** Same as {@link #assumesDefaultIsPY2}, but for PY3. */
  public static void assumesDefaultIsPY3() {
    // No-op.
  }
}
