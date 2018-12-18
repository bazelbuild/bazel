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

import static com.google.common.truth.Truth.assertWithMessage;

/** Helpers for Python tests. */
public class PythonTestUtils {

  // Static utilities class.
  private PythonTestUtils() {}

  /**
   * Assert that {@link PythonVersion#DEFAULT_TARGET_VALUE} hasn't changed.
   *
   * <p>Use this to indicate that the PY2 and PY3 values of your test should be flipped if this
   * default value is changed. In general, it is useful to write tests with expected values that
   * differ from the default, so that they don't spuriously succeed if the default is erroneously
   * returned.
   */
  public static void ensureDefaultIsPY2() {
    assertWithMessage(
            "This test case is written with the assumption that the default is Python 2. When "
                + "updating the default to Python 3, flip all the PY2/PY3 constants in the test "
                + "case and this helper function.")
        .that(PythonVersion.DEFAULT_TARGET_VALUE)
        .isEqualTo(PythonVersion.PY2);
  }
}
