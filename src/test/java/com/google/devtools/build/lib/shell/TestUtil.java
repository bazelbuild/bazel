// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.shell;

import static com.google.common.truth.Truth.assertThat;

/**
 * Some tiny conveniences for writing tests.
 */
class TestUtil {

  private TestUtil() {}

  public static void assertArrayEquals(byte[] expected, byte[] actual) {
    assertThat(actual).isEqualTo(expected);
  }

  public static void assertArrayEquals(Object[] expected, Object[] actual) {
    assertThat(actual).isEqualTo(expected);
  }

}
