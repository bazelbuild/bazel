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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PythonVersion}. */
@RunWith(JUnit4.class)
public class PythonVersionTest {

  private static void assertIsInvalidForParseTargetValue(String value) {
    assertThat(
            assertThrows(
                IllegalArgumentException.class, () -> PythonVersion.parseTargetValue(value)))
        .hasMessageThat()
        .contains("not a valid Python major version");
  }

  private static void assertIsInvalidForParseTargetOrSentinelValue(String value) {
    assertThat(
            assertThrows(
                IllegalArgumentException.class,
                () -> PythonVersion.parseTargetOrSentinelValue(value)))
        .hasMessageThat()
        .contains("not a valid Python major version");
  }

  private static void assertIsInvalidForParseSrcsValue(String value) {
    assertThat(
            assertThrows(IllegalArgumentException.class, () -> PythonVersion.parseSrcsValue(value)))
        .hasMessageThat()
        .contains("not a valid Python srcs_version value");
  }

  @Test
  public void isTargetValue() {
    assertThat(PythonVersion.PY2.isTargetValue()).isTrue();
    assertThat(PythonVersion.PY3.isTargetValue()).isTrue();
    assertThat(PythonVersion.PY2AND3.isTargetValue()).isFalse();
    assertThat(PythonVersion.PY2ONLY.isTargetValue()).isFalse();
    assertThat(PythonVersion.PY3ONLY.isTargetValue()).isFalse();
    assertThat(PythonVersion._INTERNAL_SENTINEL.isTargetValue()).isFalse();
  }

  @Test
  public void parseTargetValue() {
    assertThat(PythonVersion.parseTargetValue("PY2")).isEqualTo(PythonVersion.PY2);
    assertThat(PythonVersion.parseTargetValue("PY3")).isEqualTo(PythonVersion.PY3);
    assertIsInvalidForParseTargetValue("PY2AND3");
    assertIsInvalidForParseTargetValue("PY2ONLY");
    assertIsInvalidForParseTargetValue("PY3ONLY");
    assertIsInvalidForParseTargetValue("_INTERNAL_SENTINEL");
    assertIsInvalidForParseTargetValue("not an enum value");
  }

  @Test
  public void parseTargetOrSentinelValue() {
    assertThat(PythonVersion.parseTargetOrSentinelValue("PY2")).isEqualTo(PythonVersion.PY2);
    assertThat(PythonVersion.parseTargetOrSentinelValue("PY3")).isEqualTo(PythonVersion.PY3);
    assertIsInvalidForParseTargetOrSentinelValue("PY2AND3");
    assertIsInvalidForParseTargetOrSentinelValue("PY2ONLY");
    assertIsInvalidForParseTargetOrSentinelValue("PY3ONLY");
    assertThat(PythonVersion.parseTargetOrSentinelValue("_INTERNAL_SENTINEL"))
        .isEqualTo(PythonVersion._INTERNAL_SENTINEL);
    assertIsInvalidForParseTargetOrSentinelValue("not an enum value");
  }

  @Test
  public void parseSrcsValue() {
    assertThat(PythonVersion.parseSrcsValue("PY2")).isEqualTo(PythonVersion.PY2);
    assertThat(PythonVersion.parseSrcsValue("PY3")).isEqualTo(PythonVersion.PY3);
    assertThat(PythonVersion.parseSrcsValue("PY2AND3")).isEqualTo(PythonVersion.PY2AND3);
    assertThat(PythonVersion.parseSrcsValue("PY2ONLY")).isEqualTo(PythonVersion.PY2ONLY);
    assertThat(PythonVersion.parseSrcsValue("PY3ONLY")).isEqualTo(PythonVersion.PY3ONLY);
    assertIsInvalidForParseSrcsValue("_INTERNAL_SENTINEL");
    assertIsInvalidForParseSrcsValue("not an enum value");
  }
}
