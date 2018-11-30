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

  @Test
  public void parseTargetValue() {
    assertThat(PythonVersion.parseTargetValue("PY2")).isEqualTo(PythonVersion.PY2);

    IllegalArgumentException expected =
        assertThrows(
            IllegalArgumentException.class, () -> PythonVersion.parseTargetValue("PY2AND3"));
    assertThat(expected).hasMessageThat().contains("not a valid Python major version");

    expected =
        assertThrows(
            IllegalArgumentException.class,
            () -> PythonVersion.parseTargetValue("not an enum value"));
    assertThat(expected).hasMessageThat().contains("not a valid Python major version");

    expected =
        assertThrows(IllegalArgumentException.class, () -> PythonVersion.parseTargetValue("py2"));
    assertThat(expected).hasMessageThat().contains("not a valid Python major version");
  }

  @Test
  public void parseSrcsValue() {
    assertThat(PythonVersion.parseSrcsValue("PY2")).isEqualTo(PythonVersion.PY2);

    assertThat(PythonVersion.parseSrcsValue("PY2AND3")).isEqualTo(PythonVersion.PY2AND3);

    IllegalArgumentException expected =
        assertThrows(
            IllegalArgumentException.class,
            () -> PythonVersion.parseSrcsValue("not an enum value"));
    assertThat(expected).hasMessageThat().contains("No enum constant");

    expected =
        assertThrows(IllegalArgumentException.class, () -> PythonVersion.parseSrcsValue("py2"));
    assertThat(expected).hasMessageThat().contains("No enum constant");
  }
}
