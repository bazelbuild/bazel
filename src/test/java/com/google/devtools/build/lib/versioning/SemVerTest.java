// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.versioning;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SemVer}. */
@RunWith(JUnit4.class)
public class SemVerTest {

  @Test
  public void testFrom_unspecifiedComponentsAreZero() {
    assertThat(SemVer.from(8)).isEqualTo(SemVer.from(8, 0, 0));
    assertThat(SemVer.from(9, 15)).isEqualTo(SemVer.from(9, 15, 0));
  }

  @Test
  public void testParse_ok() throws Exception {
    assertThat(SemVer.parse("1")).isEqualTo(SemVer.from(1));
    assertThat(SemVer.parse("2.3")).isEqualTo(SemVer.from(2, 3));
    assertThat(SemVer.parse("3.5.1")).isEqualTo(SemVer.from(3, 5, 1));
    assertThat(SemVer.parse("91.582.0945")).isEqualTo(SemVer.from(91, 582, 945));
  }

  @Test
  public void testParse_errors() {
    for (String s : new String[] {"", "foo", "1..2", "1.2.", "1.-1.2"}) {
      ParseException e = assertThrows(ParseException.class, () -> SemVer.parse(s));
      assertThat(e).hasMessageThat().contains("Invalid semver");
    }

    String bigint = "5000000000"; // Larger than Integer.MAX_VALUE.
    for (String s : new String[] {bigint + ".0.0", "0." + bigint + ".0", "0.0." + bigint}) {
      ParseException e = assertThrows(ParseException.class, () -> SemVer.parse(s));
      assertThat(e).hasMessageThat().contains("Invalid number in semver component");
    }
  }

  @Test
  public void testComparison_oneComponent() {
    assertThat(SemVer.from(1, 0, 0)).isLessThan(SemVer.from(2, 0, 0));
    assertThat(SemVer.from(1, 0, 0)).isEqualTo(SemVer.from(1, 0, 0));
    assertThat(SemVer.from(2, 0, 0)).isGreaterThan(SemVer.from(1, 0, 0));

    assertThat(SemVer.from(0, 1, 0)).isLessThan(SemVer.from(0, 2, 0));
    assertThat(SemVer.from(0, 1, 0)).isEqualTo(SemVer.from(0, 1, 0));
    assertThat(SemVer.from(0, 2, 0)).isGreaterThan(SemVer.from(0, 1, 0));

    assertThat(SemVer.from(0, 0, 1)).isLessThan(SemVer.from(0, 0, 2));
    assertThat(SemVer.from(0, 0, 1)).isEqualTo(SemVer.from(0, 0, 1));
    assertThat(SemVer.from(0, 0, 2)).isGreaterThan(SemVer.from(0, 0, 1));
  }

  @Test
  public void testComparison_multipleComponents() {
    assertThat(SemVer.from(1, 0, 0)).isLessThan(SemVer.from(1, 1, 0));
    assertThat(SemVer.from(1, 1, 0)).isGreaterThan(SemVer.from(1, 0, 0));

    assertThat(SemVer.from(1, 0, 0)).isLessThan(SemVer.from(1, 0, 1));
    assertThat(SemVer.from(1, 0, 1)).isGreaterThan(SemVer.from(1, 0, 0));
  }

  @Test
  public void testToString_keepsAllComponents() {
    assertThat(SemVer.from(0, 0, 0).toString()).isEqualTo("0.0.0");
    assertThat(SemVer.from(7, 1, 8).toString()).isEqualTo("7.1.8");
  }
}
