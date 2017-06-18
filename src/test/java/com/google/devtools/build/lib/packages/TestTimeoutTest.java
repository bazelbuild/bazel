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
package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.packages.TestTimeout.ETERNAL;
import static com.google.devtools.build.lib.packages.TestTimeout.LONG;
import static com.google.devtools.build.lib.packages.TestTimeout.MODERATE;
import static com.google.devtools.build.lib.packages.TestTimeout.SHORT;
import static com.google.devtools.build.lib.packages.TestTimeout.getSuggestedTestTimeout;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests the various methods of {@link TestTimeout}
 */
@RunWith(JUnit4.class)
public class TestTimeoutTest {

  @Test
  public void testBasicConversion() throws Exception {
    assertThat(TestTimeout.valueOf("SHORT")).isSameAs(SHORT);
    assertThat(TestTimeout.valueOf("MODERATE")).isSameAs(MODERATE);
    assertThat(TestTimeout.valueOf("LONG")).isSameAs(LONG);
    assertThat(TestTimeout.valueOf("ETERNAL")).isSameAs(ETERNAL);
  }

  @Test
  public void testSuggestedTestSize() throws Exception {
    assertThat(getSuggestedTestTimeout(0)).isEqualTo(SHORT);
    assertThat(getSuggestedTestTimeout(2)).isEqualTo(SHORT);
    assertThat(getSuggestedTestTimeout(6)).isEqualTo(SHORT);
    assertThat(getSuggestedTestTimeout(59)).isEqualTo(SHORT);
    assertThat(getSuggestedTestTimeout(60)).isEqualTo(MODERATE);
    assertThat(getSuggestedTestTimeout(299)).isEqualTo(MODERATE);
    assertThat(getSuggestedTestTimeout(300)).isEqualTo(LONG);
    assertThat(getSuggestedTestTimeout(899)).isEqualTo(LONG);
    assertThat(getSuggestedTestTimeout(900)).isEqualTo(ETERNAL);
    assertThat(getSuggestedTestTimeout(1234567890)).isEqualTo(ETERNAL);
  }

  @Test
  public void testIsInRangeExact() throws Exception {
    assertThat(SHORT.isInRangeExact(0)).isTrue();
    assertThat(SHORT.isInRangeExact(1)).isTrue();
    assertThat(SHORT.isInRangeExact(60)).isFalse();
    assertThat(MODERATE.isInRangeExact(60)).isTrue();
    assertThat(MODERATE.isInRangeExact(299)).isTrue();
    assertThat(MODERATE.isInRangeExact(300)).isFalse();
    assertThat(LONG.isInRangeExact(300)).isTrue();
    assertThat(LONG.isInRangeExact(899)).isTrue();
    assertThat(LONG.isInRangeExact(900)).isFalse();
    assertThat(ETERNAL.isInRangeExact(900)).isTrue();
    assertThat(ETERNAL.isInRangeExact(1234567890)).isFalse();
  }

  @Test
  public void testIsInRangeFuzzy() throws Exception {
    assertFuzzyRange(SHORT, 0, 105);
    assertFuzzyRange(MODERATE, 8, 525);
    assertFuzzyRange(LONG, 75, 1575);
    assertFuzzyRange(ETERNAL, 225, Integer.MAX_VALUE);
  }

  private void assertFuzzyRange(TestTimeout timeout, int min, int max) {
    if (min > 0) {
      assertThat(timeout.isInRangeFuzzy(min - 1)).isFalse();
    }
    assertThat(timeout.isInRangeFuzzy(min)).isTrue();
    assertThat(timeout.isInRangeFuzzy(min + 1)).isTrue();
    assertThat(timeout.isInRangeFuzzy(max - 1)).isTrue();
    assertThat(timeout.isInRangeFuzzy(max)).isTrue();
    if (max < Integer.MAX_VALUE) {
      assertThat(timeout.isInRangeFuzzy(max + 1)).isFalse();
    }
  }
}
