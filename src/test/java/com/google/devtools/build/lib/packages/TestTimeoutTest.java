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
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.packages.TestTimeout.ETERNAL;
import static com.google.devtools.build.lib.packages.TestTimeout.LONG;
import static com.google.devtools.build.lib.packages.TestTimeout.MODERATE;
import static com.google.devtools.build.lib.packages.TestTimeout.SHORT;
import static com.google.devtools.build.lib.packages.TestTimeout.getSuggestedTestTimeout;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
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
    assertThat(getSuggestedTestTimeout(30)).isEqualTo(SHORT);
    assertThat(getSuggestedTestTimeout(50)).isEqualTo(MODERATE);
    assertThat(getSuggestedTestTimeout(250)).isEqualTo(LONG);
    assertThat(getSuggestedTestTimeout(700)).isEqualTo(ETERNAL);
    assertThat(getSuggestedTestTimeout(60 * 60 * 24 * 360)).isEqualTo(ETERNAL);
  }

  @Test
  public void testAllTimesHaveSuggestions() throws Exception {
    for (int timeout = 0; timeout < ETERNAL.getTimeoutSeconds(); timeout++) {
      TestTimeout suggested = getSuggestedTestTimeout(timeout);
      assertWithMessage("No suggested TestTimeout found for timeout " + timeout)
          .that(suggested)
          .isNotNull();
      assertWithMessage(
              "Suggested timeout " + suggested + " is not in the fuzzy range for " + timeout)
          .that(suggested.isInRangeFuzzy(timeout))
          .isTrue();
    }
  }

  @Test
  public void testIsInRangeFuzzy() throws Exception {
    assertThat(SHORT.isInRangeFuzzy(0)).isTrue();
    assertThat(SHORT.isInRangeFuzzy(30)).isTrue();
    assertThat(SHORT.isInRangeFuzzy(55)).isFalse();
    assertThat(MODERATE.isInRangeFuzzy(10)).isFalse();
    assertThat(MODERATE.isInRangeFuzzy(40)).isTrue();
    assertThat(MODERATE.isInRangeFuzzy(290)).isFalse();
    assertThat(LONG.isInRangeFuzzy(30)).isFalse();
    assertThat(LONG.isInRangeFuzzy(200)).isTrue();
    assertThat(LONG.isInRangeFuzzy(890)).isFalse();
    assertThat(ETERNAL.isInRangeFuzzy(50)).isFalse();
    assertThat(ETERNAL.isInRangeFuzzy(500)).isTrue();
    assertThat(ETERNAL.isInRangeFuzzy(3500)).isTrue();
    assertThat(ETERNAL.isInRangeFuzzy(60 * 60 * 24 * 360)).isTrue();
  }

  @Test
  public void testAllFuzzyRangesCovered() throws Exception {
    for (int timeout = 0; timeout < ETERNAL.getTimeoutSeconds(); timeout++) {
      List<Boolean> truthValues = new ArrayList<>();
      for (TestTimeout testTimeout : Arrays.asList(SHORT, MODERATE, LONG, ETERNAL)) {
        truthValues.add(testTimeout.isInRangeFuzzy(timeout));
      }
      assertWithMessage("Timeout " + timeout + " is not in any fuzzy range.")
          .that(truthValues)
          .contains(true);
    }
  }
}
