// Copyright 2009 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.packages.TestTimeout.ETERNAL;
import static com.google.devtools.build.lib.packages.TestTimeout.LONG;
import static com.google.devtools.build.lib.packages.TestTimeout.MODERATE;
import static com.google.devtools.build.lib.packages.TestTimeout.SHORT;
import static com.google.devtools.build.lib.packages.TestTimeout.getSuggestedTestTimeout;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

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
    assertSame(SHORT, TestTimeout.valueOf("SHORT"));
    assertSame(MODERATE, TestTimeout.valueOf("MODERATE"));
    assertSame(LONG, TestTimeout.valueOf("LONG"));
    assertSame(ETERNAL, TestTimeout.valueOf("ETERNAL"));
  }

  @Test
  public void testSuggestedTestSize() throws Exception {
    assertEquals(SHORT, getSuggestedTestTimeout(0));
    assertEquals(SHORT, getSuggestedTestTimeout(2));
    assertEquals(SHORT, getSuggestedTestTimeout(6));
    assertEquals(SHORT, getSuggestedTestTimeout(59));
    assertEquals(MODERATE, getSuggestedTestTimeout(60));
    assertEquals(MODERATE, getSuggestedTestTimeout(299));
    assertEquals(LONG, getSuggestedTestTimeout(300));
    assertEquals(LONG, getSuggestedTestTimeout(899));
    assertEquals(ETERNAL, getSuggestedTestTimeout(900));
    assertEquals(ETERNAL, getSuggestedTestTimeout(1234567890));
  }

  @Test
  public void testIsInRangeExact() throws Exception {
    assertTrue(SHORT.isInRangeExact(0));
    assertTrue(SHORT.isInRangeExact(1));
    assertFalse(SHORT.isInRangeExact(60));
    assertTrue(MODERATE.isInRangeExact(60));
    assertTrue(MODERATE.isInRangeExact(299));
    assertFalse(MODERATE.isInRangeExact(300));
    assertTrue(LONG.isInRangeExact(300));
    assertTrue(LONG.isInRangeExact(899));
    assertFalse(LONG.isInRangeExact(900));
    assertTrue(ETERNAL.isInRangeExact(900));
    assertFalse(ETERNAL.isInRangeExact(1234567890));
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
      assertFalse(timeout.isInRangeFuzzy(min - 1));
    }
    assertTrue(timeout.isInRangeFuzzy(min));
    assertTrue(timeout.isInRangeFuzzy(min + 1));
    assertTrue(timeout.isInRangeFuzzy(max - 1));
    assertTrue(timeout.isInRangeFuzzy(max));
    if (max < Integer.MAX_VALUE) {
      assertFalse(timeout.isInRangeFuzzy(max + 1));
    }
  }
}
