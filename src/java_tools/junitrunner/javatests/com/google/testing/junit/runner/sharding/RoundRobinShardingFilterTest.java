// Copyright 2010 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.sharding;

import com.google.testing.junit.runner.sharding.api.ShardingFilterFactory;
import com.google.testing.junit.runner.sharding.testing.RoundRobinShardingFilterFactory;
import com.google.testing.junit.runner.sharding.testing.ShardingFilterTestCase;

import org.junit.runner.Description;
import org.junit.runner.manipulation.Filter;

import java.util.List;
import java.util.Map;

/**
 * Tests for the {@link RoundRobinShardingFilter}.
 */
public class RoundRobinShardingFilterTest extends ShardingFilterTestCase {

  private static final List<Description> GENERIC_TEST_DESCRIPTIONS =
      ShardingFilterTestCase.createGenericTestCaseDescriptions(6);

  private static final List<Filter> FILTERS_1 =
      createFilters(GENERIC_TEST_DESCRIPTIONS, 3, new RoundRobinShardingFilterFactory());
  private static final List<Filter> FILTERS_2 =
      createFilters(GENERIC_TEST_DESCRIPTIONS, 4, new RoundRobinShardingFilterFactory());

  public void testShardingIsBalanced() {
    Map<Filter, List<Description>> run1 = simulateTestRun(FILTERS_1, GENERIC_TEST_DESCRIPTIONS);
    assertEquals(2, run1.get(FILTERS_1.get(0)).size());
    assertEquals(2, run1.get(FILTERS_1.get(1)).size());
    assertEquals(2, run1.get(FILTERS_1.get(2)).size());

    Map<Filter, List<Description>> run2 = simulateTestRun(FILTERS_2, GENERIC_TEST_DESCRIPTIONS);
    assertEquals(2, run2.get(FILTERS_2.get(0)).size());
    assertEquals(2, run2.get(FILTERS_2.get(1)).size());
    assertEquals(1, run2.get(FILTERS_2.get(2)).size());
    assertEquals(1, run2.get(FILTERS_2.get(3)).size());
  }

  public void testShouldRun_throwsExceptionForUnknownDescription() {
    assertThrowsExceptionForUnknownDescription(FILTERS_1.get(0));
  }

  @Override
  protected ShardingFilterFactory createShardingFilterFactory() {
    return new RoundRobinShardingFilterFactory();
  }
}
