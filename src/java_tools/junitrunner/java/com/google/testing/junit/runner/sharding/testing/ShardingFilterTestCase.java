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

package com.google.testing.junit.runner.sharding.testing;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Lists;
import com.google.testing.junit.runner.sharding.api.ShardingFilterFactory;

import junit.framework.TestCase;

import org.junit.Test;
import org.junit.runner.Description;
import org.junit.runner.manipulation.Filter;

import java.util.Deque;
import java.util.List;

/**
 * Common base class for all sharding filter tests.
 */
public abstract class ShardingFilterTestCase extends TestCase {

  static final List<Description> TEST_DESCRIPTIONS = createGenericTestCaseDescriptions(6);

  /**
   * Returns a filter of the subclass type using the given descriptions,
   * shard index, and total number of shards.
   */
  protected abstract ShardingFilterFactory createShardingFilterFactory();
  
  public final void testShardingIsCompleteAndPartitioned_oneShard() {
    assertShardingIsCompleteAndPartitioned(createFilters(TEST_DESCRIPTIONS, 1), TEST_DESCRIPTIONS);
  }

  public final void testShardingIsStable_oneShard() {
    assertShardingIsStable(createFilters(TEST_DESCRIPTIONS, 1), TEST_DESCRIPTIONS);
  }

  public final void testShardingIsCompleteAndPartitioned_moreTestsThanShards() {
    assertShardingIsCompleteAndPartitioned(createFilters(TEST_DESCRIPTIONS, 5), TEST_DESCRIPTIONS);
  }

  public final void testShardingIsStable_moreTestsThanShards() {
    assertShardingIsStable(createFilters(TEST_DESCRIPTIONS, 5), TEST_DESCRIPTIONS);
  }

  public final void testShardingIsCompleteAndPartitioned_sameNumberOfTestsAndShards() {
    assertShardingIsCompleteAndPartitioned(createFilters(TEST_DESCRIPTIONS, 6), TEST_DESCRIPTIONS);
  }

  public final void testShardingIsStable_sameNumberOfTestsAndShards() {
    assertShardingIsStable(createFilters(TEST_DESCRIPTIONS, 6), TEST_DESCRIPTIONS);
  }

  public final void testShardingIsCompleteAndPartitioned_moreShardsThanTests() {
    assertShardingIsCompleteAndPartitioned(createFilters(TEST_DESCRIPTIONS, 7), TEST_DESCRIPTIONS);
  }

  public final void testShardingIsStable_moreShardsThanTests() {
    assertShardingIsStable(createFilters(TEST_DESCRIPTIONS, 7), TEST_DESCRIPTIONS);
  }

  public final void testShardingIsCompleteAndPartitioned_duplicateDescriptions() {
    ImmutableList<Description> descriptions = ImmutableList.<Description>builder()
        .addAll(createGenericTestCaseDescriptions(6))
        .addAll(createGenericTestCaseDescriptions(6))
        .build();
    assertShardingIsCompleteAndPartitioned(createFilters(descriptions, 7), descriptions);
  }

  public final void testShardingIsStable_duplicateDescriptions() {
    ImmutableList<Description> descriptions = ImmutableList.<Description>builder()
        .addAll(createGenericTestCaseDescriptions(6))
        .addAll(createGenericTestCaseDescriptions(6))
        .build();
    assertShardingIsStable(createFilters(descriptions, 7), descriptions);
  }
  
  public final void testShouldRunTestSuite() {    
    Description testSuiteDescription = createTestSuiteDescription();   
    Filter filter = createShardingFilterFactory().createFilter(TEST_DESCRIPTIONS, 0, 1); 
    assertTrue(filter.shouldRun(testSuiteDescription));    
  }

  /**
   * Creates a list of generic test case descriptions.
   *
   * @param numDescriptions the number of generic test descriptions to add to the list.
   */
  public static List<Description> createGenericTestCaseDescriptions(int numDescriptions) {
    ImmutableList.Builder<Description> builder = ImmutableList.builder();
    for (int i = 0; i < numDescriptions; i++) {
      builder.add(Description.createTestDescription(Test.class, "test" + i));
    }
    return builder.build();
  }
  
  protected static final List<Filter> createFilters(List<Description> descriptions, int numShards,
      ShardingFilterFactory factory) {
    ImmutableList.Builder<Filter> builder = ImmutableList.builder();
    for (int shardIndex = 0; shardIndex < numShards; shardIndex++) {
      builder.add(factory.createFilter(descriptions, shardIndex, numShards));
    }
    return builder.build();
  }
  
  protected final List<Filter> createFilters(List<Description> descriptions, int numShards) {
    return createFilters(descriptions, numShards, createShardingFilterFactory());
  }

  protected static void assertThrowsExceptionForUnknownDescription(Filter filter) {
    try {
      filter.shouldRun(Description.createTestDescription(Object.class, "unknown"));
      fail("expected thrown exception");
    } catch (IllegalArgumentException expected) { }
  }

  /**
   * Simulates test sharding with the given filters and test descriptions.
   *
   * @param filters a list of filters, one per test shard
   * @param descriptions a list of test descriptions
   * @return a mapping from each filter to the descriptions of the tests that would be run
   *   by the shard associated with that filter.
   */
  protected static ListMultimap<Filter, Description> simulateTestRun(List<Filter> filters,
      List<Description> descriptions) {
    ListMultimap<Filter, Description> descriptionsRun = ArrayListMultimap.create();
    for (Filter filter : filters) {
      for (Description description : descriptions) {
        if (filter.shouldRun(description)) {
          descriptionsRun.put(filter, description);
        }
      }
    }
    return descriptionsRun;
  }

  /**
   * Simulates test sharding with the given filters and test descriptions, for a
   * set of test descriptions that is in a different order in every test shard.
   *
   * @param filters a list of filters, one per test shard
   * @param descriptions a list of test descriptions
   * @return a mapping from each filter to the descriptions of the tests that would be run
   *   by the shard associated with that filter.
   */
  protected static ListMultimap<Filter, Description> simulateSelfRandomizingTestRun(
      List<Filter> filters, List<Description> descriptions) {
    if (descriptions.isEmpty()) {
      return ArrayListMultimap.create();
    }
    Deque<Description> mutatingDescriptions = Lists.newLinkedList(descriptions);
    ListMultimap<Filter, Description> descriptionsRun = ArrayListMultimap.create();

    for (Filter filter : filters) {
      // rotate the queue so that each filter gets the descriptions in a different order
      mutatingDescriptions.addLast(mutatingDescriptions.pollFirst());
      for (Description description : descriptions) {
        if (filter.shouldRun(description)) {
          descriptionsRun.put(filter, description);
        }
      }
    }
    return descriptionsRun;
  }

  /**
   * Creates a test suite description (a Description that returns true
   * when {@link org.junit.runner.Description#isSuite()} is called.)
   */
  protected static Description createTestSuiteDescription() {
    Description testSuiteDescription = Description.createSuiteDescription("testSuite");
    testSuiteDescription.addChild(Description.createSuiteDescription("testCase"));
    return testSuiteDescription;
  }

  /**
   * Tests that the sharding is complete (each test is run at least once) and
   * partitioned (each test is run at most once) -- in other words, that
   * each test is run exactly once.  This is a requirement of all test
   * sharding functions.
   */
  protected static void assertShardingIsCompleteAndPartitioned(List<Filter> filters,
      List<Description> descriptions) {
    ListMultimap<Filter, Description> run = simulateTestRun(filters, descriptions);
    assertThat(run.values()).containsExactlyElementsIn(descriptions);

    simulateSelfRandomizingTestRun(filters, descriptions);
    assertThat(run.values()).containsExactlyElementsIn(descriptions);
  }

  /**
   * Tests that sharding is stable for the given filters, regardless of the
   * ordering of the descriptions.  This is useful for verifying that sharding
   * works with self-randomizing test suites, and a requirement of all test
   * sharding functions.
   */
  protected static void assertShardingIsStable(
      List<Filter> filters, List<Description> descriptions) {
    ListMultimap<Filter, Description> run1 = simulateTestRun(filters, descriptions);
    ListMultimap<Filter, Description> run2 = simulateTestRun(filters, descriptions);
    assertEquals(run1, run2);

    ListMultimap<Filter, Description> randomizedRun1 =
        simulateSelfRandomizingTestRun(filters, descriptions);
    ListMultimap<Filter, Description> randomizedRun2 =
        simulateSelfRandomizingTestRun(filters, descriptions);
    assertEquals(randomizedRun1, randomizedRun2);
  }
}
