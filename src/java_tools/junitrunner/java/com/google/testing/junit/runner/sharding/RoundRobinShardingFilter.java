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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

import org.junit.runner.Description;
import org.junit.runner.manipulation.Filter;

import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/**
 * Implements the round-robin sharding strategy.
 *
 * <p>This is done by equally dividing up the tests across all the shards
 * Each test is numbered and the test number is modded with the number of
 * shards and checked against the shard number to see whether it should run
 * on a particular shard.
 *
 * <p>Equals and hashCode implementations are not necessary for correct
 * sharding, but are done so that this filter can be compared in tests.
 */
public final class RoundRobinShardingFilter extends Filter {

  @VisibleForTesting
  final Map<Description, Integer> testToShardMap;
  @VisibleForTesting
  final int shardIndex;
  @VisibleForTesting
  final int totalShards;

  public RoundRobinShardingFilter(Collection<Description> testDescriptions,
      int shardIndex, int totalShards) {
    Preconditions.checkArgument(shardIndex >= 0);
    Preconditions.checkArgument(totalShards > shardIndex);
    this.testToShardMap = buildTestToShardMap(testDescriptions);
    this.shardIndex = shardIndex;
    this.totalShards = totalShards;
  }

  /**
   * Given a list of test case descriptions, returns a mapping from each
   * to its index in the list.
   */
  private static Map<Description, Integer> buildTestToShardMap(
      Collection<Description> testDescriptions) {
    Map<Description, Integer> map = Maps.newHashMap();

    // Sorting this list is incredibly important to correctness. Otherwise,
    // "shuffled" suites would break the sharding protocol.
    List<Description> sortedDescriptions = Lists.newArrayList(testDescriptions);
    Collections.sort(sortedDescriptions, new DescriptionComparator());

    // If we get two descriptions that are equal, the shard number for the second
    // one will overwrite the shard number for the first.  Thus they'll run on the
    // same shard.
    int index = 0;
    for (Description description : sortedDescriptions) {
      Preconditions.checkArgument(description.isTest(),
          "Test suite should not be included in the set of tests to shard: %s",
          description.getDisplayName());
      map.put(description, index);
      index++;
    }
    return Collections.unmodifiableMap(map);
  }

  @Override
  public boolean shouldRun(Description description) {
    if (description.isSuite()) {
      return true;
    }
    Integer testNumber = testToShardMap.get(description);
    if (testNumber == null) {
      throw new IllegalArgumentException("This filter keeps a mapping from each test "
          + "description to a shard, and the given description was not passed in when "
          + "filter was constructed: " + description);
    }
    return (testNumber % totalShards) == shardIndex;
  }

  @Override
  public String describe() {
    return "round robin sharding filter";
  }

  @VisibleForTesting
  static class DescriptionComparator implements Comparator<Description> {
    @Override
    public int compare(Description d1, Description d2) {
      return d1.getDisplayName().compareTo(d2.getDisplayName());
    }
  }

}
