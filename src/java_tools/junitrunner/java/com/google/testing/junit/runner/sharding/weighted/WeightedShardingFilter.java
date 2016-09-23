// Copyright 2015 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.sharding.weighted;

import com.google.testing.junit.runner.sharding.api.WeightStrategy;
import com.google.testing.util.RuntimeCost;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import org.junit.runner.Description;
import org.junit.runner.manipulation.Filter;

/**
 * A sharding function that attempts to evenly use time on all available
 * shards while considering the test's weight.
 *
 * <p>When all tests have the same weight the sharding function behaves
 * similarly to round robin.
 */
public final class WeightedShardingFilter extends Filter {
  private final Map<Description, Integer> testToShardMap;
  private final int shardIndex;

  public WeightedShardingFilter(Collection<Description> descriptions, int shardIndex,
      int totalShards, WeightStrategy weightStrategy) {
    if (shardIndex < 0 || totalShards <= shardIndex) {
      throw new IllegalArgumentException();
    }
    this.shardIndex = shardIndex;
    this.testToShardMap = buildTestToShardMap(descriptions, totalShards, weightStrategy);
  }

  @Override
  public String describe() {
    return "bin stacking filter";
  }

  @Override
  public boolean shouldRun(Description description) {
    if (description.isSuite()) {
      return true;
    }
    Integer shardForTest = testToShardMap.get(description);
    if (shardForTest == null) {
      throw new IllegalArgumentException("This filter keeps a mapping from each test "
          + "description to a shard, and the given description was not passed in when "
          + "filter was constructed: " + description);
    }
    return shardForTest == shardIndex;
  }

  private static Map<Description, Integer> buildTestToShardMap(
      Collection<Description> descriptions, int numShards, WeightStrategy weightStrategy) {
    Map<Description, Integer> map = new HashMap<>();

    // Sorting this list is incredibly important to correctness. Otherwise,
    // "shuffled" suites would break the sharding protocol.
    List<Description> sortedDescriptions = new ArrayList<>(descriptions);
    Collections.sort(sortedDescriptions, new WeightClassAndTestNameComparator(weightStrategy));

    PriorityQueue<Shard> queue = new PriorityQueue<>(numShards);
    for (int i = 0; i < numShards; i++) {
      queue.offer(new Shard(i));
    }

    // If we get two descriptions that are equal, the shard number for the second
    // one will overwrite the shard number for the first.  Thus they'll run on the
    // same shard.
    for (Description description : sortedDescriptions) {
      if (!description.isTest()) {
        throw new IllegalArgumentException("Test suite should not be included in the set of tests "
            + "to shard: " + description.getDisplayName());
      }

      Shard shard = queue.remove();
      shard.addWeight(weightStrategy.getDescriptionWeight(description));
      queue.offer(shard);
      map.put(description, shard.getIndex());
    }
    return Collections.unmodifiableMap(map);
  }

  /**
   * A comparator that sorts by weight in descending order, then by test case name.
   */
  private static class WeightClassAndTestNameComparator implements Comparator<Description> {

    private final WeightStrategy weightStrategy;

    WeightClassAndTestNameComparator(WeightStrategy weightStrategy) {
      this.weightStrategy = weightStrategy;
    }

    @Override
    public int compare(Description d1, Description d2) {
      int weight1 = weightStrategy.getDescriptionWeight(d1);
      int weight2 = weightStrategy.getDescriptionWeight(d2);
      if (weight1 != weight2) {
        // We consider the reverse order when comparing weights.
        return -1 * compareInts(weight1, weight2);
      }
      return d1.getDisplayName().compareTo(d2.getDisplayName());
    }
  }

  /**
   * A bean representing the sum of {@link RuntimeCost}s assigned to a shard.
   */
  private static class Shard implements Comparable<Shard> {
    private final int index;
    private int weight = 0;

    Shard(int index) {
      this.index = index;
    }

    void addWeight(int weight) {
      this.weight += weight;
    }

    int getIndex() {
      return index;
    }

    @Override
    public int compareTo(Shard other) {
      if (weight != other.weight) {
        return compareInts(weight, other.weight);
      }
      return compareInts(index, other.index);
    }
  }

  private static int compareInts(int value1, int value2) {
    return value1 - value2;
  }
}
