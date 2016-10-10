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

import com.google.testing.junit.runner.sharding.api.ShardingFilterFactory;
import com.google.testing.junit.runner.sharding.api.WeightStrategy;
import com.google.testing.util.RuntimeCost;
import java.util.Collection;
import org.junit.Ignore;
import org.junit.runner.Description;
import org.junit.runner.manipulation.Filter;

/**
 * A factory that creates a {@link WeightedShardingFilter} that extracts the weight for a test from
 * the {@link RuntimeCost} annotations present in descriptions of tests.
 */
public final class BinStackingShardingFilterFactory implements ShardingFilterFactory {
  static final String DEFAULT_TEST_WEIGHT_PROPERTY = "test.sharding.default_weight";
  static final int DEFAULT_TEST_WEIGHT = 1;

  private final int defaultTestWeight;

  public BinStackingShardingFilterFactory() {
    this(getDefaultTestWeight());
  }

  // VisibleForTesting
  BinStackingShardingFilterFactory(int defaultTestWeight) {
    this.defaultTestWeight = defaultTestWeight;
  }

  static int getDefaultTestWeight() {
    String property = System.getProperty(DEFAULT_TEST_WEIGHT_PROPERTY);
    if (property != null) {
      return Integer.parseInt(property);
    }
    return DEFAULT_TEST_WEIGHT;
  }

  @Override
  public Filter createFilter(
      Collection<Description> testDescriptions, int shardIndex, int totalShards) {
    return new WeightedShardingFilter(
        testDescriptions,
        shardIndex,
        totalShards,
        new RuntimeCostWeightStrategy(defaultTestWeight));
  }

  static class RuntimeCostWeightStrategy implements WeightStrategy {

    private final int defaultTestWeight;

    RuntimeCostWeightStrategy(int defaultTestWeight) {
      this.defaultTestWeight = defaultTestWeight;
    }

    @Override
    public int getDescriptionWeight(Description description) {
      RuntimeCost runtimeCost = description.getAnnotation(RuntimeCost.class);
      Ignore ignore = description.getAnnotation(Ignore.class);

      if (runtimeCost == null || ignore != null) {
        return defaultTestWeight;
      } else {
        return runtimeCost.value();
      }
    }
  }
}
