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
import java.util.Collection;
import java.util.Locale;
import org.junit.runner.Description;
import org.junit.runner.manipulation.Filter;

/**
 * A factory for test sharding filters.
 */
public class ShardingFilters {

  /**
   * An enum of strategies for generating test sharding filters.
   */
  public static enum ShardingStrategy implements ShardingFilterFactory {

    /**
     * {@link com.google.testing.junit.runner.sharding.HashBackedShardingFilter}
     */
    HASH {
      @Override
      public Filter createFilter(Collection<Description> testDescriptions,
          int shardIndex, int totalShards) {
        return new HashBackedShardingFilter(shardIndex, totalShards);
      }
    },

    /**
     * {@link com.google.testing.junit.runner.sharding.RoundRobinShardingFilter}
     */
    ROUND_ROBIN {
      @Override
      public Filter createFilter(Collection<Description> testDescriptions,
          int shardIndex, int totalShards) {
        return new RoundRobinShardingFilter(testDescriptions, shardIndex, totalShards);
      }
    }
  }

  public static final ShardingFilterFactory DEFAULT_SHARDING_STRATEGY =
      ShardingStrategy.ROUND_ROBIN;
  private final ShardingEnvironment shardingEnvironment;
  private final ShardingFilterFactory defaultShardingStrategy;

  /**
   * Creates a factory with the given sharding environment and the
   * default sharding strategy.
   */
  public ShardingFilters(ShardingEnvironment shardingEnvironment) {
    this(shardingEnvironment, DEFAULT_SHARDING_STRATEGY);
  }

  /**
   * Creates a factory with the given sharding environment and sharding
   * strategy.
   */
  public ShardingFilters(ShardingEnvironment shardingEnvironment,
      ShardingFilterFactory defaultShardingStrategy) {
    this.shardingEnvironment = shardingEnvironment;
    this.defaultShardingStrategy = defaultShardingStrategy;
  }

  /**
   * Creates a sharding filter according to strategy specified by the
   * sharding environment.
   */
  public Filter createShardingFilter(Collection<Description> descriptions) {
    ShardingFilterFactory factory = getShardingFilterFactory();
    return factory.createFilter(descriptions, shardingEnvironment.getShardIndex(),
        shardingEnvironment.getTotalShards());
  }

  private ShardingFilterFactory getShardingFilterFactory() {
    String strategy = shardingEnvironment.getTestShardingStrategy();
    if (strategy == null) {
      return defaultShardingStrategy;
    }
    ShardingFilterFactory shardingFilterFactory;
    try {
      shardingFilterFactory = ShardingStrategy.valueOf(strategy.toUpperCase(Locale.ENGLISH));
    } catch (IllegalArgumentException e) {
      try {
        ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
        Class<? extends ShardingFilterFactory> strategyClass =
            classLoader.loadClass(strategy).asSubclass(ShardingFilterFactory.class);
        shardingFilterFactory = strategyClass.getConstructor().newInstance();
      } catch (ReflectiveOperationException | IllegalArgumentException e2) {
        throw new RuntimeException(
            "Could not create custom sharding strategy class " + strategy, e2);
      }
    }
    return shardingFilterFactory;
  }
}
