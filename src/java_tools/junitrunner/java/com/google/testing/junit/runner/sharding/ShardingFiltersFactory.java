// Copyright 2016 The Bazel Authors. All Rights Reserved.
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
import com.google.testing.junit.runner.util.Factory;
import com.google.testing.junit.runner.util.Supplier;

/**
 * A factory that supplies a {@link ShardingFilters}.
 */
public final class ShardingFiltersFactory implements Factory<ShardingFilters> {
  private final Supplier<ShardingEnvironment> shardingEnvironmentSupplier;

  private final Supplier<ShardingFilterFactory> defaultShardingStrategySupplier;

  public ShardingFiltersFactory(
      Supplier<ShardingEnvironment> shardingEnvironmentSupplier,
      Supplier<ShardingFilterFactory> defaultShardingStrategySupplier) {
    assert shardingEnvironmentSupplier != null;
    this.shardingEnvironmentSupplier = shardingEnvironmentSupplier;
    assert defaultShardingStrategySupplier != null;
    this.defaultShardingStrategySupplier = defaultShardingStrategySupplier;
  }

  @Override
  public ShardingFilters get() {
    return new ShardingFilters(
        shardingEnvironmentSupplier.get(), defaultShardingStrategySupplier.get());
  }

  public static Factory<ShardingFilters> create(
      Supplier<ShardingEnvironment> shardingEnvironmentSupplier,
      Supplier<ShardingFilterFactory> defaultShardingStrategySupplier) {
    return new ShardingFiltersFactory(
        shardingEnvironmentSupplier, defaultShardingStrategySupplier);
  }
}
