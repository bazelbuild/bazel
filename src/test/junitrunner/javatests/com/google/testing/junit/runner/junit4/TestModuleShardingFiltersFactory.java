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

package com.google.testing.junit.runner.junit4;

import com.google.testing.junit.runner.sharding.ShardingEnvironment;
import com.google.testing.junit.runner.sharding.ShardingFilters;
import com.google.testing.junit.runner.sharding.api.ShardingFilterFactory;
import com.google.testing.junit.runner.util.Factory;
import com.google.testing.junit.runner.util.Supplier;

/**
 * A factory that supplies a {@link ShardingFilters} for testing purposes.
 */
public final class TestModuleShardingFiltersFactory implements Factory<ShardingFilters> {
  private final JUnit4RunnerTest.TestModule module;

  private final Supplier<ShardingEnvironment> shardingEnvironmentSupplier;

  private final Supplier<ShardingFilterFactory> defaultShardingStrategySupplier;

  public TestModuleShardingFiltersFactory(
      JUnit4RunnerTest.TestModule module,
      Supplier<ShardingEnvironment> shardingEnvironmentSupplier,
      Supplier<ShardingFilterFactory> defaultShardingStrategySupplier) {
    assert module != null;
    this.module = module;
    assert shardingEnvironmentSupplier != null;
    this.shardingEnvironmentSupplier = shardingEnvironmentSupplier;
    assert defaultShardingStrategySupplier != null;
    this.defaultShardingStrategySupplier = defaultShardingStrategySupplier;
  }

  @Override
  public ShardingFilters get() {
    ShardingFilters shardingFilters = module.shardingFilters(
        shardingEnvironmentSupplier.get(), defaultShardingStrategySupplier.get());
    if (shardingFilters == null) {
      throw new NullPointerException();
    }
    return shardingFilters;
  }

  public static Factory<ShardingFilters> create(
      JUnit4RunnerTest.TestModule module,
      Supplier<ShardingEnvironment> shardingEnvironmentSupplier,
      Supplier<ShardingFilterFactory> defaultShardingStrategySupplier) {
    return new TestModuleShardingFiltersFactory(
        module, shardingEnvironmentSupplier, defaultShardingStrategySupplier);
  }
}