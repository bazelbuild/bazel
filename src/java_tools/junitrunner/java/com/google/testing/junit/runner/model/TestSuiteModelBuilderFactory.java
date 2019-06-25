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

package com.google.testing.junit.runner.model;

import com.google.testing.junit.runner.sharding.ShardingEnvironment;
import com.google.testing.junit.runner.sharding.ShardingFilters;
import com.google.testing.junit.runner.util.Factory;
import com.google.testing.junit.runner.util.Supplier;
import com.google.testing.junit.runner.util.TestClock;

/**
 * A factory that supplies a top level suite {@link TestSuiteModel.Builder}.
 */
public final class TestSuiteModelBuilderFactory implements Factory<TestSuiteModel.Builder> {
  private final Supplier<TestClock> tickerSupplier;

  private final Supplier<ShardingFilters> shardingFiltersSupplier;

  private final Supplier<ShardingEnvironment> shardingEnvironmentSupplier;

  private final Supplier<XmlResultWriter> xmlResultWriterSupplier;

  public TestSuiteModelBuilderFactory(
      Supplier<TestClock> tickerSupplier,
      Supplier<ShardingFilters> shardingFiltersSupplier,
      Supplier<ShardingEnvironment> shardingEnvironmentSupplier,
      Supplier<XmlResultWriter> xmlResultWriterSupplier) {
    assert tickerSupplier != null;
    this.tickerSupplier = tickerSupplier;
    assert shardingFiltersSupplier != null;
    this.shardingFiltersSupplier = shardingFiltersSupplier;
    assert shardingEnvironmentSupplier != null;
    this.shardingEnvironmentSupplier = shardingEnvironmentSupplier;
    assert xmlResultWriterSupplier != null;
    this.xmlResultWriterSupplier = xmlResultWriterSupplier;
  }

  @Override
  public TestSuiteModel.Builder get() {
    return new TestSuiteModel.Builder(
        tickerSupplier.get(),
        shardingFiltersSupplier.get(),
        shardingEnvironmentSupplier.get(),
        xmlResultWriterSupplier.get());
  }

  public static Factory<TestSuiteModel.Builder> create(
      Supplier<TestClock> tickerSupplier,
      Supplier<ShardingFilters> shardingFiltersSupplier,
      Supplier<ShardingEnvironment> shardingEnvironmentSupplier,
      Supplier<XmlResultWriter> xmlResultWriterSupplier) {
    return new TestSuiteModelBuilderFactory(
        tickerSupplier,
        shardingFiltersSupplier,
        shardingEnvironmentSupplier,
        xmlResultWriterSupplier);
  }
}
