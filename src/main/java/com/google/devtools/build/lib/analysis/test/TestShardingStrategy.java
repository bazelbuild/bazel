// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.test;

import com.google.devtools.build.lib.packages.TestSize;
import com.google.devtools.common.options.EnumConverter;

/** A strategy for running the same tests in many processes. */
public enum TestShardingStrategy {
  EXPLICIT {
    @Override
    public int getNumberOfShards(
        boolean isLocal, int shardCountFromAttr, boolean testShardingCompliant, TestSize testSize) {
      return Math.max(shardCountFromAttr, 0);
    }
  },

  DISABLED {
    @Override
    public int getNumberOfShards(
        boolean isLocal, int shardCountFromAttr, boolean testShardingCompliant, TestSize testSize) {
      return 0;
    }
  };

  public abstract int getNumberOfShards(
      boolean isLocal, int shardCountFromAttr, boolean testShardingCompliant, TestSize testSize);

  /** Converts to {@link TestShardingStrategy}. */
  public static class ShardingStrategyConverter extends EnumConverter<TestShardingStrategy> {
    public ShardingStrategyConverter() {
      super(TestShardingStrategy.class, "test sharding strategy");
    }
  }
}
