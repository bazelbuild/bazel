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

import com.google.common.base.Ascii;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters.IntegerConverter;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.OptionalInt;

/** A strategy for running the same tests in many processes. */
interface TestShardingStrategy {
  int getNumberOfShards(int shardCountFromAttr);

  /** Returns the single shard index to run, if this strategy selects only one shard. */
  default OptionalInt getOnlyShardIndex() {
    return OptionalInt.empty();
  }

  /** Converts to {@link TestShardingStrategy}. */
  final class ShardingStrategyConverter extends Converter.Contextless<TestShardingStrategy> {
    private static final String FORCED_PREFIX = "forced=";
    private static final String ONLY_PREFIX = "only=";

    @Override
    public String getTypeDescription() {
      return "explicit, disabled, forced=k where k is the number of shards to enforce, or only=k"
          + " where k is the 1-based shard index to run";
    }

    @Override
    public TestShardingStrategy convert(String input) throws OptionsParsingException {
      for (TestShardingStrategy value : TestShardingStrategyNotForced.values()) {
        if (Ascii.equalsIgnoreCase(value.toString(), input)) {
          return value;
        }
      }

      if (Ascii.toLowerCase(input).startsWith(FORCED_PREFIX)) {
        int forcedShardsCount =
            new IntegerConverter().convert(input.substring(FORCED_PREFIX.length()));
        if (forcedShardsCount < 0) {
          throw new OptionsParsingException("Forced shards count cannot be negative.");
        }

        return new TestShardingStrategyForced(forcedShardsCount);
      }

      if (Ascii.toLowerCase(input).startsWith(ONLY_PREFIX)) {
        int onlyShardIndex =
            new IntegerConverter().convert(input.substring(ONLY_PREFIX.length()));
        if (onlyShardIndex < 1) {
          throw new OptionsParsingException("Shard index for only= must be at least 1.");
        }

        // Convert from 1-based (user-facing) to 0-based (internal).
        return new TestShardingStrategyOnly(onlyShardIndex - 1);
      }

      throw new OptionsParsingException(
          "Not a valid test sharding strategy: '"
              + input
              + "' (should be "
              + getTypeDescription()
              + ")");
    }
  }
}
