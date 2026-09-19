// Copyright 2025 The Bazel Authors. All rights reserved.
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

import static java.lang.Math.max;

import java.util.OptionalInt;

final class TestShardingStrategyOnly implements TestShardingStrategy {
  private final int onlyShardIndex;

  TestShardingStrategyOnly(int onlyShardIndex) {
    this.onlyShardIndex = onlyShardIndex;
  }

  @Override
  public int getNumberOfShards(int shardCountFromAttr) {
    return max(shardCountFromAttr, 0);
  }

  @Override
  public OptionalInt getOnlyShardIndex() {
    return OptionalInt.of(onlyShardIndex);
  }

  @Override
  public String toString() {
    return "only=" + (onlyShardIndex + 1);
  }
}
