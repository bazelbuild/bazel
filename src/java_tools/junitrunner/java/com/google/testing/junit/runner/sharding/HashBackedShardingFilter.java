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

import com.google.common.base.Preconditions;

import org.junit.runner.Description;
import org.junit.runner.manipulation.Filter;

/**
 * Sharding filter that uses the hashcode of the test description to
 * assign it to a shard.
 */
class HashBackedShardingFilter extends Filter {

  private final int shardIndex;
  private final int totalShards;

  public HashBackedShardingFilter(int shardIndex, int totalShards) {
    Preconditions.checkArgument(shardIndex >= 0);
    Preconditions.checkArgument(totalShards > shardIndex);
    this.shardIndex = shardIndex;
    this.totalShards = totalShards;
  }

  @Override
  public boolean shouldRun(Description description) {
    if (description.isSuite()) {
      return true;
    }
    int mod = description.getDisplayName().hashCode() % totalShards;
    if (mod < 0) {
      mod += totalShards;
    }
    Preconditions.checkState(mod >= 0 && mod < totalShards);

    return mod == shardIndex;
  }

  @Override
  public String describe() {
    return "hash-backed sharding filter";
  }

}
