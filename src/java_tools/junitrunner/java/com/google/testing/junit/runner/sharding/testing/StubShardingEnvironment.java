// Copyright 2012 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.sharding.testing;

import com.google.testing.junit.runner.sharding.ShardingEnvironment;

/**
 * Stub sharding environment.
 */
public class StubShardingEnvironment extends ShardingEnvironment {
  private boolean shardingEnabled = false;

  @Override
  public boolean isShardingEnabled() {
    return shardingEnabled;
  }
  
  public void setIsShardingEnabled(boolean shardingEnabled) {
    this.shardingEnabled = shardingEnabled;
  }

  @Override
  public void touchShardFile() {
    // Noop.
  }

  @Override
  public int getShardIndex() {
    throw new UnsupportedOperationException();
  }

  @Override
  public int getTotalShards() {
    throw new UnsupportedOperationException();
  }

  @Override
  public String getTestShardingStrategy() {
    throw new UnsupportedOperationException();
  }
}
