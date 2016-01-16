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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.io.Files;

import java.io.File;
import java.io.IOException;

/**
 * Utility class that encapsulates dependencies from sharding implementations
 * on the test environment.  See http://bazel.io/docs/test-sharding.html for a
 * list of all environment variables related to test sharding.
 */
public class ShardingEnvironment {

  /**
   * A singleton instance of ShardingEnvironment declared for convenience.
   */
  public static final ShardingEnvironment DEFAULT = new ShardingEnvironment();

  /** Usage: -Dtest.sharding.strategy=round_robin */
  private static final String TEST_SHARDING_STRATEGY = "test.sharding.strategy";

  /**
   * Return true iff the current test should be sharded.
   */
  public boolean isShardingEnabled() {
    return System.getenv("TEST_TOTAL_SHARDS") != null;
  }

  /**
   * Returns the 0-indexed test shard number, where
   * 0 <= shard index < total shards.
   * If the environment does not specify a test shard number, returns 0.
   */
  public int getShardIndex() {
    String shardIndex = System.getenv("TEST_SHARD_INDEX");
    return shardIndex == null ? 0 : Integer.parseInt(shardIndex);
  }

  /**
   * Returns the total number of test shards, or 1 if not specified by the
   * test environment.
   */
  public int getTotalShards() {
    String totalShards = System.getenv("TEST_TOTAL_SHARDS");
    return totalShards == null ? 1 : Integer.parseInt(totalShards);
  }

  /**
   * Creates the shard file that is used to indicate that tests are
   * being sharded.
   */
  public void touchShardFile() {
    String shardStatusPath = System.getenv("TEST_SHARD_STATUS_FILE");
    File shardFile = (shardStatusPath == null ? null : new File(shardStatusPath));
    touchShardFile(shardFile);
  }

  @VisibleForTesting
  static void touchShardFile(File shardFile) {
    if (shardFile != null) {
      try {
        Files.touch(shardFile);
      } catch (IOException e) {
        throw new RuntimeException("Error writing shard file " + shardFile, e);
      }
    }
  }

  /**
   * Returns the test sharding strategy optionally specified by the JVM flag
   * {@link #TEST_SHARDING_STRATEGY}, which maps to the enums in
   * {@link com.google.testing.junit.runner.sharding.ShardingFilters.ShardingStrategy}.
   */
  public String getTestShardingStrategy() {
    return System.getProperty(TEST_SHARDING_STRATEGY);
  }
}
