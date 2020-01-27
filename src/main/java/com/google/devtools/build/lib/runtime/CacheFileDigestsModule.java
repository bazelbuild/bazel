// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime;

import com.google.common.base.Preconditions;
import com.google.common.cache.CacheStats;
import com.google.devtools.build.lib.actions.cache.DigestUtils;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import java.util.logging.Logger;

/** Enables the caching of file digests in {@link DigestUtils}. */
public class CacheFileDigestsModule extends BlazeModule {

  private static final Logger logger = Logger.getLogger(CacheFileDigestsModule.class.getName());

  /** Stats gathered at the beginning of a command, to compute deltas on completion. */
  private CacheStats stats;

  /**
   * Last known size of the cache. Changes to this value cause the cache to be reinitialized. null
   * if we don't know anything about the last value yet (i.e. before any command has been run).
   */
  private Long lastKnownCacheSize;

  public CacheFileDigestsModule() {}

  /**
   * Adds a line to the log with cache statistics.
   *
   * @param message message to prefix to the written line
   * @param stats the cache statistics to be logged
   */
  private static void logStats(String message, CacheStats stats) {
    logger.info(
        message
            + ": hit count="
            + stats.hitCount()
            + ", miss count="
            + stats.missCount()
            + ", hit rate="
            + stats.hitRate()
            + ", eviction count="
            + stats.evictionCount());
  }

  @Override
  public void executorInit(CommandEnvironment env, BuildRequest request, ExecutorBuilder builder) {
    ExecutionOptions options = request.getOptions(ExecutionOptions.class);
    if (lastKnownCacheSize == null
        || options.cacheSizeForComputedFileDigests != lastKnownCacheSize) {
      logger.info("Reconfiguring cache with size=" + options.cacheSizeForComputedFileDigests);
      DigestUtils.configureCache(options.cacheSizeForComputedFileDigests);
      lastKnownCacheSize = options.cacheSizeForComputedFileDigests;
    }

    if (options.cacheSizeForComputedFileDigests == 0) {
      stats = null;
      logger.info("Disabled cache");
    } else {
      stats = DigestUtils.getCacheStats();
      logStats("Accumulated cache stats before command", stats);
    }
  }

  @Override
  public void commandComplete() {
    if (stats != null) {
      CacheStats newStats = DigestUtils.getCacheStats();
      Preconditions.checkNotNull(newStats, "The cache is enabled so we must get some stats back");
      logStats("Accumulated cache stats after command", newStats);
      logStats("Cache stats for finished command", newStats.minus(stats));
      stats = null; // Silence stats until next command that uses the executor.
    }
  }
}
