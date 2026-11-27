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

import com.github.benmanes.caffeine.cache.stats.CacheStats;
import com.google.common.base.Preconditions;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.vfs.DigestUtils;
import com.google.devtools.common.options.Options;

/** Enables the caching of file digests in {@link DigestUtils}. */
public class CacheFileDigestsModule extends BlazeModule {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /** Stats gathered at the beginning of a command, to compute deltas on completion. */
  private CacheStats stats;

  /**
   * Last known size of the cache. Changes to this value cause the cache to be reinitialized. null
   * if we don't know anything about the last value yet (i.e. before any command has been run).
   */
  private Integer lastKnownCacheSize;

  public CacheFileDigestsModule() {}

  /**
   * Adds a line to the log with cache statistics.
   *
   * @param message message to prefix to the written line
   * @param stats the cache statistics to be logged
   */
  private static void logStats(String message, CacheStats stats) {
    logger.atInfo().log(
        "%s: hit count=%d, miss count=%d, hit rate=%g, eviction count=%d",
        message, stats.hitCount(), stats.missCount(), stats.hitRate(), stats.evictionCount());
  }

  @Override
  public void beforeCommand(CommandEnvironment env) {
    // Every command that loads computes the digests of files watched by repository rules and
    // module extensions, even if it doesn't execute anything. Resizing discards the cache, so only
    // such commands resize it and only if they have the flag, which e.g. query doesn't. Other
    // commands keep the current size.
    ExecutionOptions options = env.getOptions().getOptions(ExecutionOptions.class);
    int cacheSize =
        (options != null ? options : Options.getDefaults(ExecutionOptions.class))
            .getCacheSizeForComputedFileDigests();
    if (lastKnownCacheSize == null
        || (options != null
            && env.getCommand().buildPhase().loads()
            && cacheSize != lastKnownCacheSize)) {
      configureCache(cacheSize);
    }
    startStats();
  }

  private void configureCache(int cacheSize) {
    logger.atInfo().log("Reconfiguring cache with size=%d", cacheSize);
    DigestUtils.configureCache(cacheSize);
    lastKnownCacheSize = cacheSize;
  }

  private void startStats() {
    if (lastKnownCacheSize == 0) {
      stats = null;
      logger.atInfo().log("Disabled cache");
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
      stats = null; // Silence stats until the next command.
    }
  }
}
