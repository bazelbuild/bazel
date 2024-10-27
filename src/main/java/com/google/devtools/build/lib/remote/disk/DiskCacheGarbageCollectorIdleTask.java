// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.disk;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.remote.disk.DiskCacheGarbageCollector.CollectionPolicy;
import com.google.devtools.build.lib.remote.disk.DiskCacheGarbageCollector.CollectionStats;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.server.IdleTask;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.time.Duration;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import javax.annotation.Nullable;

/** An {@link IdleTask} to run a {@link DiskCacheGarbageCollector}. */
public final class DiskCacheGarbageCollectorIdleTask implements IdleTask {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final Duration delay;
  private final DiskCacheGarbageCollector gc;

  private static final ExecutorService executorService =
      Executors.newCachedThreadPool(
          new ThreadFactoryBuilder().setNameFormat("disk-cache-gc-%d").build());

  private DiskCacheGarbageCollectorIdleTask(Duration delay, DiskCacheGarbageCollector gc) {
    this.delay = delay;
    this.gc = gc;
  }

  /**
   * Creates a new {@link DiskCacheGarbageCollectorIdleTask} according to the options.
   *
   * @param remoteOptions the remote options
   * @param workingDirectory the working directory
   * @param executorService the executor service to schedule I/O operations onto
   * @return the idle task, or null if garbage collection is disabled
   */
  @Nullable
  public static DiskCacheGarbageCollectorIdleTask create(
      RemoteOptions remoteOptions, Path workingDirectory) {
    if (remoteOptions.diskCache == null || remoteOptions.diskCache.isEmpty()) {
      return null;
    }
    Optional<Long> maxSizeBytes = Optional.empty();
    if (remoteOptions.diskCacheGcMaxSize > 0) {
      maxSizeBytes = Optional.of(remoteOptions.diskCacheGcMaxSize);
    }
    Optional<Duration> maxAge = Optional.empty();
    if (!remoteOptions.diskCacheGcMaxAge.isZero()) {
      maxAge = Optional.of(remoteOptions.diskCacheGcMaxAge);
    }
    Duration delay = remoteOptions.diskCacheGcIdleDelay;
    if (maxSizeBytes.isEmpty() && maxAge.isEmpty()) {
      return null;
    }
    var policy = new CollectionPolicy(maxSizeBytes, maxAge);
    var gc =
        new DiskCacheGarbageCollector(
            workingDirectory.getRelative(remoteOptions.diskCache), executorService, policy);
    return new DiskCacheGarbageCollectorIdleTask(delay, gc);
  }

  @VisibleForTesting
  public DiskCacheGarbageCollector getGarbageCollector() {
    return gc;
  }

  @Override
  public Duration delay() {
    return delay;
  }

  @Override
  public void run() {
    try {
      logger.atInfo().log("Disk cache garbage collection started");
      CollectionStats stats = gc.run();
      logger.atInfo().log("%s", stats.displayString());
    } catch (InterruptedException e) {
      logger.atInfo().withCause(e).log("Disk cache garbage collection interrupted");
    } catch (Throwable e) {
      logger.atWarning().withCause(e).log("Disk cache garbage collection failed");
    }
  }
}
