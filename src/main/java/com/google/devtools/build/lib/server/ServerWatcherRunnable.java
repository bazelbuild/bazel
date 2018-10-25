// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.server;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Verify;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.unix.ProcMeminfoParser;
import com.google.devtools.build.lib.unix.ProcMeminfoParser.KeywordNotFoundException;
import com.google.devtools.build.lib.util.OS;
import io.grpc.Server;
import java.io.IOException;
import java.time.Duration;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Runnable that checks to see if a {@link Server} server has been idle for too long and shuts down
 * the server if so.
 *
 * <p>TODO(bazel-team): Implement the memory checking aspect.
 */
class ServerWatcherRunnable implements Runnable {
  private static final Logger logger = Logger.getLogger(ServerWatcherRunnable.class.getName());
  private static final Duration IDLE_MEMORY_CHECK_INTERVAL = Duration.ofSeconds(5);
  private static final Duration TIME_IDLE_BEFORE_MEMORY_CHECK = Duration.ofMinutes(5);
  private static final long FREE_MEMORY_KB_ABSOLUTE_THRESHOLD = 1L << 20;
  private static final double FREE_MEMORY_PERCENTAGE_THRESHOLD = 0.05;

  private final Server server;
  private final long maxIdleSeconds;
  private final CommandManager commandManager;
  private final ProcMeminfoParserSupplier procMeminfoParserSupplier;
  private final boolean shutdownOnLowSysMem;

  ServerWatcherRunnable(
      Server server,
      long maxIdleSeconds,
      boolean shutdownOnLowSysMem,
      CommandManager commandManager) {
    this(server, maxIdleSeconds, shutdownOnLowSysMem, commandManager, ProcMeminfoParser::new);
  }

  @VisibleForTesting
  ServerWatcherRunnable(
      Server server,
      long maxIdleSeconds,
      boolean shutdownOnLowSysMem,
      CommandManager commandManager,
      ProcMeminfoParserSupplier procMeminfoParserSupplier) {
    Preconditions.checkArgument(
        maxIdleSeconds > 0,
        "Expected to only check idleness when --max_idle_secs > 0 but it was %s",
        maxIdleSeconds);
    this.server = server;
    this.maxIdleSeconds = maxIdleSeconds;
    this.commandManager = commandManager;
    this.procMeminfoParserSupplier = procMeminfoParserSupplier;
    this.shutdownOnLowSysMem = shutdownOnLowSysMem;
  }

  @Override
  public void run() {
    boolean idle = commandManager.isEmpty();
    boolean wasIdle = false;
    long shutdownTimeNanos = -1;
    long lastIdleTimeNanos = -1;

    while (true) {
      if (!wasIdle && idle) {
        shutdownTimeNanos = BlazeClock.nanoTime() + Duration.ofSeconds(maxIdleSeconds).toNanos();
        lastIdleTimeNanos = BlazeClock.nanoTime();
      }

      try {
        if (idle) {
          Verify.verify(shutdownTimeNanos > 0);
          if (shutdownOnLowSysMem && exitOnLowMemoryCheck(lastIdleTimeNanos)) {
            logger.log(Level.SEVERE, "Available RAM is low. Shutting down idle server...");
            break;
          }
          // Re-run the check every 5 seconds if no other commands have been sent to the server.
          commandManager.waitForChange(IDLE_MEMORY_CHECK_INTERVAL.toMillis());
        } else {
          commandManager.waitForChange();
        }
      } catch (InterruptedException e) {
        // Dealt with by checking the current time below.
      }

      wasIdle = idle;
      idle = commandManager.isEmpty();
      if (wasIdle && idle && BlazeClock.nanoTime() >= shutdownTimeNanos) {
        logger.info("About to shutdown due to idleness");
        break;
      }
    }
    server.shutdown();
  }

  private boolean exitOnLowMemoryCheck(long idleTimeNanos) {
    // Only run memory check on linux.
    if (OS.getCurrent() != OS.LINUX) {
      // TODO(bazel-team): Consider making this work on all operating systems.
      return false;
    }

    // Only run memory check if the server has been idle for longer than
    // TIME_IDLE_BEFORE_MEMORY_CHECK.
    if (BlazeClock.nanoTime() - idleTimeNanos < TIME_IDLE_BEFORE_MEMORY_CHECK.toNanos()) {
      return false;
    }

    try {
      ProcMeminfoParser meminfoParser = procMeminfoParserSupplier.get();
      long freeRamKb = meminfoParser.getFreeRamKb();
      long usedRamKb = meminfoParser.getTotalKb();
      double fractionRamFree = ((double) freeRamKb) / usedRamKb;

      // Shutdown when both the absolute amount and percentage of free RAM is lower than the set
      // thresholds.
      return fractionRamFree < FREE_MEMORY_PERCENTAGE_THRESHOLD
          && freeRamKb < FREE_MEMORY_KB_ABSOLUTE_THRESHOLD;
    } catch (IOException | KeywordNotFoundException e) {
      logger.log(Level.WARNING, "Unable to read memory info.", e);
      return false;
    }
  }

  /** Supplier for a {@link ProcMeminfoParser}. */
  interface ProcMeminfoParserSupplier {
    ProcMeminfoParser get() throws IOException;
  }
}
