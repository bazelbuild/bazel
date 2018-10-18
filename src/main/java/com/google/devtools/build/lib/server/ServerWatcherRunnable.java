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

import com.google.common.base.Preconditions;
import com.google.common.base.Verify;
import com.google.devtools.build.lib.clock.BlazeClock;
import io.grpc.Server;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

/**
 * Runnable that checks to see if a {@link Server} server has been idle for too long and shuts down
 * the server if so.
 *
 * <p>TODO(bazel-team): Implement the memory checking aspect.
 */
class ServerWatcherRunnable implements Runnable {
  private static final long NANOSECONDS_IN_MS = TimeUnit.MILLISECONDS.toNanos(1);
  private static final Logger logger = Logger.getLogger(ServerWatcherRunnable.class.getName());
  private final Server server;
  private final long maxIdleSeconds;
  private final CommandManager commandManager;

  ServerWatcherRunnable(Server server, long maxIdleSeconds, CommandManager commandManager) {
    Preconditions.checkArgument(
        maxIdleSeconds > 0,
        "Expected to only check idleness when --max_idle_secs > 0 but it was %s",
        maxIdleSeconds);
    this.server = server;
    this.maxIdleSeconds = maxIdleSeconds;
    this.commandManager = commandManager;
  }

  @Override
  public void run() {
    boolean idle = commandManager.isEmpty();
    boolean wasIdle = false;
    long shutdownTime = -1;

    while (true) {
      if (!wasIdle && idle) {
        shutdownTime = BlazeClock.nanoTime() + maxIdleSeconds * 1000L * NANOSECONDS_IN_MS;
      }

      try {
        if (idle) {
          Verify.verify(shutdownTime > 0);
          long waitTime = shutdownTime - BlazeClock.nanoTime();
          if (waitTime > 0) {
            // Round upwards so that we don't busy-wait in the last millisecond
            commandManager.waitForChange((waitTime + NANOSECONDS_IN_MS - 1) / NANOSECONDS_IN_MS);
          }
        } else {
          commandManager.waitForChange();
        }
      } catch (InterruptedException e) {
        // Dealt with by checking the current time below.
      }

      wasIdle = idle;
      idle = commandManager.isEmpty();
      if (wasIdle && idle && BlazeClock.nanoTime() >= shutdownTime) {
        break;
      }
    }

    logger.info("About to shutdown due to idleness");
    server.shutdown();
  }
}
