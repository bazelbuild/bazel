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

package com.google.devtools.build.lib.server;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/** A class that watches if the PID file changes and shuts down the server immediately if so. */
public class PidFileWatcher {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final Path pidFile;
  private final int serverPid;
  private final Runnable onPidFileChange;
  private final AtomicBoolean shuttingDown;

  private ScheduledFuture<?> iteration;

  private final ScheduledExecutorService executorService;

  public PidFileWatcher(Path pidFile, int serverPid) {
    this(
        pidFile,
        serverPid,
        () -> Runtime.getRuntime().halt(ExitCode.BLAZE_INTERNAL_ERROR.getNumericExitCode()));
  }

  /** Test-only constructor allowing a more gentle {@link #onPidFileChange} callback. */
  @VisibleForTesting
  PidFileWatcher(Path pidFile, int serverPid, Runnable onPidFileChange) {
    this.pidFile = pidFile;
    this.serverPid = serverPid;
    this.onPidFileChange = onPidFileChange;
    this.shuttingDown = new AtomicBoolean(false);
    this.executorService =
        Executors.newSingleThreadScheduledExecutor(
            new ThreadFactoryBuilder()
                .setNameFormat("pid-file-watcher-%d")
                .setDaemon(true)
                .build());
  }

  /** Starts the watching. Can only be called once per instance. */
  public void start() {
    Preconditions.checkState(iteration == null); // Can only be called once
    iteration = executorService.scheduleWithFixedDelay(this::iteration, 3, 3, TimeUnit.SECONDS);
  }

  /**
   * Signals a shutdown is in progress, thus PID file changes can be expected and should not trigger
   * an aggressive shutdown.
   */
  void signalShutdown() {
    shuttingDown.set(true);
  }

  /**
   * Stops the watching. Once this method returns, no more activity will happen.
   *
   * <p>This should invoked before the PID file is deleted by the Blaze server on an orderly
   * shutdown.
   */
  void endWatch() {
    executorService.shutdownNow();
    Uninterruptibles.awaitTerminationUninterruptibly(executorService);
  }

  private void iteration() {
    if (runPidFileChecks()) {
      return;
    }

    iteration.cancel(false);
  }

  /**
   * Runs one iteration of PID file checks. Returns whether or not the main loop should continue, or
   * crashes the program via {@link #onPidFileChange} callback.
   */
  @VisibleForTesting
  boolean runPidFileChecks() {
    if (pidFileValid(pidFile, serverPid)) {
      return true;
    }

    if (shuttingDown.get()) {
      logger.atWarning().log("PID file deleted or overwritten but shutdown is already in progress");
      return false;
    }

    // Someone overwrote the PID file. Maybe it's another server, so shut down as quickly
    // as possible without even running the shutdown hooks (that would delete it)
    logger.atSevere().log("PID file deleted or overwritten, exiting as quickly as possible");
    onPidFileChange.run();
    throw new IllegalStateException("Should have crashed.");
  }

  private static boolean pidFileValid(Path pidFile, int expectedPid) {
    String pidFileContents;
    try {
      pidFileContents = new String(FileSystemUtils.readContentAsLatin1(pidFile));
    } catch (IOException e) {
      logger.atInfo().log("Cannot read PID file: %s", e.getMessage());
      return false;
    }

    try {
      return Integer.parseInt(pidFileContents) == expectedPid;
    } catch (NumberFormatException e) {
      logger.atWarning().withCause(e).log("Pid was invalid: %s", pidFileContents);
      return false;
    }
  }
}
