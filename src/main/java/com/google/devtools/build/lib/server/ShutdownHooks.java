// Copyright 2016 The Bazel Authors. All rights reserved.
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
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.concurrent.GuardedBy;

/** Wrapper for common shutdown operations. */
public class ShutdownHooks {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  public static ShutdownHooks createAndRegister() {
    ShutdownHooks result = new ShutdownHooks();
    Runtime.getRuntime().addShutdownHook(new Thread(result::runHooks));
    return result;
  }

  @VisibleForTesting
  static ShutdownHooks createUnregistered() {
    return new ShutdownHooks();
  }

  private final AtomicBoolean enabled = new AtomicBoolean(true);

  @GuardedBy("this")
  private final List<Path> filesToDeleteAtExit = new ArrayList<>();

  @GuardedBy("this")
  private Runnable pidFileCleanup = null;

  private ShutdownHooks() {}

  /** Schedules the specified file for (attempted) deletion at JVM exit. */
  public synchronized void deleteAtExit(Path path) {
    filesToDeleteAtExit.add(path);
  }

  /**
   * Registers a cleanup operation for the given PID file.
   *
   * <p>The PID file watcher will be stopped before the PID file is deleted.
   *
   * <p>This operation will be executed when the JVM shuts down.
   */
  public synchronized void cleanupPidFile(Path pidFile, PidFileWatcher pidFileWatcher) {
    pidFileCleanup =
        () -> {
          try {
            pidFileWatcher.endWatch();
            Uninterruptibles.joinUninterruptibly(pidFileWatcher);
            pidFile.delete();
          } catch (IOException e) {
            printStack(e);
          }
        };
  }

  /** Disables shutdown hook execution. */
  void disable() {
    enabled.set(false);
  }

  @VisibleForTesting
  void runHooks() {
    if (!enabled.get()) {
      return;
    }

    List<Path> files;
    synchronized (this) {
      if (pidFileCleanup != null) {
        pidFileCleanup.run();
      }
      files = new ArrayList<>(filesToDeleteAtExit);
    }
    for (Path path : files) {
      try {
        path.delete();
      } catch (IOException e) {
        printStack(e);
      }
    }
  }

  private static void printStack(IOException e) {
    // Hopefully this never happens. It's not very nice to just write this to the user's console,
    // but I'm not sure what better choice we have.
    StringWriter err = new StringWriter();
    PrintWriter printErr = new PrintWriter(err);
    printErr.println("=======[BAZEL SERVER: ENCOUNTERED IO EXCEPTION]=======");
    e.printStackTrace(printErr);
    printErr.println("=====================================================");
    logger.atSevere().log("%s", err);
  }
}
