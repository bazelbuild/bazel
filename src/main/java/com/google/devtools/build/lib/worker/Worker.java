// Copyright 2015 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.worker;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.vfs.Path;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.ProcessBuilder.Redirect;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Interface to a worker process running as a child process.
 *
 * <p>A worker process must follow this protocol to be usable via this class: The worker process is
 * spawned on demand. The worker process is free to exit whenever necessary, as new instances will
 * be relaunched automatically. Communication happens via the WorkerProtocol protobuf, sent to and
 * received from the worker process via stdin / stdout.
 *
 * <p>Other code in Blaze can talk to the worker process via input / output streams provided by this
 * class.
 */
final class Worker {
  private static final AtomicInteger pidCounter = new AtomicInteger();
  private final int workerId;
  private final Process process;
  private final Thread shutdownHook;

  private Worker(Process process, Thread shutdownHook, int pid) {
    this.process = process;
    this.shutdownHook = shutdownHook;
    this.workerId = pid;
  }

  static Worker create(WorkerKey key, Path logDir, Reporter reporter, boolean verbose)
      throws IOException {
    Preconditions.checkNotNull(key);
    Preconditions.checkNotNull(logDir);

    int workerId = pidCounter.getAndIncrement();
    Path logFile = logDir.getRelative("worker-" + workerId + "-" + key.getMnemonic() + ".log");

    ProcessBuilder processBuilder =
        new ProcessBuilder(key.getArgs().toArray(new String[0]))
            .directory(key.getWorkDir().getPathFile())
            .redirectError(Redirect.appendTo(logFile.getPathFile()));
    processBuilder.environment().putAll(key.getEnv());

    final Process process = processBuilder.start();

    Thread shutdownHook =
        new Thread() {
          @Override
          public void run() {
            destroyProcess(process);
          }
        };
    Runtime.getRuntime().addShutdownHook(shutdownHook);

    if (verbose) {
      reporter.handle(
          Event.info(
              "Created new "
                  + key.getMnemonic()
                  + " worker (id "
                  + workerId
                  + "), logging to "
                  + logFile));
    }

    return new Worker(process, shutdownHook, workerId);
  }

  void destroy() {
    Runtime.getRuntime().removeShutdownHook(shutdownHook);
    destroyProcess(process);
  }

  /**
   * Destroys a process and waits for it to exit. This is necessary for the child to not become a
   * zombie.
   *
   * @param process the process to destroy.
   */
  private static void destroyProcess(Process process) {
    boolean wasInterrupted = false;
    try {
      process.destroy();
      while (true) {
        try {
          process.waitFor();
          return;
        } catch (InterruptedException ie) {
          wasInterrupted = true;
        }
      }
    } finally {
      // Read this for detailed explanation: http://www.ibm.com/developerworks/library/j-jtp05236/
      if (wasInterrupted) {
        Thread.currentThread().interrupt(); // preserve interrupted status
      }
    }
  }

  /**
   * Returns a unique id for this worker. This is used to distinguish different worker processes in
   * logs and messages.
   */
  int getWorkerId() {
    return this.workerId;
  }

  boolean isAlive() {
    // This is horrible, but Process.isAlive() is only available from Java 8 on and this is the
    // best we can do prior to that.
    try {
      process.exitValue();
      return false;
    } catch (IllegalThreadStateException e) {
      return true;
    }
  }

  InputStream getInputStream() {
    return process.getInputStream();
  }

  OutputStream getOutputStream() {
    return process.getOutputStream();
  }
}
