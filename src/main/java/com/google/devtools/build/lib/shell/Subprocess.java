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

package com.google.devtools.build.lib.shell;

import java.io.Closeable;
import java.io.InputStream;
import java.io.OutputStream;
import javax.annotation.Nullable;

/** A process started by Bazel. */
public interface Subprocess extends Closeable {

  /** Kill the process. */
  boolean destroy();

  /**
   * Returns the exit value of the process.
   *
   * <p>Throws {@code IllegalThreadStateException} if the process has not terminated yet.
   */
  int exitValue();

  /**
   * Returns the if the process has finished.
   *
   * <p>This may cause the process to be destroyed as a side effect, for example due to a timeout.
   */
  boolean finished();

  /** Returns true if the process is still alive. Does not block or cause any side effects. */
  boolean isAlive();

  /** Returns if the process timed out. */
  boolean timedout();

  /** Waits for the process to finish. */
  void waitFor() throws InterruptedException;

  /**
   * Returns a stream into which the stdin of the process can be written, or null if the stdin was
   * redirected from a file.
   */
  @Nullable
  OutputStream getOutputStream();

  /**
   * Returns a stream from which the stdout of the process can be read, or null if the stdout was
   * redirected to a file.
   */
  @Nullable
  InputStream getInputStream();

  /**
   * Returns a stream from which the stderr of the process can be read, or null if the stderr was
   * redirected to a file.
   */
  @Nullable
  InputStream getErrorStream();

  /** Returns the PID of the current process. */
  long getProcessId();

  /*
   * Terminates the process as thoroughly as the underlying implementation allows and releases
   * native data structures associated with the process.
   */
  @Override
  void close();

  /** Waits for the process to finish in a non-interruptible manner. */
  default void waitForUninterruptibly() {
    boolean wasInterrupted = false;
    try {
      while (true) {
        try {
          waitFor();
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
   * Kills the subprocess and awaits for its termination so that we know it has released any
   * resources it may have held.
   */
  default void destroyAndWait() {
    destroy();
    waitForUninterruptibly();
  }
}
