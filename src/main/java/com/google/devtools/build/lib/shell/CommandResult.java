// Copyright 2014 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;

import java.io.ByteArrayOutputStream;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Encapsulates the results of a command execution, including exit status
 * and output to stdout and stderr.
 */
public final class CommandResult {

  private static final Logger logger =
      Logger.getLogger("com.google.devtools.build.lib.shell.Command");

  private static final byte[] NO_BYTES = new byte[0];

  static final ByteArrayOutputStream EMPTY_OUTPUT =
    new ByteArrayOutputStream() {

      @Override
      public byte[] toByteArray() {
        return NO_BYTES;
      }
  };

  static final ByteArrayOutputStream NO_OUTPUT_COLLECTED =
    new ByteArrayOutputStream(){

      @Override
      public byte[] toByteArray() {
        throw new IllegalStateException("Output was not collected");
      }
  };

  private final ByteArrayOutputStream stdout;
  private final ByteArrayOutputStream stderr;
  private final TerminationStatus terminationStatus;

  CommandResult(final ByteArrayOutputStream stdout,
                final ByteArrayOutputStream stderr,
                final TerminationStatus terminationStatus) {
    checkNotNull(stdout);
    checkNotNull(stderr);
    checkNotNull(terminationStatus);
    this.stdout = stdout;
    this.stderr = stderr;
    this.terminationStatus = terminationStatus;
  }

  /**
   * @return raw bytes that were written to stdout by the command, or
   *  null if caller did chose to ignore output
   * @throws IllegalStateException if output was not collected
   */
  public byte[] getStdout() {
    return stdout.toByteArray();
  }

  /**
   * @return raw bytes that were written to stderr by the command, or
   *  null if caller did chose to ignore output
   * @throws IllegalStateException if output was not collected
   */
  public byte[] getStderr() {
    return stderr.toByteArray();
  }

  /**
   * @return the termination status of the subprocess.
   */
  public TerminationStatus getTerminationStatus() {
    return terminationStatus;
  }

  void logThis() {
    if (!logger.isLoggable(Level.FINER)) {
      return;
    }
    logger.finer(terminationStatus.toString());

    if (stdout == NO_OUTPUT_COLLECTED) {
      return;
    }
    logger.finer("Stdout: " + LogUtil.toTruncatedString(stdout.toByteArray()));
    logger.finer("Stderr: " + LogUtil.toTruncatedString(stderr.toByteArray()));
  }

}
