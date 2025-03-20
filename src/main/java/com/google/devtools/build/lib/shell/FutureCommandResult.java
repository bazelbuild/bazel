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
package com.google.devtools.build.lib.shell;

import com.google.devtools.build.lib.shell.Consumers.OutErrConsumers;
import java.io.IOException;

/**
 * Supplier of the command result which additionally allows to check if the command already
 * terminated. Implementations of this interface may not be thread-safe.
 */
public final class FutureCommandResult {
  private final Command command;
  private final Subprocess process;
  private final OutErrConsumers outErrConsumers;
  private final boolean killSubprocessOnInterrupt;

  FutureCommandResult(
      Command command,
      Subprocess process,
      OutErrConsumers outErrConsumers,
      boolean killSubprocessOnInterrupt) {
    this.command = command;
    this.process = process;
    this.outErrConsumers = outErrConsumers;
    this.killSubprocessOnInterrupt = killSubprocessOnInterrupt;
  }

  /**
   * Returns the result of command execution. If the process is not finished yet (as reported by
   * {@link #isDone()}, the call will block until that process terminates.
   *
   * @return non-null result of command execution
   * @throws AbnormalTerminationException if command execution failed
   * @throws InterruptedException if {@link #killSubprocessOnInterrupt} is true and thread is
   *     interrupted before subprocess completes.
   */
  public CommandResult get() throws AbnormalTerminationException, InterruptedException {
    TerminationStatus status;
    try {
      status = waitForProcess(process, killSubprocessOnInterrupt);
    } catch (InterruptedException e) {
      outErrConsumers.cancel();
      process.close();
      throw e;
    }
    try {
      if (Thread.currentThread().isInterrupted()) {
        // Can be interrupted if killSubprocessOnInterrupt is false, or interrupt raced with us.
        outErrConsumers.cancel();
      } else {
        outErrConsumers.waitForCompletion();
      }
    } catch (IOException ioe) {
      CommandResult noOutputResult =
          CommandResult.builder()
              .setStdoutStream(CommandResult.EMPTY_OUTPUT)
              .setStderrStream(CommandResult.EMPTY_OUTPUT)
              .setTerminationStatus(status)
              .build();
      if (status.success()) {
        // If command was otherwise successful, throw an exception about this
        throw new AbnormalTerminationException(command, noOutputResult, ioe);
      } else {
        // Otherwise, throw the more important exception -- command
        // was not successful
        String message = status + "; also encountered an error while attempting to retrieve output";
        throw status.exited()
            ? new BadExitStatusException(command, noOutputResult, message, ioe)
            : new AbnormalTerminationException(command, noOutputResult, message, ioe);
      }
    } finally {
      process.close();
    }

    CommandResult commandResult =
        CommandResult.builder()
            .setStdoutStream(outErrConsumers.getAccumulatedOut())
            .setStderrStream(outErrConsumers.getAccumulatedErr())
            .setTerminationStatus(status)
            .build();
    commandResult.logThis();
    if (status.success()) {
      return commandResult;
    } else if (status.exited()) {
      throw new BadExitStatusException(command, commandResult, status.toString());
    } else {
      throw new AbnormalTerminationException(command, commandResult, status.toString());
    }
  }

  private static TerminationStatus waitForProcess(
      Subprocess process, boolean killSubprocessOnInterrupt) throws InterruptedException {
    boolean wasInterrupted = false;
    try {
      while (true) {
        try {
          process.waitFor();
          break;
        } catch (InterruptedException ie) {
          wasInterrupted = true;
          if (killSubprocessOnInterrupt) {
            process.destroy();
          }
        }
      }
      if (wasInterrupted && killSubprocessOnInterrupt) {
        // Don't need to do any clean-up in finally block below.
        wasInterrupted = false;
        throw new InterruptedException();
      }
      return new TerminationStatus(process.exitValue(), process.timedout());

    } finally {
      // Read this for detailed explanation: http://www.ibm.com/developerworks/library/j-jtp05236/
      if (wasInterrupted) {
        Thread.currentThread().interrupt(); // preserve interrupted status
      }
    }
  }

  public boolean isDone() {
    return process.finished();
  }
}
