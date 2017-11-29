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
 * Basic and only implementation of {@link FutureCommandResult} for use by implementations of
 * {@link SubprocessFactory}.
 */
final class FutureCommandResultImpl implements FutureCommandResult {
  private final Command command;
  private final Subprocess process;
  private final OutErrConsumers outErrConsumers;
  private final boolean killSubprocessOnInterrupt;

  public FutureCommandResultImpl(
      Command command,
      Subprocess process,
      OutErrConsumers outErrConsumers,
      boolean killSubprocessOnInterrupt) {
    this.command = command;
    this.process = process;
    this.outErrConsumers = outErrConsumers;
    this.killSubprocessOnInterrupt = killSubprocessOnInterrupt;
  }

  @Override
  public CommandResult get() throws AbnormalTerminationException {
    TerminationStatus status = waitForProcess(process, killSubprocessOnInterrupt);
    try {
      if (Thread.currentThread().isInterrupted()) {
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
        String message = status
          + "; also encountered an error while attempting to retrieve output";
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
      Subprocess process, boolean killSubprocessOnInterrupt) {
    boolean wasInterrupted = false;
    try {
      while (true) {
        try {
          process.waitFor();
          return new TerminationStatus(process.exitValue(), process.timedout());
        } catch (InterruptedException ie) {
          wasInterrupted = true;
          if (killSubprocessOnInterrupt) {
            process.destroy();
          }
        }
      }
    } finally {
      // Read this for detailed explanation: http://www.ibm.com/developerworks/library/j-jtp05236/
      if (wasInterrupted) {
        Thread.currentThread().interrupt(); // preserve interrupted status
      }
    }
  }

  @Override
  public void cancel() {
    process.destroy();
  }

  @Override
  public boolean isDone() {
    return process.finished();
  }
}
