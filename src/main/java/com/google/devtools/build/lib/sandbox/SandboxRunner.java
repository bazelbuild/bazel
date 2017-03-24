// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.sandbox;

import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.shell.AbnormalTerminationException;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.KillableObserver;
import com.google.devtools.build.lib.shell.TerminationStatus;
import com.google.devtools.build.lib.util.CommandFailureUtils;
import com.google.devtools.build.lib.util.io.OutErr;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/** A common interface of all sandbox runners, no matter which platform they're working on. */
abstract class SandboxRunner {

  private static final String SANDBOX_DEBUG_SUGGESTION =
      "\n\nUse --sandbox_debug to see verbose messages from the sandbox";

  private final boolean verboseFailures;

  SandboxRunner(boolean verboseFailures) {
    this.verboseFailures = verboseFailures;
  }

  /**
   * Runs the command specified via {@code arguments} and {@code env} inside the sandbox.
   *
   * @param arguments - arguments of spawn to run inside the sandbox.
   * @param environment - environment variables to pass to the spawn.
   * @param outErr - error output to capture sandbox's and command's stderr.
   * @param timeout - after how many seconds should the process be killed.
   * @param allowNetwork - whether networking should be allowed for the process.
   * @param sandboxDebug - whether debugging message should be printed.
   * @param useFakeHostname - whether the hostname should be set to 'localhost' inside the sandbox.
   * @param useFakeUsername - whether the username should be set to 'nobody' inside the sandbox.
   */
  void run(
      List<String> arguments,
      Map<String, String> environment,
      OutErr outErr,
      int timeout,
      boolean allowNetwork,
      boolean sandboxDebug,
      boolean useFakeHostname,
      boolean useFakeUsername)
      throws ExecException {
    Command cmd;
    try {
      cmd =
          getCommand(
              arguments, environment, timeout, allowNetwork, useFakeHostname, useFakeUsername);
    } catch (IOException e) {
      throw new UserExecException("I/O error during sandboxed execution", e);
    }

    TerminationStatus status = null;
    try {
      cmd.execute(
          /* stdin */ new byte[] {},
          getCommandObserver(timeout),
          outErr.getOutputStream(),
          outErr.getErrorStream(),
          /* killSubprocessOnInterrupt */ true);
    } catch (CommandException e) {
      boolean timedOut = false;
      if (e instanceof AbnormalTerminationException) {
        status = ((AbnormalTerminationException) e).getResult().getTerminationStatus();
        timedOut = !status.exited() && (status.getTerminatingSignal() == getSignalOnTimeout());
      }

      String statusMessage = status + " [sandboxed]";

      if (!verboseFailures) {
        // Simplest possible error message.
        throw new UserExecException(statusMessage, e, timedOut);
      }

      List<String> commandList = arguments;
      if (sandboxDebug) {
        // When using --sandbox_debug, show the entire command-line including the part where we call
        // the sandbox helper binary.
        commandList = Arrays.asList(cmd.getCommandLineElements());
      }

      String commandFailureMessage =
          CommandFailureUtils.describeCommandFailure(true, commandList, environment, null);

      if (!sandboxDebug) {
        commandFailureMessage += SANDBOX_DEBUG_SUGGESTION;
      }

      throw new UserExecException(commandFailureMessage, e, timedOut);
    }
  }

  /**
   * Returns the {@link Command} that the {@link #run} method will execute inside the sandbox.
   *
   * @param arguments - arguments of spawn to run inside the sandbox.
   * @param environment - environment variables to pass to the spawn.
   * @param timeout - after how many seconds should the process be killed.
   * @param allowNetwork - whether networking should be allowed for the process.
   * @param useFakeHostname - whether the hostname should be set to 'localhost' inside the sandbox.
   * @param useFakeUsername - whether the username should be set to 'nobody' inside the sandbox.
   */
  protected abstract Command getCommand(
      List<String> arguments,
      Map<String, String> environment,
      int timeout,
      boolean allowNetwork,
      boolean useFakeHostname,
      boolean useFakeUsername)
      throws IOException;

  /**
   * Returns a {@link KillableObserver} that the {@link #run} method will use when executing the
   * command returned by {@link #getCommand}.
   */
  protected KillableObserver getCommandObserver(int timeout) {
    return Command.NO_OBSERVER;
  }

  /**
   * Returns the signal code that the command returned by {@link #getCommand} exits with in case of
   * a timeout.
   */
  protected int getSignalOnTimeout() {
    return 14; /* SIGALRM */
  }
}
