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
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/** A common interface of all sandbox runners, no matter which platform they're working on. */
abstract class SandboxRunner {

  private final boolean verboseFailures;
  private final Path sandboxExecRoot;

  SandboxRunner(Path sandboxExecRoot, boolean verboseFailures) {
    this.sandboxExecRoot = sandboxExecRoot;
    this.verboseFailures = verboseFailures;
  }

  /**
   * Runs the command specified via {@code arguments} and {@code env} inside the sandbox.
   *
   * @param arguments - arguments of spawn to run inside the sandbox.
   * @param environment - environment variables to pass to the spawn.
   * @param outErr - error output to capture sandbox's and command's stderr
   * @param timeout - after how many seconds should the process be killed
   * @param allowNetwork - whether networking should be allowed for the process
   */
  void run(
      List<String> arguments,
      Map<String, String> environment,
      OutErr outErr,
      int timeout,
      boolean allowNetwork)
      throws ExecException {
    Command cmd;
    try {
      cmd = getCommand(arguments, environment, timeout, allowNetwork);
    } catch (IOException e) {
      throw new UserExecException("I/O error during sandboxed execution", e);
    }

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
        TerminationStatus status =
            ((AbnormalTerminationException) e).getResult().getTerminationStatus();
        timedOut = !status.exited() && (status.getTerminatingSignal() == getSignalOnTimeout());
      }
      String message =
          CommandFailureUtils.describeCommandFailure(
              verboseFailures,
              Arrays.asList(cmd.getCommandLineElements()),
              environment,
              sandboxExecRoot.getPathString());
      throw new UserExecException(message, e, timedOut);
    }
  }

  /**
   * Returns the {@link Command} that the {@link #run} method will execute inside the sandbox.
   *
   * @param arguments - arguments of spawn to run inside the sandbox.
   * @param environment - environment variables to pass to the spawn.
   * @param timeout - after how many seconds should the process be killed
   * @param allowNetwork - whether networking should be allowed for the process
   */
  protected abstract Command getCommand(
      List<String> arguments, Map<String, String> environment, int timeout, boolean allowNetwork)
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
