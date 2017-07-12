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
package com.google.devtools.build.lib.exec;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.exec.SpawnResult.Status;
import com.google.devtools.build.lib.shell.TerminationStatus;
import com.google.devtools.build.lib.util.ExitCode;
import java.util.Locale;

/**
 * A specialization of {@link ExecException} that indicates something went wrong when trying to
 * execute a {@link Spawn}.
 */
public class SpawnExecException extends ExecException {
  protected final SpawnResult result;
  protected final boolean forciblyRunRemotely;

  public SpawnExecException(String message, SpawnResult result, boolean catastrophe) {
    super(message, catastrophe);
    this.result = Preconditions.checkNotNull(result);
    this.forciblyRunRemotely = false;
  }

  public SpawnExecException(
      String message, SpawnResult result, boolean forciblyRunRemotely, boolean catastrophe) {
    super(message, catastrophe);
    this.result = Preconditions.checkNotNull(result);
    this.forciblyRunRemotely = forciblyRunRemotely;
  }

  /** Returns the spawn result. */
  public SpawnResult getSpawnResult() {
    return result;
  }

  @Override
  public boolean hasTimedOut() {
    return getSpawnResult().status() == Status.TIMEOUT;
  }

  @Override
  public ActionExecutionException toActionExecutionException(String messagePrefix,
        boolean verboseFailures, Action action) {
    TerminationStatus status = new TerminationStatus(
        result.exitCode(), result.status() == Status.TIMEOUT);
    String reason = " (" + status.toShortString() + ")"; // e.g " (Exit 1)"
    String explanation = status.exited() ? "" : ": " + getMessage();

    if (!result.status().isConsideredUserError()) {
      String errorDetail = result.status().name().toLowerCase(Locale.US)
          .replace('_', ' ');
      explanation += ". Note: Remote connection/protocol failed with: " + errorDetail;
    }
    if (result.status() == Status.TIMEOUT) {
      explanation +=
          String.format(
              " (failed due to timeout after %.2f seconds.)",
              result.getWallTimeMillis() / 1000.0f);
    } else if (result.status() == Status.OUT_OF_MEMORY) {
      explanation += " (Remote action was terminated due to Out of Memory.)";
    }
    if (result.status() != Status.TIMEOUT && forciblyRunRemotely) {
      explanation += " Action tagged as local was forcibly run remotely and failed - it's "
          + "possible that the action simply doesn't work remotely";
    }

    if (messagePrefix == null) {
      messagePrefix = action.describe();
    }
    // Note: we intentionally do not include the ExecException here, unless verboseFailures is true,
    // because it creates unwieldy and useless messages. If users need more info, they can run with
    // --verbose_failures.
    if (verboseFailures) {
      return new ActionExecutionException(
          messagePrefix + " failed" + reason + explanation,
          this,
          action,
          isCatastrophic(),
          getExitCode());
    } else {
      return new ActionExecutionException(
          messagePrefix + " failed" + reason + explanation,
          action,
          isCatastrophic(),
          getExitCode());
    }
  }

  /** Return exit code depending on the spawn result. */
  protected ExitCode getExitCode() {
    if (result.status().isConsideredUserError()) {
      return null;
    }
    return (result != null && result.status() == Status.REMOTE_EXECUTOR_OVERLOADED)
        ? ExitCode.REMOTE_EXECUTOR_OVERLOADED
        : ExitCode.REMOTE_ERROR;
  }
}
