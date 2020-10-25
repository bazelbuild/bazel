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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;

/**
 * A specialization of {@link ExecException} that indicates something went wrong when trying to
 * execute a {@link com.google.devtools.build.lib.actions.Spawn}.
 */
// Non-final only for tests, do not subclass!
public class SpawnExecException extends ExecException {
  protected final SpawnResult result;
  protected final boolean forciblyRunRemotely;

  public SpawnExecException(String message, SpawnResult result, boolean forciblyRunRemotely) {
    super(message, result.isCatastrophe());
    checkArgument(
        !Status.SUCCESS.equals(result.status()),
        "Can't create exception with successful spawn result.");
    this.result = checkNotNull(result);
    this.forciblyRunRemotely = forciblyRunRemotely;
  }

  @VisibleForTesting
  public SpawnExecException(
      String message, SpawnResult result, boolean forciblyRunRemotely, boolean catastrophe) {
    super(message, catastrophe);
    this.result = checkNotNull(result);
    this.forciblyRunRemotely = forciblyRunRemotely;
  }

  /** Returns the spawn result. */
  public SpawnResult getSpawnResult() {
    return result;
  }

  public boolean hasTimedOut() {
    return getSpawnResult().status() == Status.TIMEOUT;
  }

  @Override
  protected String getMessageForActionExecutionException() {
    return result.getDetailMessage(getMessage(), isCatastrophic(), forciblyRunRemotely);
  }

  @Override
  protected FailureDetail getFailureDetail(String message) {
    return checkNotNull(result.failureDetail(), this);
  }
}
