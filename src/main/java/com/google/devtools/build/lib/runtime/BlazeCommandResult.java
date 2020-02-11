// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime;

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.server.CommandProtos.ExecRequest;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.FailureDetailUtil;
import javax.annotation.Nullable;

/**
 * The result of a Blaze command. It is usually an exit code, but can be an instruction to the
 * client to execute a particular binary for "blaze run".
 */
@Immutable
public final class BlazeCommandResult {
  private final ExitCode exitCode;
  @Nullable private final FailureDetail failureDetail;
  @Nullable private final ExecRequest execDescription;
  private final boolean shutdown;

  private BlazeCommandResult(
      ExitCode exitCode,
      @Nullable FailureDetail failureDetail,
      ExecRequest execDescription,
      boolean shutdown) {
    this.exitCode = Preconditions.checkNotNull(exitCode);
    this.failureDetail = failureDetail;
    this.execDescription = execDescription;
    this.shutdown = shutdown;
  }

  public ExitCode getExitCode() {
    return exitCode;
  }

  @Nullable
  public FailureDetail getFailureDetail() {
    return failureDetail;
  }

  public boolean shutdown() {
    return shutdown;
  }

  @Nullable
  public ExecRequest getExecRequest() {
    return execDescription;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("exitCode", exitCode)
        .add("failureDetail", failureDetail)
        .add("execDescription", execDescription)
        .add("shutdown", shutdown)
        .toString();
  }

  public static BlazeCommandResult shutdown(ExitCode exitCode) {
    return new BlazeCommandResult(exitCode, null, null, true);
  }

  public static BlazeCommandResult exitCode(ExitCode exitCode) {
    return new BlazeCommandResult(exitCode, null, null, false);
  }

  public static BlazeCommandResult failureDetail(FailureDetail failureDetail) {
    return new BlazeCommandResult(
        FailureDetailUtil.getExitCode(failureDetail), failureDetail, null, false);
  }

  public static BlazeCommandResult execute(ExecRequest execDescription) {
    return new BlazeCommandResult(
        ExitCode.SUCCESS, null, Preconditions.checkNotNull(execDescription), false);
  }
}
