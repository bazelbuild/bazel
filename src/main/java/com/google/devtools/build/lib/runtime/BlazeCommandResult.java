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
import com.google.devtools.build.lib.util.CrashFailureDetails;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import javax.annotation.Nullable;

/**
 * The result of a Blaze command. It is usually a {@link ExitCode} with optional {@link
 * FailureDetail}, but can be an instruction to the client to execute a particular binary for "blaze
 * run".
 */
@Immutable
public final class BlazeCommandResult {
  private final DetailedExitCode detailedExitCode;

  @Nullable private final ExecRequest execDescription;
  private final boolean shutdown;

  private BlazeCommandResult(
      DetailedExitCode detailedExitCode, @Nullable ExecRequest execDescription, boolean shutdown) {
    this.detailedExitCode = Preconditions.checkNotNull(detailedExitCode);
    this.execDescription = execDescription;
    this.shutdown = shutdown;
  }

  public ExitCode getExitCode() {
    return detailedExitCode.getExitCode();
  }

  public DetailedExitCode getDetailedExitCode() {
    return detailedExitCode;
  }

  @Nullable
  public FailureDetail getFailureDetail() {
    return detailedExitCode.getFailureDetail();
  }

  public boolean shutdown() {
    return shutdown;
  }

  @Nullable
  public ExecRequest getExecRequest() {
    return execDescription;
  }

  public boolean isSuccess() {
    return detailedExitCode.isSuccess();
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("exitCode", getExitCode())
        .add("failureDetail", getFailureDetail())
        .add("execDescription", execDescription)
        .add("shutdown", shutdown)
        .toString();
  }

  public static BlazeCommandResult shutdownOnSuccess() {
    return new BlazeCommandResult(DetailedExitCode.success(), null, true);
  }

  static BlazeCommandResult createShutdown(Throwable e) {
    return new BlazeCommandResult(CrashFailureDetails.detailedExitCodeForThrowable(e), null, true);
  }

  public static BlazeCommandResult success() {
    return new BlazeCommandResult(DetailedExitCode.success(), null, false);
  }

  public static BlazeCommandResult failureDetail(FailureDetail failureDetail) {
    return new BlazeCommandResult(DetailedExitCode.of(failureDetail), null, false);
  }

  public static BlazeCommandResult detailedExitCode(DetailedExitCode detailedExitCode) {
    return new BlazeCommandResult(detailedExitCode, null, false);
  }

  public static BlazeCommandResult execute(ExecRequest execDescription) {
    return new BlazeCommandResult(
        DetailedExitCode.success(), Preconditions.checkNotNull(execDescription), false);
  }
}
