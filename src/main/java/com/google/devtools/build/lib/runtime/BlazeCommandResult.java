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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.server.CommandProtos.ExecRequest;
import com.google.devtools.build.lib.util.ExitCode;
import javax.annotation.Nullable;

/**
 * The result of a Blaze command. It is usually an exit code, but can be an instruction to the
 * client to execute a particular binary for "blaze run".
 */
@Immutable
public final class BlazeCommandResult {
  private final ExitCode exitCode;
  @Nullable
  private final ExecRequest execDescription;

  private BlazeCommandResult(ExitCode exitCode, ExecRequest execDescription) {
    this.exitCode = Preconditions.checkNotNull(exitCode);
    this.execDescription = execDescription;
  }

  public ExitCode getExitCode() {
    return exitCode;
  }

  @Nullable public ExecRequest getExecRequest() {
    return execDescription;
  }

  public static BlazeCommandResult exitCode(ExitCode exitCode) {
    return new BlazeCommandResult(exitCode, null);
  }

  public static BlazeCommandResult execute(ExecRequest execDescription) {
    return new BlazeCommandResult(ExitCode.SUCCESS, Preconditions.checkNotNull(execDescription));
  }
}
