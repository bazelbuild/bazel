// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.exec.local;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Spawn.Code;
import com.google.devtools.build.lib.util.NetUtil;
import com.google.devtools.common.options.RegexPatternOption;
import java.io.IOException;
import java.io.OutputStream;
import javax.annotation.Nullable;

/**
 * Enforces {@code --allowed_local_actions_regex} for spawn runners that start subprocesses on the
 * machine Bazel runs on.
 */
public final class AllowedLocalActions {
  private static final int LOCAL_EXEC_ERROR = -1;

  private AllowedLocalActions() {}

  /**
   * Returns whether {@code --allowed_local_actions_regex} permits running {@code mnemonic} on this
   * machine.
   */
  public static boolean isAllowed(LocalExecutionOptions options, String mnemonic) {
    RegexPatternOption allowed = options.getAllowedLocalAction();
    return allowed == null || allowed.matcher().test(mnemonic);
  }

  /**
   * Returns a failed result if {@code --allowed_local_actions_regex} forbids running {@code
   * mnemonic} on this machine, or {@code null} if it may run.
   *
   * <p>Runners must call this before doing any other work for the spawn, so that a forbidden spawn
   * neither fetches its inputs nor occupies local resources.
   */
  @Nullable
  public static SpawnResult denyIfNotAllowed(
      LocalExecutionOptions options,
      SpawnRunner runner,
      String mnemonic,
      SpawnExecutionContext context)
      throws IOException {
    RegexPatternOption allowed = options.getAllowedLocalAction();
    if (allowed == null || allowed.matcher().test(mnemonic)) {
      return null;
    }
    OutputStream err = context.getFileOutErr().getErrorStream();
    err.write(
        ("Action type "
                + mnemonic
                + " is not allowed to run locally due to regex filter: "
                + allowed.regexPattern()
                + "\n")
            .getBytes(UTF_8));
    err.flush();
    return runner
        .getSpawnResultBuilder(context)
        .setExecutorHostname(NetUtil.getCachedShortHostName())
        .setStatus(Status.EXECUTION_DENIED)
        .setExitCode(LOCAL_EXEC_ERROR)
        .setFailureDetail(
            FailureDetail.newBuilder()
                .setMessage("local spawn failed for " + mnemonic)
                .setSpawn(FailureDetails.Spawn.newBuilder().setCode(Code.EXECUTION_DENIED))
                .build())
        .build();
  }
}
