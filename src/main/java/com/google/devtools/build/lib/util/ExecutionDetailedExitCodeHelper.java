// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util;

import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.server.FailureDetails.Execution;
import com.google.devtools.build.lib.server.FailureDetails.Execution.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.skyframe.EvaluationResult;

/** A collection of helper methods around execution-related DetailedExitCode. */
public final class ExecutionDetailedExitCodeHelper {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  public static DetailedExitCode createDetailedExitCodeForUndetailedExecutionCause(
      EvaluationResult<?> result, Throwable undetailedCause) {
    if (undetailedCause == null) {
      logger.atWarning().log("No exceptions found despite error in %s", result);
      return createDetailedExecutionExitCode(
          "keep_going execution failed without an action failure",
          Code.NON_ACTION_EXECUTION_FAILURE);
    }
    logger.atWarning().withCause(undetailedCause).log("No detailed exception found in %s", result);
    return createDetailedExecutionExitCode(
        "keep_going execution failed without an action failure: "
            + undetailedCause.getMessage()
            + " ("
            + undetailedCause.getClass().getSimpleName()
            + ")",
        Code.NON_ACTION_EXECUTION_FAILURE);
  }

  public static DetailedExitCode createDetailedExecutionExitCode(
      String message, Code detailedCode) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setExecution(Execution.newBuilder().setCode(detailedCode))
            .build());
  }

  private ExecutionDetailedExitCodeHelper() {}
}
