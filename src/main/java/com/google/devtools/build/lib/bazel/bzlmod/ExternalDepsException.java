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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.server.FailureDetails.ExternalDeps;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.DetailedException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.errorprone.annotations.FormatMethod;
import com.google.errorprone.annotations.FormatString;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkThread.CallStackEntry;

/** Exception indicates that something went wrong while processing external dependencies. */
public class ExternalDepsException extends Exception implements DetailedException {

  private final DetailedExitCode detailedExitCode;

  private ExternalDepsException(String message, @Nullable Throwable cause, ExternalDeps.Code code) {
    super(message, cause);
    detailedExitCode =
        DetailedExitCode.of(
            FailureDetail.newBuilder()
                .setMessage(message)
                .setExternalDeps(ExternalDeps.newBuilder().setCode(code).build())
                .build());
  }

  @FormatMethod
  public static ExternalDepsException withMessage(
      ExternalDeps.Code code, @FormatString String format, Object... args) {
    return new ExternalDepsException(String.format(format, args), null, code);
  }

  @FormatMethod
  public static ExternalDepsException withCallStackAndMessage(
      ExternalDeps.Code code,
      ImmutableList<CallStackEntry> callStack,
      @FormatString String format,
      Object... args) {
    return new ExternalDepsException(
        EvalException.formatCallStack(
            callStack, String.format(format, args), EvalException.newSourceReader()),
        null,
        code);
  }

  @FormatMethod
  public static ExternalDepsException withCauseAndMessage(
      ExternalDeps.Code code, Throwable cause, @FormatString String format, Object... args) {
    return new ExternalDepsException(
        String.format(format, args) + ": " + cause.getMessage(), cause, code);
  }

  public static ExternalDepsException withCause(ExternalDeps.Code code, Throwable cause) {
    return new ExternalDepsException(cause.getMessage(), cause, code);
  }

  @Override
  public DetailedExitCode getDetailedExitCode() {
    return detailedExitCode;
  }
}
