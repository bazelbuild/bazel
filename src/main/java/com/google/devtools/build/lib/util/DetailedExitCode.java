// Copyright 2020 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import javax.annotation.Nullable;

/** An {@link ExitCode} and an optional {@link FailureDetail}. */
public class DetailedExitCode {
  private final ExitCode exitCode;
  @Nullable private final FailureDetail failureDetail;

  private DetailedExitCode(ExitCode exitCode, @Nullable FailureDetail failureDetail) {
    this.exitCode = exitCode;
    this.failureDetail = failureDetail;
  }

  public ExitCode getExitCode() {
    return exitCode;
  }

  @Nullable
  public FailureDetail getFailureDetail() {
    return failureDetail;
  }

  public boolean isSuccess() {
    return exitCode.equals(ExitCode.SUCCESS);
  }

  /**
   * Returns a {@link DetailedExitCode} specifying {@link ExitCode} but no {@link FailureDetail}.
   *
   * <p>This method exists in order to allow for code which has not yet been wired for {@link
   * FailureDetail) support to interact with {@link FailureDetail}-handling code infrastructure.
   *
   * <p>Callsites should migrate to using either:
   *
   * <ul>
   *   <li>{@link #of(ExitCode, FailureDetail)}, when they're wired for {@link FailureDetail}
   *   support but not yet ready to have {@link FailureDetail} metadata determine exit code behavior
   *   <li>{@link #of(FailureDetail)}, when changing exit code behavior is desired.
   * </ul>
   *
   */
  public static DetailedExitCode justExitCode(ExitCode exitCode) {
    return new DetailedExitCode(checkNotNull(exitCode), null);
  }

  /**
   * Returns a {@link DetailedExitCode} combining the provided {@link FailureDetail} and {@link
   * ExitCode}.
   *
   * <p>This method exists in order to allow for the introduction of new {@link
   * FailureDetail)-handling code infrastructure without requiring any simultaneous change in exit
   * code behavior.
   *
   * <p>Callsites should migrate to using {@link #of(FailureDetail)} instead.
   */
  // TODO(b/138456686): consider controlling this behavior by flag if migration appears risky.
  public static DetailedExitCode of(ExitCode exitCode, FailureDetail failureDetail) {
    return new DetailedExitCode(checkNotNull(exitCode), checkNotNull(failureDetail));
  }

  /**
   * Returns a {@link DetailedExitCode} whose {@link ExitCode} is chosen referencing {@link
   * FailureDetail}'s metadata.
   */
  public static DetailedExitCode of(FailureDetail failureDetail) {
    return new DetailedExitCode(FailureDetailUtil.getExitCode(failureDetail), failureDetail);
  }

  @Override
  public String toString() {
    return String.format(
        "DetailedExitCode{exitCode=%s, failureDetail=%s}", exitCode, failureDetail);
  }
}
