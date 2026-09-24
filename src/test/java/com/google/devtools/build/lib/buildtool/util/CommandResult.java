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
package com.google.devtools.build.lib.buildtool.util;

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import java.time.Duration;
import java.util.regex.Pattern;

/** Result of executing a command through {@link BazelServer}. */
public final class CommandResult {
  private static final Splitter LINE_SPLITTER = Splitter.on(Pattern.compile("\r?\n"));

  private final BlazeCommandResult blazeCommandResult;
  private final String standardOutput;
  private final String standardError;
  private final Duration executionTime;

  public CommandResult(
      BlazeCommandResult blazeCommandResult,
      String standardOutput,
      String standardError,
      Duration executionTime) {
    this.blazeCommandResult = blazeCommandResult;
    this.standardOutput = standardOutput;
    this.standardError = standardError;
    this.executionTime = executionTime;
  }

  /** Returns true if the command succeeded (exit code 0). */
  public boolean isSuccess() {
    return blazeCommandResult.isSuccess();
  }

  /**
   * Returns true if the command succeeded (exit code 0). Backwards-compatible alias for {@link
   * #isSuccess()}.
   */
  public boolean getSuccess() {
    return isSuccess();
  }

  /** Returns the exit code object. */
  public ExitCode exitCode() {
    return blazeCommandResult.getExitCode();
  }

  /** Returns the exit code object. */
  public ExitCode getExitCode() {
    return exitCode();
  }

  /** Returns the numeric exit code. */
  public int numericExitCode() {
    return blazeCommandResult.getExitCode().getNumericExitCode();
  }

  /** Returns the detailed exit code. */
  public DetailedExitCode getDetailedExitCode() {
    return blazeCommandResult.getDetailedExitCode();
  }

  /** Returns the detailed exit code. Alias for getDetailedExitCode. */
  public DetailedExitCode detailedExitCode() {
    return getDetailedExitCode();
  }

  /** Returns the underlying {@link BlazeCommandResult}. */
  public BlazeCommandResult blazeCommandResult() {
    return blazeCommandResult;
  }

  /** Returns captured stdout string. */
  public String getStandardOutput() {
    return standardOutput;
  }

  /** Returns captured stderr string. */
  public String getStandardError() {
    return standardError;
  }

  /** Returns lines of captured stdout. */
  public ImmutableList<String> getOutputLines() {
    if (standardOutput.isEmpty()) {
      return ImmutableList.of();
    }
    return ImmutableList.copyOf(LINE_SPLITTER.splitToList(standardOutput));
  }

  /** Returns lines of captured stderr. */
  public ImmutableList<String> getErrorLines() {
    if (standardError.isEmpty()) {
      return ImmutableList.of();
    }
    return ImmutableList.copyOf(LINE_SPLITTER.splitToList(standardError));
  }

  /** Returns elapsed execution time. */
  public Duration executionTime() {
    return executionTime;
  }

  @Override
  public String toString() {
    return String.format(
        "CommandResult{success=%s, exitCode=%s, stdout=[%s], stderr=[%s]}",
        isSuccess(), exitCode(), standardOutput, standardError);
  }
}
