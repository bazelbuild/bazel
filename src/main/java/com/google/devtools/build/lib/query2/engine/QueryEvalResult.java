// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.query2.engine;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.util.DetailedExitCode;

/**
 * Information about the query evaluation, like if it was successful and number of elements
 * returned.
 */
public class QueryEvalResult {

  private final boolean success;
  private final boolean empty;
  private final DetailedExitCode detailedExitCode;

  QueryEvalResult(boolean success, boolean empty, DetailedExitCode detailedExitCode) {
    checkNotNull(detailedExitCode);
    checkArgument(
        !success || detailedExitCode.isSuccess(),
        "successful query evaluations should not have non-success exit codes. detailedExitCode=%s",
        detailedExitCode);
    this.success = success;
    this.empty = empty;
    this.detailedExitCode = detailedExitCode;
  }

  public static QueryEvalResult success(boolean empty) {
    return new QueryEvalResult(true, empty, DetailedExitCode.success());
  }

  public static QueryEvalResult failure(boolean empty, DetailedExitCode detailedExitCode) {
    return new QueryEvalResult(false, empty, detailedExitCode);
  }

  /**
   * Whether the query was successful. This can only be false if the query was run with {@code
   * keep_going}, otherwise evaluation will throw a {@link QueryException}.
   */
  public boolean getSuccess() {
    return success;
  }

  /** True if the query did not return any result; */
  public boolean isEmpty() {
    return empty;
  }

  /**
   * Returns {@link DetailedExitCode#success()} if successful, and otherwise a {@link
   * DetailedExitCode} describing the failure if unsuccessful.
   */
  public DetailedExitCode getDetailedExitCode() {
    return detailedExitCode;
  }

  @Override
  public String toString() {
    return (getSuccess() ? "Successful" : "Unsuccessful") + ", empty = " + empty;
  }
}
