// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.cmdline;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.io.InconsistentFilesystemException;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.TargetPatterns;
import com.google.devtools.build.lib.skyframe.DetailedException;
import com.google.devtools.build.lib.util.DetailedExitCode;

/** An exception indicating a target label that cannot be parsed. */
public class TargetParsingException extends Exception implements DetailedException {

  private final DetailedExitCode detailedExitCode;

  public TargetParsingException(String message, TargetPatterns.Code code) {
    super(Preconditions.checkNotNull(message));
    this.detailedExitCode = DetailedExitCode.of(createFailureDetail(message, code));
  }

  public TargetParsingException(String message, Throwable cause, TargetPatterns.Code code) {
    super(Preconditions.checkNotNull(message), cause);
    this.detailedExitCode = DetailedExitCode.of(createFailureDetail(message, code));
  }

  public TargetParsingException(
      String message, Throwable cause, DetailedExitCode detailedExitCode) {
    super(Preconditions.checkNotNull(message), cause);
    this.detailedExitCode = Preconditions.checkNotNull(detailedExitCode);
  }

  public TargetParsingException(String message, DetailedExitCode detailedExitCode) {
    super(Preconditions.checkNotNull(message));
    this.detailedExitCode = Preconditions.checkNotNull(detailedExitCode);
  }

  public TargetParsingException(InconsistentFilesystemException cause) {
    super(cause.getMessage(), cause);
    this.detailedExitCode =
        DetailedExitCode.of(
            FailureDetails.FailureDetail.newBuilder()
                .setPackageLoading(
                    FailureDetails.PackageLoading.newBuilder()
                        .setCode(
                            FailureDetails.PackageLoading.Code
                                .TRANSIENT_INCONSISTENT_FILESYSTEM_ERROR))
                .setMessage(getMessage())
                .build());
  }

  private static FailureDetail createFailureDetail(String message, TargetPatterns.Code code) {
    return FailureDetail.newBuilder()
        .setMessage(message)
        .setTargetPatterns(TargetPatterns.newBuilder().setCode(code).build())
        .build();
  }

  /**
   * Returns the detailed exit code that contains the failure detail associated with the error
   * during parsing.
   */
  @Override
  public DetailedExitCode getDetailedExitCode() {
    return detailedExitCode;
  }
}
