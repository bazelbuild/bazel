// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.StarlarkLoading;
import com.google.devtools.build.lib.server.FailureDetails.StarlarkLoading.Code;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;

/** Exceptions from {@link BzlLoadFunction}. */
public final class BzlLoadFailedException extends AbstractSaneAnalysisException {
  private final Transience transience;
  private final DetailedExitCode detailedExitCode;

  private BzlLoadFailedException(
      String errorMessage, DetailedExitCode detailedExitCode, Transience transience) {
    super(errorMessage);
    this.transience = transience;
    this.detailedExitCode = detailedExitCode;
  }

  BzlLoadFailedException(String errorMessage, DetailedExitCode detailedExitCode) {
    this(errorMessage, detailedExitCode, Transience.PERSISTENT);
  }

  BzlLoadFailedException(
      String errorMessage,
      DetailedExitCode detailedExitCode,
      Exception cause,
      Transience transience) {
    super(errorMessage, cause);
    this.transience = transience;
    this.detailedExitCode = detailedExitCode;
  }

  BzlLoadFailedException(String errorMessage, Code code) {
    this(errorMessage, createDetailedExitCode(errorMessage, code), Transience.PERSISTENT);
  }

  BzlLoadFailedException(String errorMessage, Code code, Exception cause, Transience transience) {
    this(errorMessage, createDetailedExitCode(errorMessage, code), cause, transience);
  }

  Transience getTransience() {
    return transience;
  }

  @Override
  public DetailedExitCode getDetailedExitCode() {
    return detailedExitCode;
  }

  static DetailedExitCode createDetailedExitCode(String message, Code code) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setStarlarkLoading(StarlarkLoading.newBuilder().setCode(code))
            .build());
  }
}
