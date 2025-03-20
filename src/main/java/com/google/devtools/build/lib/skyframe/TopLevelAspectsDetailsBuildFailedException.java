// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.server.FailureDetails.Analysis;
import com.google.devtools.build.lib.server.FailureDetails.Analysis.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.DetailedExitCode;

/** Exception that encapsulates the ones that's thrown when evaluating the top level aspects. */
public final class TopLevelAspectsDetailsBuildFailedException
    extends AbstractSaneAnalysisException {
  private final DetailedExitCode detailedExitCode;

  TopLevelAspectsDetailsBuildFailedException(String errorMessage, Code code) {
    super(errorMessage);
    this.detailedExitCode =
        DetailedExitCode.of(
            FailureDetail.newBuilder()
                .setMessage(errorMessage)
                .setAnalysis(Analysis.newBuilder().setCode(code))
                .build());
  }

  @Override
  public DetailedExitCode getDetailedExitCode() {
    return detailedExitCode;
  }
}
