// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.test;

import com.google.common.base.Joiner;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.server.FailureDetails.Analysis;
import com.google.devtools.build.lib.server.FailureDetails.Analysis.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.AbstractSaneAnalysisException;
import com.google.devtools.build.lib.util.DetailedExitCode;

/**
 * An exception thrown when {@code --allow_analysis_failures=true} but information about some
 * analysis-phase failure could not be tracked and propagated using the normal mechanism (i.e. could
 * not be encapsulated by an AnalysisFailureInfo provided by a dummy ConfiguredTarget).
 *
 * <p>For instance, we do not allow one analysis failure test to transitively depend upon another
 * (see cl/220144957). In that case, we cannot create a ConfiguredTarget to propagate the failure to
 * create the inner test. Nor would we want to propagate it: the failure is a limitation of the
 * analysis testing machinery making the outer test unusable, not a failure in the rule(s) which the
 * outer test is testing.
 */
public final class AnalysisFailurePropagationException extends AbstractSaneAnalysisException {

  public AnalysisFailurePropagationException(Label label, Iterable<String> causes) {
    super(
        String.format(
            "Error while collecting analysis-phase failure information for '%s': %s",
            label, Joiner.on("; ").join(causes)));
  }

  @Override
  public DetailedExitCode getDetailedExitCode() {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(getMessage())
            .setAnalysis(Analysis.newBuilder().setCode(Code.ANALYSIS_FAILURE_PROPAGATION_FAILED))
            .build());
  }
}
