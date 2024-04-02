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
package com.google.devtools.build.lib.skyframe;

import static com.google.devtools.build.lib.analysis.config.BuildConfigurationValue.configurationIdMessage;

import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.causes.AnalysisFailedCause;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.causes.LabelCause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.server.FailureDetails.Analysis;
import com.google.devtools.build.lib.server.FailureDetails.Analysis.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.DetailedExitCode;
import javax.annotation.Nullable;

/** An exception indicating that there was a problem creating an aspect. */
public final class AspectCreationException extends AbstractSaneAnalysisException {
  private final NestedSet<Cause> causes;
  // TODO(b/138456686): if warranted by a need for finer-grained details, replace the constructors
  //  that specify the general Code.ASPECT_CREATION_FAILED
  private final DetailedExitCode detailedExitCode;

  public AspectCreationException(
      String message, NestedSet<Cause> causes, DetailedExitCode detailedExitCode) {
    super(message);
    this.causes = causes;
    this.detailedExitCode = detailedExitCode;
  }

  public AspectCreationException(
      String message,
      Label currentTarget,
      @Nullable BuildConfigurationValue configuration,
      DetailedExitCode detailedExitCode) {
    this(
        message,
        NestedSetBuilder.<Cause>stableOrder()
            .add(
                new AnalysisFailedCause(
                    currentTarget, configurationIdMessage(configuration), detailedExitCode))
            .build(),
        detailedExitCode);
  }

  public AspectCreationException(
      String message, Label currentTarget, @Nullable BuildConfigurationValue configuration) {
    this(
        message,
        currentTarget,
        configuration,
        createDetailedExitCode(message, Code.ASPECT_CREATION_FAILED));
  }

  public AspectCreationException(
      String message, Label currentTarget, DetailedExitCode detailedExitCode) {
    this(message, currentTarget, null, detailedExitCode);
  }

  public AspectCreationException(String message, Label currentTarget) {
    this(
        message, currentTarget, null, createDetailedExitCode(message, Code.ASPECT_CREATION_FAILED));
  }

  public AspectCreationException(String message, LabelCause cause) {
    this(
        message,
        NestedSetBuilder.<Cause>stableOrder().add(cause).build(),
        cause.getDetailedExitCode());
  }

  public NestedSet<Cause> getCauses() {
    return causes;
  }

  @Override
  public DetailedExitCode getDetailedExitCode() {
    return detailedExitCode;
  }

  private static DetailedExitCode createDetailedExitCode(String message, Code code) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setAnalysis(Analysis.newBuilder().setCode(code))
            .build());
  }
}
