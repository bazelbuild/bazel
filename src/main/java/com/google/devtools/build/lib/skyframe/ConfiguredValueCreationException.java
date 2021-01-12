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

import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.causes.AnalysisFailedCause;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.server.FailureDetails.Analysis;
import com.google.devtools.build.lib.server.FailureDetails.Analysis.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.DetailedExitCode;
import javax.annotation.Nullable;

/**
 * An exception indicating that there was a problem during the construction of a
 * ConfiguredTargetValue.
 */
public final class ConfiguredValueCreationException extends Exception
    implements SaneAnalysisException {

  @Nullable private final BuildEventId configuration;
  private final NestedSet<Cause> rootCauses;
  // TODO(b/138456686): if warranted by a need for finer-grained details, replace the constructors
  //  that specify the general Code.CONFIGURED_VALUE_CREATION_FAILED
  private final DetailedExitCode detailedExitCode;

  private ConfiguredValueCreationException(
      String message,
      @Nullable BuildEventId configuration,
      NestedSet<Cause> rootCauses,
      DetailedExitCode detailedExitCode) {
    super(message);
    this.configuration = configuration;
    this.rootCauses = rootCauses;
    this.detailedExitCode = detailedExitCode;
  }

  public ConfiguredValueCreationException(
      String message,
      Label currentTarget,
      @Nullable BuildConfiguration configuration,
      DetailedExitCode detailedExitCode) {
    this(
        message,
        configuration == null ? null : configuration.getEventId(),
        NestedSetBuilder.<Cause>stableOrder()
            .add(
                new AnalysisFailedCause(
                    currentTarget,
                    configuration == null ? null : configuration.getEventId().getConfiguration(),
                    detailedExitCode))
            .build(),
        detailedExitCode);
  }

  public ConfiguredValueCreationException(
      String message, Label currentTarget, @Nullable BuildConfiguration configuration) {
    this(
        message,
        currentTarget,
        configuration,
        createDetailedExitCode(message, Code.CONFIGURED_VALUE_CREATION_FAILED));
  }

  public ConfiguredValueCreationException(
      String message,
      @Nullable BuildConfiguration configuration,
      NestedSet<Cause> rootCauses,
      DetailedExitCode detailedExitCode) {
    this(
        message,
        configuration == null ? null : configuration.getEventId(),
        rootCauses,
        detailedExitCode);
  }

  public ConfiguredValueCreationException(
      String message, @Nullable BuildConfiguration configuration, NestedSet<Cause> rootCauses) {
    this(
        message,
        configuration,
        rootCauses,
        createDetailedExitCode(message, Code.CONFIGURED_VALUE_CREATION_FAILED));
  }

  public NestedSet<Cause> getRootCauses() {
    return rootCauses;
  }

  @Nullable
  public BuildEventId getConfiguration() {
    return configuration;
  }

  @Override
  public DetailedExitCode getDetailedExitCode() {
    return detailedExitCode;
  }

  private static DetailedExitCode createDetailedExitCode(String message, Analysis.Code code) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setAnalysis(Analysis.newBuilder().setCode(code))
            .build());
  }
}
