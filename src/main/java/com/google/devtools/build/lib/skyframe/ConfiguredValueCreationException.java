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

import static com.google.devtools.build.lib.analysis.config.BuildConfigurationValue.configurationId;

import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
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
import net.starlark.java.syntax.Location;

/**
 * An exception indicating that there was a problem during the construction of a
 * ConfiguredTargetValue.
 */
public final class ConfiguredValueCreationException extends Exception
    implements SaneAnalysisException {

  @Nullable private final Location location;
  private final BuildEventId configuration;
  private final NestedSet<Cause> rootCauses;
  // TODO(b/138456686): if warranted by a need for finer-grained details, replace the constructors
  //  that specify the general Code.CONFIGURED_VALUE_CREATION_FAILED
  private final DetailedExitCode detailedExitCode;

  public ConfiguredValueCreationException(
      @Nullable Location location,
      String message,
      Label label,
      BuildEventId configuration,
      @Nullable NestedSet<Cause> rootCauses,
      @Nullable DetailedExitCode detailedExitCode) {
    super(message);
    this.location = location;
    this.configuration = configuration;
    this.detailedExitCode =
        detailedExitCode != null ? detailedExitCode : createDetailedExitCode(message);
    this.rootCauses =
        rootCauses != null
            ? rootCauses
            : NestedSetBuilder.<Cause>stableOrder()
                .add(
                    new AnalysisFailedCause(
                        label, configuration.getConfiguration(), this.detailedExitCode))
                .build();
  }

  public ConfiguredValueCreationException(
      TargetAndConfiguration ctgValue,
      String message,
      @Nullable NestedSet<Cause> rootCauses,
      @Nullable DetailedExitCode detailedExitCode) {
    this(
        ctgValue.getTarget().getLocation(),
        message,
        ctgValue.getLabel(),
        configurationId(ctgValue.getConfiguration()),
        rootCauses,
        detailedExitCode);
  }

  public ConfiguredValueCreationException(TargetAndConfiguration ctgValue, String message) {
    this(ctgValue, message, /*rootCauses=*/ null, /*detailedExitCode=*/ null);
  }

  @Nullable
  public Location getLocation() {
    return location;
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

  private static DetailedExitCode createDetailedExitCode(String message) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setAnalysis(Analysis.newBuilder().setCode(Code.CONFIGURED_VALUE_CREATION_FAILED))
            .build());
  }
}
