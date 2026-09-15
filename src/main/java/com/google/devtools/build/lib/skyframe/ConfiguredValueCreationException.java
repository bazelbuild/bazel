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


import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.ConfigurationId;
import com.google.devtools.build.lib.causes.AnalysisFailedCause;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.Target;
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
public final class ConfiguredValueCreationException extends AbstractSaneAnalysisException {

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
    DetailedExitCode exitCode =
        detailedExitCode != null ? detailedExitCode : createDetailedExitCode(message);
    this.detailedExitCode = exitCode;
    this.rootCauses =
        rootCauses != null
            ? rootCauses
            : NestedSetBuilder.create(
                Order.STABLE_ORDER, createRootCause(label, configuration, exitCode));
  }

  public ConfiguredValueCreationException(
      @Nullable Target target,
      @Nullable BuildEventId configuration,
      String message,
      @Nullable NestedSet<Cause> rootCauses,
      @Nullable DetailedExitCode detailedExitCode) {
    this(
        target == null ? null : target.getLocation(),
        message,
        target.getLabel(),
        configuration,
        rootCauses,
        detailedExitCode);
  }

  public ConfiguredValueCreationException(@Nullable Target target, String message) {
    this(
        target,
        /* configuration= */ null,
        message,
        /* rootCauses= */ null,
        /* detailedExitCode= */ null);
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

  private static AnalysisFailedCause createRootCause(
      Label label, BuildEventId configuration, DetailedExitCode detailedExitCode) {
    return new AnalysisFailedCause(
        label,
        configuration == null
            ? ConfigurationId.newBuilder().setId("none").build()
            : configuration.getConfiguration(),
        detailedExitCode);
  }
}
