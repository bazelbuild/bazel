// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.causes;

import com.google.common.base.MoreObjects;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.ConfigurationId;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.util.DetailedExitCode;
import java.util.Objects;

/**
 * Class describing a {@link Cause} that can uniquely be described by a {@link Label} and {@link
 * com.google.devtools.build.lib.analysis.config.BuildConfigurationValue}.
 */
public class AnalysisFailedCause implements Cause {
  private final Label label;
  private final ConfigurationId configurationId;
  private final DetailedExitCode detailedExitCode;

  public AnalysisFailedCause(
      Label label, ConfigurationId configurationId, DetailedExitCode detailedExitCode) {
    this.label = label;
    this.configurationId = configurationId;
    this.detailedExitCode = detailedExitCode;
  }

  @Override
  public String toString() {
    // TODO(mschaller): Tests expect non-escaped message strings, and protobuf (the FailureDetail in
    //  detailedExitCode) escapes them. Better versions of tests would check structured data, and
    //  doing that requires unwinding test infrastructure. Note the "inTest" blocks in
    //  SkyframeBuildView#processAnalysisErrors.
    return MoreObjects.toStringHelper(this)
        .add("label", label)
        .add("configurationId", configurationId)
        .add("detailedExitCode", detailedExitCode)
        .add("msg", detailedExitCode.getFailureDetail().getMessage())
        .toString();
  }

  @Override
  public Label getLabel() {
    return label;
  }

  @Override
  public BuildEventStreamProtos.BuildEventId getIdProto() {
    // This needs to match AnalysisRootCauseEvent.getEventId.
    return BuildEventIdUtil.configuredLabelId(label, configurationId);
  }

  @Override
  public DetailedExitCode getDetailedExitCode() {
    return detailedExitCode;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    } else if (!(o instanceof AnalysisFailedCause)) {
      return false;
    }
    AnalysisFailedCause a = (AnalysisFailedCause) o;
    return Objects.equals(label, a.label)
        && Objects.equals(configurationId, a.configurationId)
        && Objects.equals(detailedExitCode, a.detailedExitCode);
  }

  @Override
  public int hashCode() {
    return Objects.hash(label, configurationId, detailedExitCode);
  }
}
