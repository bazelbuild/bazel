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
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.ConfigurationId;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * Class describing a {@link Cause} that can uniquely be described by a {@link Label} and
 * {@link com.google.devtools.build.lib.analysis.config.BuildConfiguration}. Note that the
 * configuration may be null, in which case this generates an UnconfiguredLabel event.
 */
public class AnalysisFailedCause implements Cause {
  private final Label label;
  @Nullable private final ConfigurationId configuration;
  private final String msg;

  public AnalysisFailedCause(Label label, @Nullable ConfigurationId configuration, String msg) {
    this.label = label;
    this.configuration = configuration;
    this.msg = msg;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("label", label)
        .add("configuration", configuration)
        .add("msg", msg)
        .toString();
  }

  @Override
  public Label getLabel() {
    return label;
  }

  @Override
  public BuildEventStreamProtos.BuildEventId getIdProto() {
    // This needs to match AnalysisRootCauseEvent.
    if (configuration == null) {
      return BuildEventStreamProtos.BuildEventId.newBuilder()
          .setUnconfiguredLabel(
              BuildEventStreamProtos.BuildEventId.UnconfiguredLabelId.newBuilder()
                  .setLabel(label.toString())
                  .build())
          .build();
    }
    return BuildEventStreamProtos.BuildEventId.newBuilder()
        .setConfiguredLabel(
            BuildEventStreamProtos.BuildEventId.ConfiguredLabelId.newBuilder()
                .setLabel(label.toString())
                .setConfiguration(configuration)
                .build())
        .build();
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
        && Objects.equals(configuration, a.configuration)
        && Objects.equals(msg, a.msg);
  }

  @Override
  public int hashCode() {
    return Objects.hash(label, configuration, msg);
  }
}
