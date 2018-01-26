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

package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventConverters;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.buildeventstream.NullConfiguration;
import com.google.devtools.build.lib.causes.LabelCause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import java.util.Collection;

/**
 * This event is fired during the build, when it becomes known that the analysis of a target cannot
 * be completed because of an error in one of its dependencies.
 */
public class AnalysisFailureEvent implements BuildEvent {
  private final ConfiguredTargetKey failedTarget;
  private final Label failureReason;
  private final BuildEventId configuration;

  public AnalysisFailureEvent(
      ConfiguredTargetKey failedTarget, BuildEventId configuration, Label failureReason) {
    this.failedTarget = failedTarget;
    this.failureReason = failureReason;
    if (configuration != null) {
      this.configuration = configuration;
    } else {
      this.configuration = (new NullConfiguration()).getEventId();
    }
  }

  public AnalysisFailureEvent(ConfiguredTargetKey failedTarget, Label failureReason) {
    this(failedTarget, null, failureReason);
  }

  public ConfiguredTargetKey getFailedTarget() {
    return failedTarget;
  }

  public Label getFailureReason() {
    return failureReason;
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventId.targetCompleted(failedTarget.getLabel(), configuration);
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    // TODO(aehlig): the root cause is not necessarily a label; e.g., it could
    // also be a configured label.
    return ImmutableList.of(BuildEventId.fromCause(new LabelCause(failureReason)));
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventConverters converters) {
    return GenericBuildEvent.protoChaining(this)
        .setAborted(
            BuildEventStreamProtos.Aborted.newBuilder()
                .setReason(BuildEventStreamProtos.Aborted.AbortReason.ANALYSIS_FAILURE)
                .build())
        .build();
  }
}
