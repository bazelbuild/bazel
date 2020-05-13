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

package com.google.devtools.build.lib.analysis;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.buildeventstream.NullConfiguration;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import java.util.Collection;
import javax.annotation.Nullable;

/**
 * This event is fired during the build, when it becomes known that the analysis of a top-level
 * target cannot be completed because of an error in one of its dependencies.
 */
public class AnalysisFailureEvent implements BuildEvent {
  private final ConfiguredTargetKey failedTarget;
  private final BuildEventId configuration;
  private final NestedSet<Cause> rootCauses;

  public AnalysisFailureEvent(
      ConfiguredTargetKey failedTarget, BuildEventId configuration, NestedSet<Cause> rootCauses) {
    this.failedTarget = failedTarget;
    if (configuration != null) {
      this.configuration = configuration;
    } else {
      this.configuration = NullConfiguration.INSTANCE.getEventId();
    }
    this.rootCauses = rootCauses;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("failedTarget", failedTarget)
        .add("configuration", configuration)
        .add("legacyFailureReason", getLegacyFailureReason())
        .toString();
  }

  public ConfiguredTargetKey getFailedTarget() {
    return failedTarget;
  }

  @VisibleForTesting
  BuildEventId getConfigurationId() {
    return configuration;
  }

  /**
   * Returns the label of a single root cause. Use {@link #getRootCauses} to report all root causes.
   */
  @Nullable public Label getLegacyFailureReason() {
    if (rootCauses.isEmpty()) {
      return null;
    }
    return rootCauses.toList().get(0).getLabel();
  }

  public NestedSet<Cause> getRootCauses() {
    return rootCauses;
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventIdUtil.targetCompleted(failedTarget.getLabel(), configuration);
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.copyOf(
        Iterables.transform(rootCauses.toList(), cause -> cause.getIdProto()));
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    return GenericBuildEvent.protoChaining(this)
        .setAborted(
            BuildEventStreamProtos.Aborted.newBuilder()
                .setReason(BuildEventStreamProtos.Aborted.AbortReason.ANALYSIS_FAILURE)
                .build())
        .build();
  }
}
