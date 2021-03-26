// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithOrderConstraint;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import javax.annotation.Nullable;

/** Event summarizing the building and testing (if applicable) of a given configured target. */
@Immutable
public final class TargetSummaryEvent implements BuildEventWithOrderConstraint {

  static TargetSummaryEvent create(
      ConfiguredTarget target,
      boolean overallBuildSuccess,
      boolean expectTestSummary,
      @Nullable BlazeTestStatus overallTestStatus) {
    Label label = target.getOriginalLabel();
    BuildEventId configId =
        target.getConfigurationChecksum() != null
            ? BuildEventIdUtil.configurationId(target.getConfigurationChecksum())
            : BuildEventIdUtil.nullConfigurationId();
    ImmutableList.Builder<BuildEventId> postAfter = ImmutableList.builder();
    postAfter.add(BuildEventIdUtil.targetCompleted(label, configId));
    if (expectTestSummary) {
      // Always post after test summary, even if we get here without having seen it yet
      postAfter.add(BuildEventIdUtil.testSummary(label, configId));
    }
    return new TargetSummaryEvent(
        BuildEventIdUtil.targetSummary(label, configId),
        overallBuildSuccess,
        overallBuildSuccess && expectTestSummary ? overallTestStatus : null,
        postAfter.build());
  }

  private final BuildEventId id;
  private final boolean overallBuildSuccess;
  @Nullable private final BlazeTestStatus overallTestStatus;
  private final ImmutableList<BuildEventId> postedAfter;

  private TargetSummaryEvent(
      BuildEventId id,
      boolean overallBuildSuccess,
      @Nullable BlazeTestStatus overallTestStatus,
      ImmutableList<BuildEventId> postedAfter) {
    checkArgument(id.hasTargetSummary(), "Unexpected event id: %s", id);
    this.id = id;
    this.overallBuildSuccess = overallBuildSuccess;
    this.overallTestStatus = overallTestStatus;
    this.postedAfter = postedAfter;
  }

  @VisibleForTesting
  boolean isOverallBuildSuccess() {
    return overallBuildSuccess;
  }

  @Nullable
  @VisibleForTesting
  BlazeTestStatus getOverallTestStatus() {
    return overallTestStatus;
  }

  @Override
  public ImmutableList<BuildEventId> postedAfter() {
    return postedAfter;
  }

  @Override
  public BuildEvent asStreamProto(BuildEventContext context) {
    BuildEventStreamProtos.TargetSummary.Builder summaryBuilder =
        BuildEventStreamProtos.TargetSummary.newBuilder()
            .setOverallBuildSuccess(overallBuildSuccess);
    if (overallBuildSuccess && overallTestStatus != null) {
      summaryBuilder.setOverallTestStatus(BuildEventStreamerUtils.bepStatus(overallTestStatus));
    }
    return GenericBuildEvent.protoChaining(this).setTargetSummary(summaryBuilder.build()).build();
  }

  @Override
  public BuildEventId getEventId() {
    return id;
  }

  @Override
  public ImmutableList<BuildEventId> getChildrenEvents() {
    return ImmutableList.of();
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("id", id)
        .add("overallBuildSuccess", overallBuildSuccess)
        .add("overallTestStatus", overallTestStatus)
        .add("postedAfter", postedAfter)
        .toString();
  }
}
