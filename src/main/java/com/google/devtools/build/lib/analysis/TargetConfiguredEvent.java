// Copyright 2017 The Bazel Authors. All rights reserved.
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
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithConfiguration;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.TestSize;
import com.google.devtools.build.lib.packages.Type;
import javax.annotation.Nullable;

/** Event reporting about the configuration associated with a given target */
public class TargetConfiguredEvent implements BuildEventWithConfiguration {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private final Target target;
  @Nullable private final BuildConfigurationValue configuration;

  public TargetConfiguredEvent(Target target, @Nullable BuildConfigurationValue configuration) {
    this.target = target;
    this.configuration = configuration;
  }

  @Override
  public ImmutableList<BuildEvent> getConfigurations() {
    return ImmutableList.of(BuildConfigurationValue.buildEvent(configuration));
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventIdUtil.targetConfigured(target.getLabel());
  }

  @Override
  public ImmutableList<BuildEventId> getChildrenEvents() {
    return ImmutableList.of(
        BuildEventIdUtil.targetCompleted(
            target.getLabel(), BuildConfigurationValue.configurationId(configuration)));
  }

  private static BuildEventStreamProtos.TestSize bepTestSize(String targetName, TestSize size) {
    if (size != null) {
      switch (size) {
        case SMALL:
          return BuildEventStreamProtos.TestSize.SMALL;
        case MEDIUM:
          return BuildEventStreamProtos.TestSize.MEDIUM;
        case LARGE:
          return BuildEventStreamProtos.TestSize.LARGE;
        case ENORMOUS:
          return BuildEventStreamProtos.TestSize.ENORMOUS;
      }
    }
    logger.atInfo().log("Target %s has a test size of: %s", targetName, size);
    return BuildEventStreamProtos.TestSize.UNKNOWN;
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    BuildEventStreamProtos.TargetConfigured.Builder builder =
        BuildEventStreamProtos.TargetConfigured.newBuilder().setTargetKind(target.getTargetKind());
    Rule rule = target.getAssociatedRule();
    if (rule != null && RawAttributeMapper.of(rule).has("tags")) {
      // Not every rule has tags, as, due to the "external" package we also have to expect
      // repository rules at this place.
      builder.addAllTag(RawAttributeMapper.of(rule).getMergedValues("tags", Type.STRING_LIST));
    }
    if (TargetUtils.isTestRule(target)) {
      builder.setTestSize(
          bepTestSize(target.getName(), TestSize.getTestSize(target.getAssociatedRule())));
    }
    return GenericBuildEvent.protoChaining(this).setConfigured(builder.build()).build();
  }
}
