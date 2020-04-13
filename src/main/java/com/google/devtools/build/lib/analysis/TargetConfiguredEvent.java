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
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithConfiguration;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.buildeventstream.NullConfiguration;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.TestSize;
import com.google.devtools.build.lib.packages.Type;
import java.util.Collection;

/** Event reporting about the configurations associated with a given target */
public class TargetConfiguredEvent implements BuildEventWithConfiguration {
  private final Target target;
  private final Collection<BuildConfiguration> configurations;

  TargetConfiguredEvent(Target target, Collection<BuildConfiguration> configurations) {
    this.configurations = configurations;
    this.target = target;
  }

  @Override
  public Collection<BuildEvent> getConfigurations() {
    ImmutableList.Builder<BuildEvent> builder = new ImmutableList.Builder<>();
    for (BuildConfiguration config : configurations) {
      if (config != null) {
        builder.add(config.toBuildEvent());
      } else {
        builder.add(new NullConfiguration());
      }
    }
    return builder.build();
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventIdUtil.targetConfigured(target.getLabel());
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    ImmutableList.Builder<BuildEventId> childrenBuilder = ImmutableList.builder();
    for (BuildConfiguration config : configurations) {
      if (config != null) {
        childrenBuilder.add(
            BuildEventIdUtil.targetCompleted(target.getLabel(), config.getEventId()));
      } else {
        childrenBuilder.add(
            BuildEventIdUtil.targetCompleted(
                target.getLabel(), BuildEventIdUtil.nullConfigurationId()));
      }
    }
    return childrenBuilder.build();
  }

  static BuildEventStreamProtos.TestSize bepTestSize(TestSize size) {
    switch (size) {
      case SMALL:
        return BuildEventStreamProtos.TestSize.SMALL;
      case MEDIUM:
        return BuildEventStreamProtos.TestSize.MEDIUM;
      case LARGE:
        return BuildEventStreamProtos.TestSize.LARGE;
      case ENORMOUS:
        return BuildEventStreamProtos.TestSize.ENORMOUS;
      default:
        return BuildEventStreamProtos.TestSize.UNKNOWN;
    }
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
      builder.setTestSize(bepTestSize(TestSize.getTestSize(target.getAssociatedRule())));
    }
    return GenericBuildEvent.protoChaining(this).setConfigured(builder.build()).build();
  }
}
