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
import com.google.devtools.build.lib.cmdline.Label;
import java.util.Collection;

/** Event reporting about the configurations associated with a given apect for a target */
public class AspectConfiguredEvent implements BuildEventWithConfiguration {
  private final Label target;
  private final String aspect;
  private final Collection<BuildConfiguration> configurations;

  AspectConfiguredEvent(
      Label target, String aspect, Collection<BuildConfiguration> configurations) {
    this.configurations = configurations;
    this.target = target;
    this.aspect = aspect;
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
    return BuildEventIdUtil.aspectConfigured(target, aspect);
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    ImmutableList.Builder<BuildEventId> childrenBuilder = ImmutableList.builder();
    for (BuildConfiguration config : configurations) {
      if (config != null) {
        childrenBuilder.add(BuildEventIdUtil.targetCompleted(target, config.getEventId()));
      } else {
        childrenBuilder.add(
            BuildEventIdUtil.targetCompleted(target, BuildEventIdUtil.nullConfigurationId()));
      }
    }
    return childrenBuilder.build();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    BuildEventStreamProtos.TargetConfigured.Builder builder =
        BuildEventStreamProtos.TargetConfigured.newBuilder();
    return GenericBuildEvent.protoChaining(this).setConfigured(builder.build()).build();
  }
}
