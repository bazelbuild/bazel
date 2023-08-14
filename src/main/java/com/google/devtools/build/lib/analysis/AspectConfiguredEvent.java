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

import static com.google.devtools.build.lib.analysis.config.BuildConfigurationValue.buildEvent;
import static com.google.devtools.build.lib.analysis.config.BuildConfigurationValue.configurationId;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithConfiguration;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.cmdline.Label;
import javax.annotation.Nullable;

/** Event reporting about the configurations associated with a given apect for a target */
public class AspectConfiguredEvent implements BuildEventWithConfiguration {
  private final Label target;
  private final String aspectClassName;
  private final String aspectDescription;
  @Nullable private final BuildConfigurationValue configuration;

  public AspectConfiguredEvent(
      Label target,
      String aspectClassName,
      String aspectDescription,
      @Nullable BuildConfigurationValue configuration) {
    this.target = target;
    this.aspectClassName = aspectClassName;
    this.aspectDescription = aspectDescription;
    this.configuration = configuration;
  }

  @Override
  public ImmutableList<BuildEvent> getConfigurations() {
    return ImmutableList.of(buildEvent(configuration));
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventIdUtil.aspectConfigured(target, aspectClassName);
  }

  @Override
  public ImmutableList<BuildEventId> getChildrenEvents() {
    return ImmutableList.of(
        BuildEventIdUtil.aspectCompleted(
            target, configurationId(configuration), aspectDescription));
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    BuildEventStreamProtos.TargetConfigured.Builder builder =
        BuildEventStreamProtos.TargetConfigured.newBuilder();
    return GenericBuildEvent.protoChaining(this).setConfigured(builder.build()).build();
  }
}
