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

package com.google.devtools.build.lib.bazel.rules;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventConverters;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithConfiguration;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.Collection;

/** Class reporting that a configured target will not be built due an error in the analysis phase */
public class VisibilityErrorEvent implements BuildEventWithConfiguration {
  BuildConfiguration configuration;
  Label label;
  String errorMessage;

  public VisibilityErrorEvent(BuildConfiguration configuration, Label label, String errorMessage) {
    this.configuration = configuration;
    this.label = label;
    this.errorMessage = errorMessage;
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventId.targetCompleted(label, configuration.getEventId());
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.<BuildEventId>of();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventConverters converters) {
    return GenericBuildEvent.protoChaining(this)
        .setAnalysisFailed(
            BuildEventStreamProtos.AnalysisFailure.newBuilder().setDetails(errorMessage).build())
        .build();
  }

  @Override
  public Collection<BuildEvent> getConfigurations() {
    return ImmutableList.<BuildEvent>of(configuration);
  }
}
