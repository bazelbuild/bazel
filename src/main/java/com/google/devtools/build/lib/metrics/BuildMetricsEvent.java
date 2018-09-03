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
package com.google.devtools.build.lib.metrics;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildMetrics;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithOrderConstraint;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import java.util.Collection;

class BuildMetricsEvent implements BuildEventWithOrderConstraint {
  private final BuildMetrics buildMetrics;

  BuildMetricsEvent(BuildMetrics buildMetrics) {
    this.buildMetrics = buildMetrics;
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventId.buildMetrics();
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.of();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    return GenericBuildEvent.protoChaining(this).setBuildMetrics(buildMetrics).build();
  }

  public BuildMetrics getBuildMetrics() {
    return buildMetrics;
  }

  @Override
  public Collection<BuildEventId> postedAfter() {
    return ImmutableList.of(BuildEventId.buildFinished());
  }
}
