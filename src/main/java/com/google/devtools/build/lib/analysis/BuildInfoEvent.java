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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithOrderConstraint;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import java.util.Collection;
import java.util.Map;

/** This event is fired once build info data is available. */
public final class BuildInfoEvent
    implements BuildEventWithOrderConstraint, ExtendedEventHandler.ProgressLike {
  private final Map<String, String> buildInfoMap;

  /**
   * Construct the event from a map.
   */
  public BuildInfoEvent(Map<String, String> buildInfo) {
    buildInfoMap = ImmutableMap.copyOf(buildInfo);
  }

  /**
   * Return immutable map populated with build info key/value pairs.
   */
  public Map<String, String> getBuildInfoMap() {
    return buildInfoMap;
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventIdUtil.workspaceStatusId();
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.of();
  }

  @Override
  public Collection<BuildEventId> postedAfter() {
    return ImmutableList.of(BuildEventIdUtil.buildStartedId());
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    BuildEventStreamProtos.WorkspaceStatus.Builder status =
        BuildEventStreamProtos.WorkspaceStatus.newBuilder();
    for (Map.Entry<String, String> entry : getBuildInfoMap().entrySet()) {
      status.addItem(
          BuildEventStreamProtos.WorkspaceStatus.Item.newBuilder()
              .setKey(entry.getKey())
              .setValue(entry.getValue())
              .build());
    }
    return GenericBuildEvent.protoChaining(this).setWorkspaceStatus(status.build()).build();
  }
}
