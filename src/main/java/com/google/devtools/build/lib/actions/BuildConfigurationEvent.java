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

package com.google.devtools.build.lib.actions;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.util.Collection;
import java.util.Objects;

/**
 * Encapsulation of {@link BuildEvent} info associated with a {@link
 * com.google.devtools.build.lib.analysis.config.BuildConfiguration}.
 */
@AutoCodec
public class BuildConfigurationEvent implements BuildEvent {

  private final BuildEventId eventId;
  private final BuildEventStreamProtos.BuildEvent eventProto;

  public BuildConfigurationEvent(
      BuildEventId eventId, BuildEventStreamProtos.BuildEvent eventProto) {
    this.eventId = eventId;
    this.eventProto = eventProto;
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext unusedConverters) {
    return eventProto;
  }

  @Override
  public BuildEventId getEventId() {
    return eventId;
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.of();
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof BuildConfigurationEvent)) {
      return false;
    }
    BuildConfigurationEvent that = (BuildConfigurationEvent) other;
    return Objects.equals(eventId, that.eventId) && Objects.equals(eventProto, that.eventProto);
  }

  @Override
  public int hashCode() {
    return Objects.hash(eventId, eventProto);
  }
}
