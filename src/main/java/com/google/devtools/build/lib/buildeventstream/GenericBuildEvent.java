// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.buildeventstream;

import java.util.Collection;

/**
 * Class for a generic {@link BuildEvent}.
 *
 * <p>This class implements a basic {@link BuildEvent}. The main purpose of this class is to provide
 * the common infrastructure for the infrastructural events.
 */
public class GenericBuildEvent implements BuildEvent {
  private final BuildEventId id;
  private final Collection<BuildEventId> children;

  public GenericBuildEvent(BuildEventId id, Collection<BuildEventId> children) {
    this.id = id;
    this.children = children;
  }

  @Override
  public BuildEventId getEventId() {
    return id;
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return children;
  }

  public static BuildEventStreamProtos.BuildEvent.Builder protoChaining(ChainableEvent event) {
    BuildEventStreamProtos.BuildEvent.Builder builder =
        BuildEventStreamProtos.BuildEvent.newBuilder();
    builder.setId(event.getEventId().asStreamProto());
    for (BuildEventId childId : event.getChildrenEvents()) {
      builder.addChildren(childId.asStreamProto());
    }
    return builder;
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    return protoChaining(this).build();
  }
}
