// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import java.util.Collection;

/** Wrapper class for a build event marking it as the final event in the protocol. */
public class LastBuildEvent implements BuildEvent {
  private final BuildEvent event;

  public LastBuildEvent(BuildEvent event) {
    this.event = event;
  }

  @Override
  public BuildEventId getEventId() {
    return event.getEventId();
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return event.getChildrenEvents();
  }

  @Override
  public Collection<LocalFile> referencedLocalFiles() {
    return event.referencedLocalFiles();
  }

  @Override
  public Collection<ListenableFuture<String>> remoteUploads() {
    return event.remoteUploads();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters)
      throws InterruptedException {
    return BuildEventStreamProtos.BuildEvent.newBuilder(event.asStreamProto(converters))
        .setLastMessage(true)
        .build();
  }
}
