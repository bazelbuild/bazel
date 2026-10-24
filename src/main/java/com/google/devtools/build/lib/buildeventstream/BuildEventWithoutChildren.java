// Copyright 2026 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import java.util.Collection;

/** An event whose children have already been announced by preceding progress events. */
public record BuildEventWithoutChildren(BuildEvent originalEvent) implements BuildEvent {
  @Override
  public BuildEventId getEventId() {
    return originalEvent.getEventId();
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.of();
  }

  @Override
  public Collection<LocalFile> referencedLocalFiles() {
    return originalEvent.referencedLocalFiles();
  }

  @Override
  public Collection<ListenableFuture<String>> remoteUploads() {
    return originalEvent.remoteUploads();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext context)
      throws InterruptedException {
    return originalEvent.asStreamProto(context).toBuilder().clearChildren().build();
  }
}
