// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool.buildevent;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.ConvenienceSymlink;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import java.util.Collection;

/**
 * This event is fired from ExecutionTool#handleConvenienceSymlinks() whenever convenience symlinks
 * are managed. If the value {@link ConvenienceSymlinksMode.NORMAL}, LOG_ONLY, CLEAN is passed into
 * the build request option {@code --experimental_create_convenience_symlinks}, then this event will
 * be populated with convenience symlink entries. However, if {@link ConvenienceSymlinksMode.IGNORE}
 * is passed, then this will be an empty event.
 */
public final class ConvenienceSymlinksIdentifiedEvent implements BuildEvent {
  private final ImmutableList<ConvenienceSymlink> convenienceSymlinks;

  /** Construct the ConvenienceSymlinksIdentifiedEvent. */
  public ConvenienceSymlinksIdentifiedEvent(ImmutableList<ConvenienceSymlink> convenienceSymlinks) {
    this.convenienceSymlinks = convenienceSymlinks;
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventIdUtil.convenienceSymlinksIdentifiedId();
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.of();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    BuildEventStreamProtos.ConvenienceSymlinksIdentified convenienceSymlinksIdentified =
        BuildEventStreamProtos.ConvenienceSymlinksIdentified.newBuilder()
            .addAllConvenienceSymlinks(convenienceSymlinks)
            .build();
    return GenericBuildEvent.protoChaining(this)
        .setConvenienceSymlinksIdentified(convenienceSymlinksIdentified)
        .build();
  }
}
