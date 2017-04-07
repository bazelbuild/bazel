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

package com.google.devtools.build.lib.buildtool.buildevent;

import static com.google.devtools.build.lib.util.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventConverters;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildFinished;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.buildtool.BuildResult;
import java.util.Collection;

/**
 * This event is fired from BuildTool#stopRequest().
 *
 * <p>This class also implements the {@link BuildFinished} event of the build event protocol (BEP).
 */
public final class BuildCompleteEvent implements BuildEvent {
  private final BuildResult result;

  /**
   * Construct the BuildCompleteEvent.
   */
  public BuildCompleteEvent(BuildResult result) {
    this.result = checkNotNull(result);
  }

  /**
   * @return the build summary
   */
  public BuildResult getResult() {
    return result;
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventId.buildFinished();
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.of();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventConverters converters) {
    BuildFinished finished =
        BuildFinished.newBuilder()
            .setOverallSuccess(result.getSuccess())
            .setFinishTimeMillis(result.getStopTime())
            .build();
    return GenericBuildEvent.protoChaining(this).setFinished(finished).build();
  }
}
