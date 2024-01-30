// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.test;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import java.util.Collection;

/**
 * Signal that the coverage actions are finished. Only used as a prerequisite for {@link
 * com.google.devtools.build.lib.analysis.TargetCompleteEvent} in Skymeld mode.
 */
public class CoverageActionFinishedEvent implements BuildEvent {

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext context)
      throws InterruptedException {
    return GenericBuildEvent.protoChaining(this).build();
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventIdUtil.coverageActionsFinished();
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.of();
  }
}
