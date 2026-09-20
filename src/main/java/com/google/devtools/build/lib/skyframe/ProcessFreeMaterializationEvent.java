// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import java.util.Collection;

/** Typed BEP result for one process-free materialization command. */
final class ProcessFreeMaterializationEvent implements BuildEvent {
  private final long selectedActions;
  private final long materializedActions;
  private final long deferredActions;
  private final long visitedActions;
  private final long unresolvedArtifacts;
  private final boolean success;

  ProcessFreeMaterializationEvent(
      long selectedActions,
      long materializedActions,
      long deferredActions,
      long visitedActions,
      long unresolvedArtifacts,
      boolean success) {
    this.selectedActions = selectedActions;
    this.materializedActions = materializedActions;
    this.deferredActions = deferredActions;
    this.visitedActions = visitedActions;
    this.unresolvedArtifacts = unresolvedArtifacts;
    this.success = success;
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventIdUtil.processFreeMaterializationResultId();
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.of();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    BuildEventStreamProtos.ProcessFreeMaterializationResult result =
        BuildEventStreamProtos.ProcessFreeMaterializationResult.newBuilder()
            .setSelectedActionCount(selectedActions)
            .setMaterializedActionCount(materializedActions)
            .setDeferredActionCount(deferredActions)
            .setVisitedActionCount(visitedActions)
            .setUnresolvedArtifactCount(unresolvedArtifacts)
            .setSuccess(success)
            .build();
    return GenericBuildEvent.protoChaining(this)
        .setProcessFreeMaterializationResult(result)
        .build();
  }
}
