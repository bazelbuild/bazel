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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.skyframe.ActionRewindStrategy.RewindPlanStats;
import com.google.devtools.build.lib.skyframe.proto.ActionRewind.ActionDescription;
import com.google.devtools.build.lib.skyframe.proto.ActionRewind.ActionRewindEvent;
import com.google.devtools.build.lib.skyframe.proto.ActionRewind.LostInput;

/** Event that encapsulates data about action rewinding during a build. */
public class ActionRewindingStats implements ExtendedEventHandler.Postable {
  private final int lostInputsCount;
  private final ImmutableList<ActionRewindEvent> actionRewindEvents;

  ActionRewindingStats(int lostInputsCount, ImmutableList<ActionRewindEvent> actionRewindEvents) {
    this.lostInputsCount = lostInputsCount;
    this.actionRewindEvents = actionRewindEvents;
  }

  public int lostInputsCount() {
    return lostInputsCount;
  }

  public ImmutableList<ActionRewindEvent> actionRewindEvents() {
    return actionRewindEvents;
  }

  public static ActionRewindEvent toActionRewindEventProto(RewindPlanStats rewindPlanStats) {
    ActionOwner failedActionOwner = rewindPlanStats.failedAction().getOwner();
    return ActionRewindEvent.newBuilder()
        .setActionDescription(
            ActionDescription.newBuilder()
                .setType(rewindPlanStats.failedAction().getMnemonic())
                .setRuleLabel(
                    failedActionOwner != null ? failedActionOwner.getLabel().toString() : null)
                .build())
        .addAllLostInputs(
            rewindPlanStats.sampleLostInputRecords().stream()
                .map(
                    lostInputRecord ->
                        LostInput.newBuilder()
                            .setPath(lostInputRecord.lostInputPath())
                            .setDigest(lostInputRecord.lostInputDigest())
                            .build())
                .collect(toImmutableList()))
        .setTotalLostInputsCount(rewindPlanStats.lostInputRecordsCount())
        .setInvalidatedNodesCount(rewindPlanStats.invalidatedNodesCount())
        .build();
  }
}
