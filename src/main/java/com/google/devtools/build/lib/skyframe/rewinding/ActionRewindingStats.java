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

package com.google.devtools.build.lib.skyframe.rewinding;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.skyframe.proto.ActionRewind.ActionRewindEvent;

/** Event that encapsulates data about action rewinding during a build. */
public final class ActionRewindingStats implements ExtendedEventHandler.Postable {

  private final int lostInputsCount;
  private final int lostOutputsCount;
  private final ImmutableList<ActionRewindEvent> actionRewindEvents;

  ActionRewindingStats(
      int lostInputsCount,
      int lostOutputsCount,
      ImmutableList<ActionRewindEvent> actionRewindEvents) {
    this.lostInputsCount = lostInputsCount;
    this.lostOutputsCount = lostOutputsCount;
    this.actionRewindEvents = checkNotNull(actionRewindEvents);
  }

  public int lostInputsCount() {
    return lostInputsCount;
  }

  public int lostOutputsCount() {
    return lostOutputsCount;
  }

  public ImmutableList<ActionRewindEvent> actionRewindEvents() {
    return actionRewindEvents;
  }
}
