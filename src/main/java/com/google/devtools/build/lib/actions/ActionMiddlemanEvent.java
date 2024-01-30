// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;

/**
 * This event is fired during the build, when a middleman action is executed. Middleman actions
 * don't usually do any computation but we need them in the critical path because they depend on
 * other actions.
 */
public final class ActionMiddlemanEvent implements Postable {

  private final Action action;
  private final long nanoTimeStart;
  private final long nanoTimeFinish;

  /**
   * Create an event for action that has been started.
   *
   * @param action the middleman action.
   * @param nanoTimeStart the time when the action was started. This allow us to record more
   *     accurately the time spent by the middleman action, since even for middleman actions we
   *     execute some.
   */
  public ActionMiddlemanEvent(Action action, long nanoTimeStart, long nanoTimeFinish) {
    Preconditions.checkArgument(action.getActionType().isMiddleman(),
        "Only middleman actions should be passed: %s", action);
    this.action = action;
    this.nanoTimeStart = nanoTimeStart;
    this.nanoTimeFinish = nanoTimeFinish;
  }

  /**
   * Returns the associated action.
   */
  public Action getAction() {
    return action;
  }

  public long getNanoTimeStart() {
    return nanoTimeStart;
  }

  public long getNanoTimeFinish() {
    return nanoTimeFinish;
  }
}
