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
package com.google.devtools.build.lib.actions;

import com.google.common.base.MoreObjects;
import com.google.devtools.build.lib.events.ExtendedEventHandler.ProgressLike;

/**
 * An event that is fired after an action completes (either successfully or not).
 */
public final class ActionCompletionEvent implements ProgressLike {

  private final long relativeActionStartTime;
  private final Action action;
  private final ActionLookupData actionLookupData;

  public ActionCompletionEvent(
      long relativeActionStartTime, Action action, ActionLookupData actionLookupData) {
    this.relativeActionStartTime = relativeActionStartTime;
    this.action = action;
    this.actionLookupData = actionLookupData;
  }

  /**
   * Returns the action.
   */
  public Action getAction() {
    return action;
  }

  public long getRelativeActionStartTime() {
    return relativeActionStartTime;
  }

  public ActionLookupData getActionLookupData() {
    return actionLookupData;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper("ActionCompletionEvent")
        .add("relativeActionStartTime", relativeActionStartTime)
        .add("action", action)
        .add("actionLookupData", actionLookupData)
        .toString();
  }
}
