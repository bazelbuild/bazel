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
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;

/** An event that is fired after an action completes (either successfully or not). */
public final class ActionCompletionEvent implements Postable {

  private final long relativeActionStartTimeNanos;
  private final long finishTimeNanos;
  private final Action action;
  private final ActionLookupData actionLookupData;

  public ActionCompletionEvent(
      long relativeActionStartTimeNanos,
      long finishTimeNanos,
      Action action,
      ActionLookupData actionLookupData) {
    this.relativeActionStartTimeNanos = relativeActionStartTimeNanos;
    this.finishTimeNanos = finishTimeNanos;
    this.action = action;
    this.actionLookupData = actionLookupData;
  }

  /**
   * Returns the action.
   */
  public Action getAction() {
    return action;
  }

  public long getRelativeActionStartTimeNanos() {
    return relativeActionStartTimeNanos;
  }

  public long getFinishTimeNanos() {
    return finishTimeNanos;
  }

  public ActionLookupData getActionLookupData() {
    return actionLookupData;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper("ActionCompletionEvent")
        .add("relativeActionStartTimeNanos", relativeActionStartTimeNanos)
        .add("finishTimeNanos", finishTimeNanos)
        .add("action", action)
        .add("actionLookupData", actionLookupData)
        .toString();
  }
}
