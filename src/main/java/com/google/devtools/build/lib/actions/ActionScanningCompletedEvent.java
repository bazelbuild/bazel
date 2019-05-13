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
package com.google.devtools.build.lib.actions;

import com.google.common.base.MoreObjects;
import com.google.devtools.build.lib.events.ExtendedEventHandler.ProgressLike;

/**
 * An event that is fired when a input-discovering secondary shared action completes. Such actions
 * are not actually executed, so there is no corresponding {@link ActionCompletionEvent}.
 */
public final class ActionScanningCompletedEvent implements ProgressLike {
  private final Action action;
  private final ActionLookupData actionLookupData;

  public ActionScanningCompletedEvent(Action action, ActionLookupData actionLookupData) {
    this.action = action;
    this.actionLookupData = actionLookupData;
  }

  /** Returns the action. */
  public Action getAction() {
    return action;
  }

  public ActionLookupData getActionLookupData() {
    return actionLookupData;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper("ActionScanningCompletedEvent")
        .add("action", action)
        .add("actionLookupData", actionLookupData)
        .toString();
  }
}
