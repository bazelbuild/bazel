// Copyright 2014 Google Inc. All rights reserved.
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

/**
 * This event is fired when an action fails, either in execution or when checking the action's
 * outputs. (In that sense, it is a superset of failures for which an
 * {@link ActionNotExecutedEvent} will be created).
 */
public final class ActionFailedEvent {
  private final Action action;

  /**
   * Construct the event.
   *
   * @param action The action that failed.
   */
  public ActionFailedEvent(Action action) {
    this.action = action;
  }

  /**
   * The action which failed.
   */
  public Action getAction() {
    return action;
  }
}
