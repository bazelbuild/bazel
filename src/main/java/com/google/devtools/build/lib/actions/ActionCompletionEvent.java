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
 * An event that is fired after an action completes (either successfully or not).
 */
public final class ActionCompletionEvent {

  private final long relativeActionStartTime;
  private final Action action;

  public ActionCompletionEvent(long relativeActionStartTime, Action action) {
    this.relativeActionStartTime = relativeActionStartTime;
    this.action = action;
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
}
