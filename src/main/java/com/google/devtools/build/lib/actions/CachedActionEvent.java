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

import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;

/** This event is fired during the build if an action was in the action cache. */
public final class CachedActionEvent implements Postable {

  private final Action action;
  private final long nanoTimeStart;
  private final long nanoTimeFinish;

  /**
   * Create an event for an action that was cached.
   *
   * @param action the cached action
   * @param nanoTimeStart the time when the action was started. This allow us to record more
   *     accurately the time spend by the action, since we execute some code before deciding if we
   *     execute the action or not.
   */
  public CachedActionEvent(Action action, long nanoTimeStart, long nanoTimeFinish) {
    this.action = action;
    this.nanoTimeStart = nanoTimeStart;
    this.nanoTimeFinish = nanoTimeFinish;
  }

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
