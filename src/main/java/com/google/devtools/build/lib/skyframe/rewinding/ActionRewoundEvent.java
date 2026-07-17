// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;

/**
 * This event is fired during the build for an action that failed because of lost inputs, and that
 * will try to recover through rewinding (see {@link ActionRewindStrategy}).
 */
public final class ActionRewoundEvent implements Postable {

  private final long relativeActionStartTimeNanos;
  private final long relativeActionFinishTimeNanos;
  private final Action failedRewoundAction;

  /**
   * Create an event for action that that failed because of lost inputs, and that will try to
   * recover through rewinding.
   *
   * @param relativeActionStartTime a nanotime taken before action execution began
   * @param failedRewoundAction the failed action.
   */
  public ActionRewoundEvent(
      long relativeActionStartTimeNanos,
      long relativeActionFinishTimeNanos,
      Action failedRewoundAction) {
    this.relativeActionStartTimeNanos = relativeActionStartTimeNanos;
    this.relativeActionFinishTimeNanos = relativeActionFinishTimeNanos;
    this.failedRewoundAction = failedRewoundAction;
  }

  /** Returns a nanotime taken before action execution began. */
  public long getRelativeActionStartTimeNanos() {
    return relativeActionStartTimeNanos;
  }

  /** Returns a nanotime taken after action execution finished. */
  public long getRelativeActionFinishTimeNanos() {
    return relativeActionFinishTimeNanos;
  }

  /** Returns the associated action. */
  public Action getFailedRewoundAction() {
    return failedRewoundAction;
  }
}
