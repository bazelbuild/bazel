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

import com.google.devtools.build.lib.events.ExtendedEventHandler.ProgressLike;

/**
 * Notifies that an in-flight action is being scheduled.
 *
 * <p>This event should only appear in-between corresponding {@link ActionStartedEvent} and {@link
 * ActionCompletionEvent} events, and should only appear after a corresponding {@link
 * AnalyzingActionEvent}. TODO(jmmv): But this theory is not true today. Investigate.
 */
public class SchedulingActionEvent implements ProgressLike {

  private final ActionExecutionMetadata action;
  private final String strategy;

  /** Constructs a new event. */
  public SchedulingActionEvent(ActionExecutionMetadata action, String strategy) {
    this.action = action;
    this.strategy = strategy;
  }

  /** Gets the metadata associated with the action being scheduled. */
  public ActionExecutionMetadata getActionMetadata() {
    return action;
  }

  /** Gets the name of the strategy on which the action is scheduling. */
  public String getStrategy() {
    return strategy;
  }
}
