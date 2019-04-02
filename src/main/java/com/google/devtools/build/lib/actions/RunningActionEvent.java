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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.events.ExtendedEventHandler.ProgressLike;

/**
 * Notifies that an in-flight action is running.
 *
 * <p>This event should only appear in-between corresponding {@link ActionStartedEvent} and {@link
 * ActionCompletionEvent} events, and should only appear after corresponding {@link
 * AnalyzingActionEvent} and {@link SchedulingActionEvent} events. TODO(jmmv): But this theory is
 * not true today. Investigate.
 */
public class RunningActionEvent implements ProgressLike {

  private final ActionExecutionMetadata action;
  private final String strategy;

  /** Constructs a new event. */
  public RunningActionEvent(ActionExecutionMetadata action, String strategy) {
    this.action = action;
    this.strategy = checkNotNull(strategy, "Strategy names are not optional");
  }

  /** Gets the metadata associated with the action that is running. */
  public ActionExecutionMetadata getActionMetadata() {
    return action;
  }

  /** Gets the name of the strategy used to run the action. */
  public String getStrategy() {
    return strategy;
  }
}
