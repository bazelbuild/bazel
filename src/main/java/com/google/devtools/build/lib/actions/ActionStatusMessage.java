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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.events.ExtendedEventHandler;

/**
 * A message used to update in-flight action status. An action's status may change low down in the
 * execution stack (for instance, from running remotely to running locally), so this message can be
 * used to notify any interested parties.
 */
public class ActionStatusMessage implements ExtendedEventHandler.ProgressLike {
  private final ActionExecutionMetadata action;
  private final String message;
  private final String strategy;
  public static final String PREPARING = "Preparing";

  private ActionStatusMessage(ActionExecutionMetadata action, String message, String strategy) {
    this.action = action;
    this.message = message;
    this.strategy = strategy;
  }

  public ActionExecutionMetadata getActionMetadata() {
    return action;
  }

  public String getMessage() {
    return message;
  }

  /**
   * Return the strategy of the action; null if not created by {@link #runningStrategy}.
   */
  public String getStrategy() {
    return strategy;
  }

  /** Creates "Analyzing" status message. */
  public static ActionStatusMessage analysisStrategy(ActionExecutionMetadata action) {
    Preconditions.checkNotNull(action.getOwner());
    return new ActionStatusMessage(action, "Analyzing", null);
  }

  /** Creates "Preparing" status message. */
  public static ActionStatusMessage preparingStrategy(ActionExecutionMetadata action) {
    Preconditions.checkNotNull(action.getOwner());
    return new ActionStatusMessage(action, PREPARING, null);
  }

  /** Creates "Scheduling" status message. */
  public static ActionStatusMessage schedulingStrategy(ActionExecutionMetadata action) {
    Preconditions.checkNotNull(action.getOwner());
    return new ActionStatusMessage(action, "Scheduling", null);
  }

  /** Creates "Running (strategy)" status message. */
  public static ActionStatusMessage runningStrategy(
      ActionExecutionMetadata action, String strategy) {
    Preconditions.checkNotNull(action.getOwner());
    return new ActionStatusMessage(action, String.format("Running (%s)", strategy), strategy);
  }
}
