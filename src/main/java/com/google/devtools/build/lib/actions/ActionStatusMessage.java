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

/**
 * A message used to update in-flight action status. An action's status may change low down in the
 * execution stack (for instance, from running remotely to running locally), so this message can be
 * used to notify any interested parties.
 */
public class ActionStatusMessage {
  private final ActionMetadata action;
  private final String message;
  public static final String PREPARING = "Preparing";

  private ActionStatusMessage(ActionMetadata action, String message) {
    this.action = action;
    this.message = message;
  }

  public ActionMetadata getActionMetadata() {
    return action;
  }

  public String getMessage() {
    return message;
  }

  /** Creates "Analyzing" status message. */
  public static ActionStatusMessage analysisStrategy(ActionMetadata action) {
    return new ActionStatusMessage(action, "Analyzing");
  }

  /** Creates "Preparing" status message. */
  public static ActionStatusMessage preparingStrategy(ActionMetadata action) {
    return new ActionStatusMessage(action, PREPARING);
  }

  /** Creates "Scheduling" status message. */
  public static ActionStatusMessage schedulingStrategy(ActionMetadata action) {
    return new ActionStatusMessage(action, "Scheduling");
  }

  /** Creates "Running (strategy)" status message. */
  public static ActionStatusMessage runningStrategy(ActionMetadata action, String strategy) {
    return new ActionStatusMessage(action, String.format("Running (%s)", strategy));
  }
}
