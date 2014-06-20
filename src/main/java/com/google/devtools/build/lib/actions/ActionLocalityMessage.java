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
 * A message used to update in-flight action locality. This is useful in cases where the locality is
 * not known statically. For example, local fallback could use this to update an action that would
 * otherwise be run remotely.
 */
public class ActionLocalityMessage {
  private final ActionMetadata action;
  private final String message;
  private final String strategy;

  public ActionLocalityMessage(ActionMetadata action, String strategy, String message) {
    this.strategy = strategy;
    this.action = action;
    this.message = message;
  }

  public ActionMetadata getActionMetadata() {
    return action;
  }

  public String getStrategy() {
    return strategy;
  }

  public String getMessage() {
    return message;
  }

  /*
   * Creates "analysis" status message.
   */
  public static ActionLocalityMessage analysisStrategy(ActionMetadata action) {
    return new ActionLocalityMessage(action, "analysis", "Analyzing");
  }
}
