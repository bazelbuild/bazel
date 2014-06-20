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
 * An event class to notify listeners of metrics about the action graph.
 */
public class ActionGraphInfoEvent {

  private final int numActions;

  public ActionGraphInfoEvent(int numActions) {
    this.numActions = numActions;
  }

  /**
   * @return the total number of non-middleman actions in the action graph.
   */
  public int getNumRealActions() {
    return numActions;
  }
}
