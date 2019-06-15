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
 * Notifies that an in-flight action has started being scanned for discovered inputs.
 *
 * <p>This phase ends when an {@link StoppedScanningActionEvent} is posted for this action.
 */
public class ScanningActionEvent implements ProgressLike {

  private final ActionExecutionMetadata action;

  /** Constructs a new event. */
  public ScanningActionEvent(ActionExecutionMetadata action) {
    this.action = action;
  }

  /** Gets the metadata associated with the action being analyzed. */
  public ActionExecutionMetadata getActionMetadata() {
    return action;
  }
}
