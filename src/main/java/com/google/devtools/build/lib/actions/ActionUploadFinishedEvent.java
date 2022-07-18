// Copyright 2021 The Bazel Authors. All rights reserved.
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

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;

/** The event that is fired when the file being uploaded by the action is finished. */
@AutoValue
public abstract class ActionUploadFinishedEvent implements Postable {

  public static ActionUploadFinishedEvent create(
      ActionExecutionMetadata action, String resourceId) {
    return new AutoValue_ActionUploadFinishedEvent(action, resourceId);
  }

  /** Returns the associated action. */
  public abstract ActionExecutionMetadata action();

  /**
   * Returns the id that uniquely determines the resource being uploaded among all upload events.
   */
  public abstract String resourceId();
}
