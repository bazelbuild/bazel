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

package com.google.devtools.build.lib.view;

import com.google.devtools.build.lib.actions.Artifact;

/**
 * This event is fired as soon as a target is either built or fails.
 */
public final class TargetCompleteEvent {

  private final ConfiguredTarget target;
  private final Artifact problem;
  private final Throwable exception;

  /**
   * Construct the event.
   *
   * @param target the target which just completed.
   * @param problem if failed, the artifact associated with the failure.
   * @param exception if failed, the exception which caused the target to fail.
   */
  public TargetCompleteEvent(ConfiguredTarget target,
                             Artifact problem,
                             Throwable exception) {
    this.target = target;
    this.problem = problem;
    this.exception = exception;
  }

  /**
   * Returns the target associated with the event.
   */
  public ConfiguredTarget getTarget() {
    return target;
  }

  /**
   * Determines whether the target has failed or succeeded.
   * A successful target has a null problem and a null exception.
   */
  public boolean failed() {
    return problem != null || exception != null;
  }
}
