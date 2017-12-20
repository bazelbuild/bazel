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

package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;

/**
 * This event is fired during the build, when it becomes known that the analysis
 * of a target cannot be completed because of an error in one of its
 * dependencies.
 */
public class AnalysisFailureEvent {
  private final ConfiguredTargetKey failedTarget;
  private final Label failureReason;

  public AnalysisFailureEvent(ConfiguredTargetKey failedTarget, Label failureReason) {
    this.failedTarget = failedTarget;
    this.failureReason = failureReason;
  }

  public ConfiguredTargetKey getFailedTarget() {
    return failedTarget;
  }

  public Label getFailureReason() {
    return failureReason;
  }
}
