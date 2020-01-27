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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import java.time.Duration;

/**
 * Event signaling the end of the execution phase. Contains statistics about the action cache,
 * the metadata cache and about last file save times.
 */
public class ExecutionFinishedEvent {
  public static final ExecutionFinishedEvent EMPTY =
      new ExecutionFinishedEvent(0, 0, Duration.ZERO, Duration.ZERO);

  private final int outputDirtyFiles;
  private final int outputModifiedFilesDuringPreviousBuild;
  private final Duration sourceDiffCheckingDuration;
  private final Duration outputTreeDiffCheckingDuration;

  ExecutionFinishedEvent(
      int outputDirtyFiles,
      int outputModifiedFilesDuringPreviousBuild,
      Duration sourceDiffCheckingDuration,
      Duration outputTreeDiffCheckingDuration) {
    this.outputDirtyFiles = outputDirtyFiles;
    this.outputModifiedFilesDuringPreviousBuild = outputModifiedFilesDuringPreviousBuild;
    this.sourceDiffCheckingDuration = Preconditions.checkNotNull(sourceDiffCheckingDuration);
    this.outputTreeDiffCheckingDuration =
        Preconditions.checkNotNull(outputTreeDiffCheckingDuration);
  }

  public int getOutputDirtyFiles() {
    return outputDirtyFiles;
  }

  public int getOutputModifiedFilesDuringPreviousBuild() {
    return outputModifiedFilesDuringPreviousBuild;
  }

  public Duration getSourceDiffCheckingDuration() {
    return sourceDiffCheckingDuration;
  }

  public Duration getOutputTreeDiffCheckingDuration() {
    return outputTreeDiffCheckingDuration;
  }
}
