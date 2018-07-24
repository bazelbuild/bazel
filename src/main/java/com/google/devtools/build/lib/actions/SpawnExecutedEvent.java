// Copyright 2018 The Bazel Authors. All rights reserved.
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
import java.time.Duration;

/** This event is fired during the build, when a subprocess is executed. */
public class SpawnExecutedEvent {
  private final ActionAnalysisMetadata actionMetadata;
  private final int exitCode;
  private final Duration totalTime;
  private final SpawnResult result;

  public SpawnExecutedEvent(
      ActionAnalysisMetadata actionMetadata,
      int exitCode,
      Duration totalTime,
      SpawnResult result) {
    this.actionMetadata = actionMetadata;
    this.exitCode = exitCode;
    this.totalTime = totalTime;
    this.result = result;
  }

  /** Returns the action. */
  public ActionAnalysisMetadata getActionMetadata() {
    return actionMetadata;
  }

  /** Returns the action exit code. */
  public int getExitCode() {
    return exitCode;
  }

  /** Returns the total time of the subprocess; may include network round trip. */
  public Duration getTotalTime() {
    return totalTime;
  }

  /** Returns the distributor reply. */
  public SpawnResult getSpawnResult() {
    return result;
  }
}
