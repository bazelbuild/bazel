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

import com.google.devtools.build.lib.events.ExtendedEventHandler;

/** This event is fired during the build, when a subprocess is executed. */
public class SpawnExecutedEvent implements ExtendedEventHandler.ProgressLike {
  private final Spawn spawn;
  private final SpawnResult result;

  public SpawnExecutedEvent(
      Spawn spawn,
      SpawnResult result) {
    this.spawn = spawn;
    this.result = result;
  }

  /** Returns the Spawn. */
  public Spawn getSpawn() {
    return spawn;
  }

  /** Returns the action. */
  public ActionAnalysisMetadata getActionMetadata() {
    return spawn.getResourceOwner();
  }

  /** Returns the action exit code. */
  public int getExitCode() {
    return result.exitCode();
  }

  /** Returns the distributor reply. */
  public SpawnResult getSpawnResult() {
    return result;
  }
}
