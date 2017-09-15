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
package com.google.devtools.build.lib.standalone;

import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.exec.AbstractSpawnStrategy;
import com.google.devtools.build.lib.exec.SpawnRunner;

/**
 * Strategy that uses subprocessing to execute a process.
 */
@ExecutionStrategy(name = { "standalone", "local" }, contextType = SpawnActionContext.class)
public class StandaloneSpawnStrategy extends AbstractSpawnStrategy {
  public StandaloneSpawnStrategy(SpawnRunner spawnRunner) {
    super(spawnRunner);
  }

  @Override
  public String toString() {
    return "standalone";
  }
}
