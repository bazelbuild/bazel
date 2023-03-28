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

package com.google.devtools.build.lib.sandbox;

import com.google.devtools.build.lib.exec.AbstractSpawnStrategy;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.vfs.Path;

/** Strategy that uses sandboxing to execute a process. */
public final class WindowsSandboxedStrategy extends AbstractSpawnStrategy {
  WindowsSandboxedStrategy(
      Path execRoot, SpawnRunner spawnRunner, ExecutionOptions executionOptions) {
    super(execRoot, spawnRunner, executionOptions);
  }

  @Override
  public String toString() {
    return "windows-sandbox";
  }
}
