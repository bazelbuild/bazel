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

package com.google.devtools.build.lib.sandbox;

import com.google.devtools.build.lib.exec.AbstractSpawnStrategy;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.RunfilesTreeUpdater;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.time.Duration;

/** Strategy that uses sandboxing to execute a process. */
public final class LinuxSandboxedStrategy extends AbstractSpawnStrategy {
  LinuxSandboxedStrategy(SpawnRunner spawnRunner, ExecutionOptions executionOptions) {
    super(spawnRunner, executionOptions);
  }

  @Override
  public String toString() {
    return "linux-sandbox";
  }

  /**
   * Creates a sandboxed spawn runner that uses the {@code linux-sandbox} tool.
   *
   * @param cmdEnv the command environment to use
   * @param sandboxBase path to the sandbox base directory
   * @param timeoutKillDelay additional grace period before killing timing out commands
   */
  static LinuxSandboxedSpawnRunner create(
      CommandEnvironment cmdEnv,
      Path sandboxBase,
      Duration timeoutKillDelay,
      TreeDeleter treeDeleter,
      RunfilesTreeUpdater runfilesTreeUpdater)
      throws IOException {
    Path inaccessibleHelperFile = LinuxSandboxUtil.getInaccessibleHelperFile(sandboxBase);
    Path inaccessibleHelperDir = LinuxSandboxUtil.getInaccessibleHelperDir(sandboxBase);

    return new LinuxSandboxedSpawnRunner(
        cmdEnv,
        sandboxBase,
        inaccessibleHelperFile,
        inaccessibleHelperDir,
        timeoutKillDelay,
        treeDeleter,
        runfilesTreeUpdater);
  }
}
