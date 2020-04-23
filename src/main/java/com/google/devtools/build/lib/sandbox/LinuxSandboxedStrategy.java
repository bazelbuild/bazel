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
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.time.Duration;
import javax.annotation.Nullable;

/** Strategy that uses sandboxing to execute a process. */
public final class LinuxSandboxedStrategy extends AbstractSpawnStrategy {
  LinuxSandboxedStrategy(Path execRoot, SpawnRunner spawnRunner) {
    super(execRoot, spawnRunner);
  }

  @Override
  public String toString() {
    return "linux-sandbox";
  }

  /**
   * Creates a sandboxed spawn runner that uses the {@code linux-sandbox} tool.
   *
   * @param helpers common tools and state across all spawns during sandboxed execution
   * @param cmdEnv the command environment to use
   * @param sandboxBase path to the sandbox base directory
   * @param timeoutKillDelay additional grace period before killing timing out commands
   * @param sandboxfsProcess instance of the sandboxfs process to use; may be null for none, in
   *     which case the runner uses a symlinked sandbox
   * @param sandboxfsMapSymlinkTargets map the targets of symlinks within the sandbox if true
   */
  static LinuxSandboxedSpawnRunner create(
      SandboxHelpers helpers,
      CommandEnvironment cmdEnv,
      Path sandboxBase,
      Duration timeoutKillDelay,
      @Nullable SandboxfsProcess sandboxfsProcess,
      boolean sandboxfsMapSymlinkTargets,
      TreeDeleter treeDeleter)
      throws IOException {
    Path inaccessibleHelperFile = sandboxBase.getRelative("inaccessibleHelperFile");
    FileSystemUtils.touchFile(inaccessibleHelperFile);
    inaccessibleHelperFile.setReadable(false);
    inaccessibleHelperFile.setWritable(false);
    inaccessibleHelperFile.setExecutable(false);

    Path inaccessibleHelperDir = sandboxBase.getRelative("inaccessibleHelperDir");
    inaccessibleHelperDir.createDirectory();
    inaccessibleHelperDir.setReadable(false);
    inaccessibleHelperDir.setWritable(false);
    inaccessibleHelperDir.setExecutable(false);

    return new LinuxSandboxedSpawnRunner(
        helpers,
        cmdEnv,
        sandboxBase,
        inaccessibleHelperFile,
        inaccessibleHelperDir,
        timeoutKillDelay,
        sandboxfsProcess,
        sandboxfsMapSymlinkTargets,
        treeDeleter);
  }
}
