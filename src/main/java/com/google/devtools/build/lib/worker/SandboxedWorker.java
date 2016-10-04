// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.worker;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.sandbox.SymlinkedExecRoot;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;

/** A {@link Worker} that runs inside a sandboxed execution root. */
final class SandboxedWorker extends Worker {
  private final Path workDir;
  private final SymlinkedExecRoot symlinkedExecRoot;

  SandboxedWorker(WorkerKey workerKey, int workerId, Path workDir, Path logFile) {
    super(workerKey, workerId, workDir, logFile);
    this.workDir = workDir;
    this.symlinkedExecRoot = new SymlinkedExecRoot(workDir);
  }

  @Override
  void destroy() throws IOException {
    super.destroy();
    if (symlinkedExecRoot != null) {
      FileSystemUtils.deleteTree(workDir);
    }
  }

  @Override
  public void prepareExecution(WorkerKey key) throws IOException {
    symlinkedExecRoot.createFileSystem(
        key.getInputFiles(), key.getOutputFiles(), ImmutableSet.<Path>of());
  }

  @Override
  public void finishExecution(WorkerKey key) throws IOException {
    symlinkedExecRoot.copyOutputs(key.getExecRoot(), key.getOutputFiles());
  }
}
