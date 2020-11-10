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

import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Set;

/** A {@link SingleplexWorker} that runs inside a sandboxed execution root. */
final class SandboxedWorker extends SingleplexWorker {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private WorkerExecRoot workerExecRoot;

  SandboxedWorker(WorkerKey workerKey, int workerId, Path workDir, Path logFile) {
    super(workerKey, workerId, workDir, logFile);
  }

  @Override
  public void prepareExecution(
      SandboxInputs inputFiles, SandboxOutputs outputs, Set<PathFragment> workerFiles)
      throws IOException {
    // Note that workerExecRoot isn't necessarily null at this point, so we can't do a Preconditions
    // check for it: If a WorkerSpawnStrategy gets interrupted, finishExecution is not guaranteed to
    // be called.
    workerExecRoot = new WorkerExecRoot(workDir, inputFiles, outputs, workerFiles);
    workerExecRoot.createFileSystem();

    super.prepareExecution(inputFiles, outputs, workerFiles);
  }

  @Override
  public void finishExecution(Path execRoot) throws IOException {
    super.finishExecution(execRoot);

    workerExecRoot.copyOutputs(execRoot);
    workerExecRoot = null;
  }

  @Override
  void destroy() {
    super.destroy();
    try {
      workDir.deleteTree();
    } catch (IOException e) {
      logger.atWarning().withCause(e).log("Caught IOException while deleting workdir.");
    }
  }
}
