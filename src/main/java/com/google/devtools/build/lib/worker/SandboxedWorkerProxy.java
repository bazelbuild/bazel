// Copyright 2021 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.sandbox.SandboxHelpers;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import java.io.IOException;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * A multiplex worker proxy with sandboxing. The multiplexer process runs in {@code workDir}, while
 * each proxy has a fixed subdir where it sets up its files. The subdir is then passed to the worker
 * in {@link WorkRequest#sandbox_dir}. The worker implementation is responsible for reading from and
 * writing to that subdir only.
 */
public class SandboxedWorkerProxy extends WorkerProxy {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /** The sandbox directory for the current request, inside {@code workDir}. */
  private final Path sandboxDir;

  private final PathFragment sandboxName;

  private final TreeDeleter treeDeleter;

  SandboxedWorkerProxy(
      WorkerKey workerKey,
      int workerId,
      Path logFile,
      WorkerMultiplexer workerMultiplexer,
      Path workDir,
      TreeDeleter treeDeleter) {
    super(workerKey, workerId, logFile, workerMultiplexer, workDir);
    sandboxName =
        PathFragment.create(
            Joiner.on(PathFragment.SEPARATOR_CHAR)
                .join(
                    "__sandbox",
                    Integer.toString(workerId),
                    workerKey.getExecRoot().getBaseName()));
    sandboxDir = this.workDir.getRelative(sandboxName);
    this.treeDeleter = treeDeleter;
  }

  @Override
  public boolean isSandboxed() {
    return true;
  }

  @Override
  public void prepareExecution(
      SandboxInputs inputFiles,
      SandboxOutputs outputs,
      Set<PathFragment> workerFiles,
      ImmutableMap<String, String> clientEnv)
      throws IOException, InterruptedException {
    workerMultiplexer.createSandboxedProcess(
        workDir, workerFiles, inputFiles, treeDeleter, clientEnv);

    sandboxDir.createDirectoryAndParents();
    LinkedHashSet<PathFragment> dirsToCreate = new LinkedHashSet<>();
    Set<PathFragment> inputsToCreate = new HashSet<>();

    SandboxHelpers.populateInputsAndDirsToCreate(
        ImmutableSet.of(),
        inputsToCreate,
        dirsToCreate,
        Iterables.concat(inputFiles.getFiles().keySet(), inputFiles.getSymlinks().keySet()),
        outputs);
    SandboxHelpers.cleanExisting(
        sandboxDir.getParentDirectory(),
        inputFiles,
        inputsToCreate,
        dirsToCreate,
        sandboxDir,
        treeDeleter);
    // Finally, create anything that is still missing. This is non-strict only for historical
    // reasons, we haven't seen what would break if we make it strict.
    SandboxHelpers.createDirectories(dirsToCreate, sandboxDir, /* strict= */ false);
    WorkerExecRoot.createInputs(inputsToCreate, inputFiles, sandboxDir);
  }

  /** Send the WorkRequest to multiplexer. */
  @Override
  protected void putRequest(WorkRequest request) throws IOException {
    // Modifying the request on the way out is not great. The alternatives are having the
    // spawn runner ask the worker for the dir or making the spawn runner understand the sandbox,
    // dir structure, neither of which are nice either.
    workerMultiplexer.putRequest(
        request.toBuilder().setSandboxDir(sandboxName.getPathString()).build());
  }

  @Override
  public void finishExecution(Path execRoot, SandboxOutputs outputs)
      throws IOException, InterruptedException {
    super.finishExecution(execRoot, outputs);
    SandboxHelpers.moveOutputs(outputs, sandboxDir, execRoot);
  }

  @Override
  synchronized void destroy() {
    super.destroy();
    try {
      sandboxDir.deleteTree();
    } catch (IOException e) {
      logger.atWarning().withCause(e).log("Caught IOException while deleting workdir.");
    }
  }
}
