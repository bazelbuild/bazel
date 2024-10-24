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
package com.google.devtools.build.lib.worker;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.sandbox.SandboxHelpers;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxContents;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

/** Creates and manages the contents of a working directory of a persistent worker. */
final class WorkerExecRoot {
  private final Path workDir;
  private final List<PathFragment> extraDirs;

  private final boolean useInMemoryTracking;
  @Nullable private SandboxContents sandboxContents;
  private long sandboxContentsTimestamp;

  /**
   * Creates a new WorkerExecRoot.
   *
   * @param workDir The directory (workspace dir) that the worker will be executing in.
   * @param extraDirs Directories that must survive sandbox cleanup, e.g. for things that are
   *     bind-mounted.
   */
  public WorkerExecRoot(Path workDir, List<PathFragment> extraDirs, boolean useInMemoryTracking) {
    this.workDir = workDir;
    this.extraDirs = extraDirs;
    this.useInMemoryTracking = useInMemoryTracking;
  }

  public void createFileSystem(
      Set<PathFragment> workerFiles,
      SandboxInputs inputs,
      SandboxOutputs outputs,
      TreeDeleter treeDeleter)
      throws IOException, InterruptedException {
    workDir.createDirectoryAndParents();

    // First compute all the inputs and directories that we need. This is based only on
    // `workerFiles`, `inputs` and `outputs` and won't do any I/O.
    Set<PathFragment> inputsToCreate = new LinkedHashSet<>();
    LinkedHashSet<PathFragment> dirsToCreate = new LinkedHashSet<>(extraDirs);
    SandboxHelpers.populateInputsAndDirsToCreate(
        ImmutableSet.of(),
        inputsToCreate,
        dirsToCreate,
        Iterables.concat(workerFiles, inputs.getFiles().keySet(), inputs.getSymlinks().keySet()),
        outputs);

    // If we have information about the previous contents of the sandbox, update it to reflect
    // filesystem changes that have happened in the interim, to speed up the cleanup process below.
    // TODO(tjgq): Consider doing this asynchronously in between worker invocations.
    if (sandboxContents != null) {
      SandboxHelpers.updateContentMap(
          workDir.getParentDirectory(), sandboxContentsTimestamp, sandboxContents);
    }

    // Then do a full traversal of the parent directory of `workDir`. This will use what we computed
    // above, delete anything unnecessary and update `inputsToCreate`/`dirsToCreate` if something is
    // can be left without changes (e.g., a symlink that already points to the right destination).
    // We're traversing from workDir's parent directory because external repositories can now be
    // symlinked as siblings of workDir when --experimental_sibling_repository_layout is in effect.
    SandboxHelpers.cleanExisting(
        workDir.getParentDirectory(),
        inputs,
        inputsToCreate,
        dirsToCreate,
        workDir,
        treeDeleter,
        sandboxContents);

    // Finally, create anything that is still missing. This is non-strict only for historical
    // reasons, we haven't seen what would break if we make it strict.
    SandboxHelpers.createDirectories(dirsToCreate, workDir, /* strict= */ false);
    createInputs(inputsToCreate, inputs, workDir);

    // Track the sandbox contents in memory. This makes the cleanup faster in subsequent runs.
    if (useInMemoryTracking) {
      sandboxContents = SandboxHelpers.createContentMap(workDir, inputs, outputs);
      sandboxContentsTimestamp = System.currentTimeMillis();
    }
  }

  static void createInputs(Iterable<PathFragment> inputsToCreate, SandboxInputs inputs, Path dir)
      throws IOException, InterruptedException {
    for (PathFragment fragment : inputsToCreate) {
      if (Thread.interrupted()) {
        throw new InterruptedException();
      }
      Path key = dir.getRelative(fragment);
      if (inputs.getFiles().containsKey(fragment)) {
        Path fileDest = inputs.getFiles().get(fragment);
        if (fileDest != null) {
          key.createSymbolicLink(fileDest);
        } else {
          FileSystemUtils.createEmptyFile(key);
        }
      } else if (inputs.getSymlinks().containsKey(fragment)) {
        PathFragment symlinkDest = inputs.getSymlinks().get(fragment);
        if (symlinkDest != null) {
          key.createSymbolicLink(symlinkDest);
        }
      }
    }
  }

  public void copyOutputs(Path execRoot, SandboxOutputs outputs) throws IOException {
    SandboxHelpers.moveOutputs(outputs, workDir, execRoot);
  }
}
