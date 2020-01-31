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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.sandbox.SymlinkedSandboxedSpawn;
import com.google.devtools.build.lib.sandbox.SynchronousTreeDeleter;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.util.Map;
import java.util.Set;

/** Creates and manages the contents of a working directory of a persistent worker. */
final class WorkerExecRoot extends SymlinkedSandboxedSpawn {
  private final Path workDir;
  private final Set<PathFragment> workerFiles;

  public WorkerExecRoot(
      Path workDir, SandboxInputs inputs, SandboxOutputs outputs, Set<PathFragment> workerFiles) {
    super(
        workDir,
        workDir,
        ImmutableList.of(),
        ImmutableMap.of(),
        inputs,
        outputs,
        ImmutableSet.of(),
        new SynchronousTreeDeleter(),
        /*statisticsPath=*/ null);
    this.workDir = workDir;
    this.workerFiles = workerFiles;
  }

  @Override
  public void createFileSystem() throws IOException {
    workDir.createDirectoryAndParents();
    deleteExceptAllowedFiles(workDir, workerFiles);
    super.createFileSystem();
  }

  private void deleteExceptAllowedFiles(Path root, Set<PathFragment> allowedFiles)
      throws IOException {
    for (Path p : root.getDirectoryEntries()) {
      FileStatus stat = p.stat(Symlinks.NOFOLLOW);
      if (!stat.isDirectory()) {
        if (!allowedFiles.contains(p.relativeTo(workDir))) {
          p.delete();
        }
      } else {
        deleteExceptAllowedFiles(p, allowedFiles);
        if (p.readdir(Symlinks.NOFOLLOW).isEmpty()) {
          p.delete();
        }
      }
    }
  }

  @Override
  protected void createInputs(SandboxInputs inputs) throws IOException {
    // All input files are relative to the execroot.
    for (Map.Entry<PathFragment, Path> entry : inputs.getFiles().entrySet()) {
      Path key = workDir.getRelative(entry.getKey());
      FileStatus keyStat = key.statNullable(Symlinks.NOFOLLOW);
      if (keyStat != null) {
        if (keyStat.isSymbolicLink()
            && entry.getValue() != null
            && key.readSymbolicLink().equals(entry.getValue().asFragment())) {
          continue;
        }
        key.delete();
      }
      // A null value means that we're supposed to create an empty file as the input.
      if (entry.getValue() != null) {
        key.createSymbolicLink(entry.getValue());
      } else {
        FileSystemUtils.createEmptyFile(key);
      }
    }

    for (Map.Entry<PathFragment, PathFragment> entry : inputs.getSymlinks().entrySet()) {
      Path key = workDir.getRelative(entry.getKey());
      FileStatus keyStat = key.statNullable(Symlinks.NOFOLLOW);
      if (keyStat != null) {
        // TODO(lberki): Why? Isn't this method supposed to be called on a fresh tree?
        key.delete();
      }
      key.createSymbolicLink(entry.getValue());
    }
  }
}
