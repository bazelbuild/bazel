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

import static com.google.devtools.build.lib.vfs.Dirent.Type.DIRECTORY;
import static com.google.devtools.build.lib.vfs.Dirent.Type.SYMLINK;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.sandbox.SandboxHelpers;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.util.LinkedHashSet;
import java.util.Optional;
import java.util.Set;

/** Creates and manages the contents of a working directory of a persistent worker. */
final class WorkerExecRoot {
  private final Path workDir;

  public WorkerExecRoot(Path workDir) {
    this.workDir = workDir;
  }

  public void createFileSystem(
      Set<PathFragment> workerFiles, SandboxInputs inputs, SandboxOutputs outputs)
      throws IOException {
    workDir.createDirectoryAndParents();

    // First compute all the inputs and directories that we need. This is based only on
    // `workerFiles`, `inputs` and `outputs` and won't do any I/O.
    Set<PathFragment> inputsToCreate = new LinkedHashSet<>();
    LinkedHashSet<PathFragment> dirsToCreate = new LinkedHashSet<>();
    populateInputsAndDirsToCreate(inputs, workerFiles, outputs, inputsToCreate, dirsToCreate);

    // Then do a full traversal of the parent directory of `workDir`. This will use what we computed
    // above, delete anything unnecessary and update `inputsToCreate`/`dirsToCreate` if something is
    // can be left without changes (e.g., a symlink that already points to the right destination).
    // We're traversing from workDir's parent directory because external repositories can now be
    // symlinked as siblings of workDir when --experimental_sibling_repository_layout is in effect.
    cleanExisting(workDir.getParentDirectory(), inputs, inputsToCreate, dirsToCreate);

    // Finally, create anything that is still missing.
    createDirectories(dirsToCreate);
    createInputs(inputsToCreate, inputs);

    inputs.materializeVirtualInputs(workDir);
  }

  /** Populates the provided sets with the inputs and directories than need to be created. */
  private void populateInputsAndDirsToCreate(
      SandboxInputs inputs,
      Set<PathFragment> workerFiles,
      SandboxOutputs outputs,
      Set<PathFragment> inputsToCreate,
      LinkedHashSet<PathFragment> dirsToCreate) {
    // Add all worker files and the ancestor directories.
    for (PathFragment path : workerFiles) {
      inputsToCreate.add(path);
      for (int i = 0; i < path.segmentCount(); i++) {
        dirsToCreate.add(path.subFragment(0, i));
      }
    }

    // Add all inputs files and the ancestor directories.
    Iterable<PathFragment> allInputs =
        Iterables.concat(inputs.getFiles().keySet(), inputs.getSymlinks().keySet());
    for (PathFragment path : allInputs) {
      inputsToCreate.add(path);
      for (int i = 0; i < path.segmentCount(); i++) {
        dirsToCreate.add(path.subFragment(0, i));
      }
    }

    // And all ancestor directories of outputs. Note that we don't add the files themselves -- any
    // pre-existing files that have the same path as an output should get deleted.
    for (PathFragment path : Iterables.concat(outputs.files(), outputs.dirs())) {
      for (int i = 0; i < path.segmentCount(); i++) {
        dirsToCreate.add(path.subFragment(0, i));
      }
    }

    // Add all ouput directories, must be created after their parents above
    dirsToCreate.addAll(outputs.dirs());
  }

  /**
   * Deletes unnecessary files/directories and updates the sets if something on disk is already
   * correct and doesn't need any changes.
   */
  private void cleanExisting(
      Path root,
      SandboxInputs inputs,
      Set<PathFragment> inputsToCreate,
      Set<PathFragment> dirsToCreate)
      throws IOException {
    Path execroot = workDir.getParentDirectory();
    for (Dirent dirent : root.readdir(Symlinks.NOFOLLOW)) {
      Path absPath = root.getChild(dirent.getName());
      PathFragment pathRelativeToWorkDir;
      if (absPath.startsWith(workDir)) {
        // path is under workDir, i.e. execroot/<workspace name>. Simply get the relative path.
        pathRelativeToWorkDir = absPath.relativeTo(workDir);
      } else {
        // path is not under workDir, which means it belongs to one of external repositories
        // symlinked directly under execroot. Get the relative path based on there and prepend it
        // with the designated prefix, '../', so that it's still a valid relative path to workDir.
        pathRelativeToWorkDir =
            LabelConstants.EXPERIMENTAL_EXTERNAL_PATH_PREFIX.getRelative(
                absPath.relativeTo(execroot));
      }
      Optional<PathFragment> destination =
          getExpectedSymlinkDestination(pathRelativeToWorkDir, inputs);
      if (destination.isPresent()) {
        if (SYMLINK.equals(dirent.getType())
            && absPath.readSymbolicLink().equals(destination.get())) {
          inputsToCreate.remove(pathRelativeToWorkDir);
        } else {
          absPath.delete();
        }
      } else if (DIRECTORY.equals(dirent.getType())) {
        if (dirsToCreate.contains(pathRelativeToWorkDir)) {
          cleanExisting(absPath, inputs, inputsToCreate, dirsToCreate);
          dirsToCreate.remove(pathRelativeToWorkDir);
        } else {
          absPath.deleteTree();
        }
      } else if (!inputsToCreate.contains(pathRelativeToWorkDir)) {
        absPath.delete();
      }
    }
  }

  private Optional<PathFragment> getExpectedSymlinkDestination(
      PathFragment fragment, SandboxInputs inputs) {
    Path file = inputs.getFiles().get(fragment);
    if (file != null) {
      return Optional.of(file.asFragment());
    }
    return Optional.ofNullable(inputs.getSymlinks().get(fragment));
  }

  private void createDirectories(Iterable<PathFragment> dirsToCreate) throws IOException {
    for (PathFragment fragment : dirsToCreate) {
      workDir.getRelative(fragment).createDirectory();
    }
  }

  private void createInputs(Iterable<PathFragment> inputsToCreate, SandboxInputs inputs)
      throws IOException {
    for (PathFragment fragment : inputsToCreate) {
      Path key = workDir.getRelative(fragment);
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
