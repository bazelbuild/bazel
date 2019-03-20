// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.sandbox;

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.FileSystemUtils.MoveResult;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;

/**
 * Implements the general flow of a sandboxed spawn that uses a container directory to build an
 * execution root for a spawn.
 */
public abstract class AbstractContainerizingSandboxedSpawn implements SandboxedSpawn {

  private static final Logger logger =
      Logger.getLogger(AbstractContainerizingSandboxedSpawn.class.getName());

  private static final AtomicBoolean warnedAboutMovesBeingCopies = new AtomicBoolean(false);

  private final Path sandboxPath;
  private final Path sandboxExecRoot;
  private final List<String> arguments;
  private final Map<String, String> environment;
  private final Map<PathFragment, Path> inputs;
  private final SandboxOutputs outputs;
  private final Set<Path> writableDirs;

  public AbstractContainerizingSandboxedSpawn(
      Path sandboxPath,
      Path sandboxExecRoot,
      List<String> arguments,
      Map<String, String> environment,
      Map<PathFragment, Path> inputs,
      SandboxOutputs outputs,
      Set<Path> writableDirs) {
    this.sandboxPath = sandboxPath;
    this.sandboxExecRoot = sandboxExecRoot;
    this.arguments = arguments;
    this.environment = environment;
    this.inputs = inputs;
    this.outputs = outputs;
    this.writableDirs = writableDirs;
  }

  @Override
  public Path getSandboxExecRoot() {
    return sandboxExecRoot;
  }

  @Override
  public List<String> getArguments() {
    return arguments;
  }

  @Override
  public Map<String, String> getEnvironment() {
    return environment;
  }

  @Override
  public void createFileSystem() throws IOException {
    createDirectories();
    createInputs(inputs);
  }

  /**
   * No input can be a child of another input, because otherwise we might try to create a symlink
   * below another symlink we created earlier - which means we'd actually end up writing somewhere
   * in the workspace.
   *
   * <p>If all inputs were regular files, this situation could naturally not happen - but
   * unfortunately, we might get the occasional action that has directories in its inputs.
   *
   * <p>Creating all parent directories first ensures that we can safely create symlinks to
   * directories, too, because we'll get an IOException with EEXIST if inputs happen to be nested
   * once we start creating the symlinks for all inputs.
   */
  private void createDirectories() throws IOException {
    LinkedHashSet<Path> dirsToCreate = new LinkedHashSet<>();

    for (PathFragment path : Iterables.concat(inputs.keySet(), outputs.files(), outputs.dirs())) {
      Preconditions.checkArgument(!path.isAbsolute());
      Preconditions.checkArgument(!path.containsUplevelReferences());
      for (int i = 0; i < path.segmentCount(); i++) {
        dirsToCreate.add(sandboxExecRoot.getRelative(path.subFragment(0, i)));
      }
    }
    for (PathFragment path : outputs.dirs()) {
      dirsToCreate.add(sandboxExecRoot.getRelative(path));
    }

    for (Path path : dirsToCreate) {
      path.createDirectory();
    }

    for (Path dir : writableDirs) {
      if (dir.startsWith(sandboxExecRoot)) {
        dir.createDirectoryAndParents();
      }
    }
  }

  protected void createInputs(Map<PathFragment, Path> inputs) throws IOException {
    // All input files are relative to the execroot.
    for (Map.Entry<PathFragment, Path> entry : inputs.entrySet()) {
      Path key = sandboxExecRoot.getRelative(entry.getKey());
      // A null value means that we're supposed to create an empty file as the input.
      if (entry.getValue() != null) {
        copyFile(entry.getValue(), key);
      } else {
        FileSystemUtils.createEmptyFile(key);
      }
    }
  }

  protected abstract void copyFile(Path source, Path target) throws IOException;

  /**
   * Moves all given outputs from a root to another.
   *
   * <p>This is a support function to help with the implementation of {@link #copyOutputs(Path)}.
   *
   * @param outputs outputs to move as relative paths to a root
   * @param sourceRoot source directory from which to resolve outputs
   * @param targetRoot target directory to which to move the resolved outputs from the source
   * @throws IOException if any of the moves fails
   */
  static void moveOutputs(SandboxOutputs outputs, Path sourceRoot, Path targetRoot)
      throws IOException {
    for (PathFragment output : Iterables.concat(outputs.files(), outputs.dirs())) {
      Path source = sourceRoot.getRelative(output);
      Path target = targetRoot.getRelative(output);
      if (source.isFile() || source.isSymbolicLink()) {
        // Ensure the target directory exists in the target. The directories for the action outputs
        // have already been created, but the spawn outputs may be different from the overall action
        // outputs. This is the case for test actions.
        target.getParentDirectory().createDirectoryAndParents();
        if (FileSystemUtils.moveFile(source, target).equals(MoveResult.FILE_COPIED)) {
          if (warnedAboutMovesBeingCopies.compareAndSet(false, true)) {
            logger.warning(
                "Moving files out of the sandbox (e.g. from "
                    + source
                    + " to "
                    + target
                    + ") had to be done with a file copy, which is detrimental to performance; are "
                    + " the two trees in different file systems?");
          }
        }
      } else if (source.isDirectory()) {
        try {
          source.renameTo(target);
        } catch (IOException e) {
          // Failed to move directory directly, thus move it recursively.
          target.createDirectory();
          FileSystemUtils.moveTreesBelow(source, target);
        }
      }
    }
  }

  @Override
  public void copyOutputs(Path execRoot) throws IOException {
    moveOutputs(outputs, sandboxExecRoot, execRoot);
  }

  @Override
  public void delete() {
    try {
      sandboxPath.deleteTree();
    } catch (IOException e) {
      // This usually means that the Spawn itself exited, but still has children running that
      // we couldn't wait for, which now block deletion of the sandbox directory. On Linux this
      // should never happen, as we use PID namespaces and where they are not available the
      // subreaper feature to make sure all children have been reliably killed before returning,
      // but on other OS this might not always work. The SandboxModule will try to delete them
      // again when the build is all done, at which point it hopefully works, so let's just go
      // on here.
    }
  }
}
