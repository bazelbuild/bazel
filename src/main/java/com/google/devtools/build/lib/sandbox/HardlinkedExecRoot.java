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

package com.google.devtools.build.lib.sandbox;

import com.google.common.collect.ImmutableMap;
import com.google.common.io.Files;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Collection;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Creates an execRoot for a Spawn that contains input files as symlinks to hardlinks of the
 * original input files.
 */
public class HardlinkedExecRoot implements SandboxExecRoot {

  private final Path execRoot;
  private final Path sandboxPath;
  private final Path sandboxExecRoot;
  private final PrintWriter errWriter;

  public HardlinkedExecRoot(
      Path execRoot, Path sandboxPath, Path sandboxExecRoot, PrintWriter errWriter) {
    this.execRoot = execRoot;
    this.sandboxPath = sandboxPath;
    this.sandboxExecRoot = sandboxExecRoot;
    this.errWriter = errWriter;
  }

  @Override
  public void createFileSystem(
      Map<PathFragment, Path> inputs, Collection<PathFragment> outputs, Set<Path> writableDirs)
      throws IOException {
    Set<Path> createdDirs = new HashSet<>();
    FileSystemUtils.createDirectoryAndParentsWithCache(createdDirs, sandboxExecRoot);
    createDirectoriesForOutputs(outputs, createdDirs);

    // Create all needed directories.
    for (Path createDir : writableDirs) {
      if (errWriter != null) {
        errWriter.printf("createdir: %s\n", createDir.getPathString());
      }
      FileSystemUtils.createDirectoryAndParentsWithCache(createdDirs, createDir);
    }

    // Link all the inputs.
    linkInputs(inputs, createdDirs);
  }

  private void createDirectoriesForOutputs(Collection<PathFragment> outputs, Set<Path> createdDirs)
      throws IOException {
    // Prepare the output directories in the sandbox.
    for (PathFragment output : outputs) {
      FileSystemUtils.createDirectoryAndParentsWithCache(
          createdDirs, sandboxExecRoot.getRelative(output.getParentDirectory()));
    }
  }

  /**
   * Make all specified inputs available in the sandbox.
   *
   * <p>We want the sandboxed process to have access only to these input files and not anything else
   * from the workspace. Furthermore, the process should not be able to modify these input files. We
   * achieve this by hardlinking all input files into a temporary "inputs" directory, then
   * symlinking them into their correct place inside the sandbox.
   *
   * <p>The hardlinks / symlinks combination (as opposed to simply directly hardlinking to the final
   * destination) is necessary, because we build a solib symlink tree for shared libraries where the
   * original file and the created symlink have two different file names (libblaze_util.so vs.
   * src_Stest_Scpp_Sblaze_Uutil_Utest.so) and our cc_wrapper.sh needs to be able to figure out both
   * names (by following solib symlinks back) to modify the paths to the shared libraries in
   * cc_binaries.
   */
  private void linkInputs(Map<PathFragment, Path> inputs, Set<Path> createdDirs)
      throws IOException {
    // Create directory for input files.
    Path inputsDir = sandboxPath.getRelative("inputs");
    if (!inputsDir.exists()) {
      inputsDir.createDirectory();
    }

    for (ImmutableMap.Entry<PathFragment, Path> entry : inputs.entrySet()) {
      Path targetName = sandboxExecRoot.getRelative(entry.getKey());
      FileSystemUtils.createDirectoryAndParentsWithCache(
          createdDirs, targetName.getParentDirectory());

      // The target is supposed to be an empty file.
      if (entry.getValue() == null) {
        FileSystemUtils.createEmptyFile(targetName);
        continue;
      }

      // Hardlink, resolve symlink here instead in finalizeLinks.
      Path target = entry.getValue().resolveSymbolicLinks();
      Path hardlinkName =
          target.startsWith(execRoot)
              ? inputsDir.getRelative(target.relativeTo(execRoot))
              : inputsDir.getRelative(entry.getKey());
      if (errWriter != null) {
        errWriter.printf("hardlink: %s -> %s\n", hardlinkName, target);
      }
      try {
        createHardLink(hardlinkName, target);
      } catch (IOException e) {
        // Creating a hardlink might fail when the input file and the sandbox directory are not on
        // the same filesystem / device. Then we use symlink instead.
        hardlinkName.createSymbolicLink(target);
      }

      // symlink
      if (errWriter != null) {
        errWriter.printf("symlink: %s -> %s\n", targetName, hardlinkName);
      }
      targetName.createSymbolicLink(hardlinkName);
    }
  }

  private void createHardLink(Path target, Path source) throws IOException {
    java.nio.file.Path targetNio = java.nio.file.Paths.get(target.toString());
    java.nio.file.Path sourceNio = java.nio.file.Paths.get(source.toString());

    if (!source.exists() || target.exists()) {
      return;
    }
    // Regular file
    if (source.isFile()) {
      Path parentDir = target.getParentDirectory();
      if (!parentDir.exists()) {
        FileSystemUtils.createDirectoryAndParents(parentDir);
      }
      java.nio.file.Files.createLink(targetNio, sourceNio);
      // Directory
    } else if (source.isDirectory()) {
      // Eagerly create target directory in case source is an empty directory.
      FileSystemUtils.createDirectoryAndParents(target);

      Collection<Path> subpaths = source.getDirectoryEntries();
      for (Path sourceSubpath : subpaths) {
        Path targetSubpath = target.getRelative(sourceSubpath.relativeTo(source));
        createHardLink(targetSubpath, sourceSubpath);
      }
    }
  }

  @Override
  public void copyOutputs(Path execRoot, Collection<PathFragment> outputs) throws IOException {
    for (PathFragment output : outputs) {
      Path source = sandboxExecRoot.getRelative(output);
      Path target = execRoot.getRelative(output);
      if (source.isFile() || source.isSymbolicLink()) {
        Files.move(source.getPathFile(), target.getPathFile());
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
}
