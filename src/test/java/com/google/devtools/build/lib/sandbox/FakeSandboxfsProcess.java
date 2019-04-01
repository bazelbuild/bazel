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

package com.google.devtools.build.lib.sandbox;

import static com.google.common.base.Preconditions.checkState;

import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.List;

/**
 * A fake in-process sandboxfs implementation that uses symlinks on the Bazel file system API.
 *
 * <p>TODO(jmmv): It's possible that this could replace {@link SymlinkedSandboxedSpawn} altogether,
 * simplifying all callers that need to perform a sandboxed spawn because they would all go through
 * the sandboxfs worker interface.  Evaluate this idea once we are confident enough that we won't
 * just remove all sandboxfs support code.
 */
final class FakeSandboxfsProcess implements SandboxfsProcess {

  /** File system on which the fake sandboxfs instance operates. */
  private final FileSystem fileSystem;

  /** Directory on which the sandboxfs is serving. */
  private final PathFragment mountPoint;

  /**
   * Whether this "process" is valid or not.  Used to better represent the workflow of a real
   * sandboxfs subprocess.
   */
  private boolean alive = true;

  /**
   * Initializes a new sandboxfs process instance.
   *
   * <p>To better represent reality, this ensures that the mount point is present and valid.
   *
   * @param fileSystem file system on which the fake sandboxfs instance operates
   * @param mountPoint directory on which the sandboxfs instance is serving
   * @throws IOException if the mount point is missing
   */
  FakeSandboxfsProcess(FileSystem fileSystem, PathFragment mountPoint) throws IOException {
    if (!fileSystem.getPath(mountPoint).exists()) {
      throw new IOException("Mount point " + mountPoint + " does not exist");
    } else if (!fileSystem.getPath(mountPoint).isDirectory()) {
      throw new IOException("Mount point " + mountPoint + " is not a directory");
    }

    this.fileSystem = fileSystem;
    this.mountPoint = mountPoint;
  }

  @Override
  public Path getMountPoint() {
    return fileSystem.getPath(mountPoint);
  }

  @Override
  public synchronized boolean isAlive() {
    return alive;
  }

  @Override
  public synchronized void destroy() {
    alive = false;
  }

  @Override
  public synchronized void map(List<Mapping> mappings) throws IOException {
    checkState(alive, "Cannot be called after destroy()");

    for (Mapping mapping : mappings) {
      checkState(mapping.path().isAbsolute(), "Mapping specifications are expected to be absolute"
          + " but %s is not", mapping.path());
      Path link = fileSystem.getPath(mountPoint).getRelative(mapping.path().toRelative());
      link.getParentDirectory().createDirectoryAndParents();

      Path target = fileSystem.getPath(mapping.target());
      if (!target.exists()) {
        // Not a requirement for the creation of a symbolic link but this reflects the behavior of
        // the real sandboxfs.
        throw new IOException("Target " + mapping.target() + " does not exist");
      }

      if (target.isSymbolicLink()) {
        // sandboxfs is able to expose symlinks as they are in the underlying file system.  Mimic
        // this behavior by respecting the symlink in that case, instead of just creating a new
        // symlink that points to the actual target.
        link.createSymbolicLink(target.readSymbolicLink());
      } else {
        link.createSymbolicLink(fileSystem.getPath(mapping.target()));
      }
    }
  }

  @Override
  public synchronized void unmap(PathFragment mapping) throws IOException {
    checkState(alive, "Cannot be called after destroy()");

    checkState(mapping.isAbsolute(), "Mapping specifications are expected to be absolute"
        + " but %s is not", mapping);
    fileSystem.getPath(mountPoint).getRelative(mapping.toRelative()).deleteTree();
  }
}
