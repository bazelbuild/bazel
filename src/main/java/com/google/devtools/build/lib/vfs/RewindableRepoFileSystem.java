// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.vfs;

import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.vfs.RewindingSynchronizer.TransferableWriteLock;
import java.io.IOException;
import java.io.InterruptedIOException;
import javax.annotation.Nullable;

/**
 * Implemented by {@link FileSystem}s that serve the contents of external repositories from a remote
 * cache and support recovering from the remote cache losing the contents of individual files by
 * refetching the repository.
 */
public interface RewindableRepoFileSystem {

  /**
   * Returns the given file system if it can replace repository contents during a command, otherwise
   * {@code null}.
   */
  @Nullable
  static RewindableRepoFileSystem of(FileSystem fileSystem) {
    return fileSystem instanceof RewindableRepoFileSystem repoFileSystem ? repoFileSystem : null;
  }

  /** A single read of repository contents. */
  @FunctionalInterface
  interface Read<T> {
    T read() throws IOException;
  }

  /**
   * Runs a single read of the given path while holding the read lock of the repository containing
   * it, so that it doesn't observe a refetch of that repository halfway. Reads of paths outside of
   * repositories or on other file systems run directly.
   */
  static <T> T readUnderRepoLock(Path path, Read<T> read) throws IOException {
    return readUnderRepoLock(path, path::asFragment, read);
  }

  /**
   * Like {@link #readUnderRepoLock(Path, Read)}, but locks the repository containing the path
   * supplied by {@code realPath}, which is the path that reading {@code path} actually reads, e.g.
   * the resolved path of an artifact that was materialized as a symlink into a repository. The
   * supplier is only consulted if the file system can replace repository contents at all.
   */
  static <T> T readUnderRepoLock(Path path, Read<PathFragment> realPath, Read<T> read)
      throws IOException {
    if (!(path.getFileSystem() instanceof RewindableRepoFileSystem repoFileSystem)) {
      return read.read();
    }
    PathFragment realPathFragment = realPath.read();
    if (!repoFileSystem.isRepoPath(realPathFragment)) {
      return read.read();
    }
    try (SilentCloseable unused =
        repoFileSystem
            .getRewindingSynchronizer()
            .acquireReadLock(() -> repoFileSystem.repoContaining(realPathFragment))) {
      return read.read();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new InterruptedIOException(
          "interrupted while waiting for the refetch of " + realPathFragment);
    }
  }

  /** Returns the synchronizer for accesses to and replacements of repository contents. */
  RewindingSynchronizer getRewindingSynchronizer();

  /** Returns whether the given path lies within the contents of a repository. */
  boolean isRepoPath(PathFragment path);

  /**
   * Returns the repository whose contents the given path lies in, which requires {@link
   * #isRepoPath} to hold for it.
   */
  RepositoryName repoContaining(PathFragment path);

  /**
   * Acquires the exclusive lock that has to be held while the contents of the given repository are
   * replaced.
   */
  default TransferableWriteLock acquireRepoWriteLock(RepositoryName repo)
      throws InterruptedException {
    return getRewindingSynchronizer().acquireWriteLock(repo);
  }

}
