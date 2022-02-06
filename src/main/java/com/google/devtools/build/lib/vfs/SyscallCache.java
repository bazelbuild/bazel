// Copyright 2022 The Bazel Authors. All rights reserved.
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

import java.io.IOException;
import java.util.Collection;

/**
 * Centralized point to perform filesystem calls, to promote caching. Ideally all filesystem
 * operations would be cached in Skyframe, but even then, implementations of this interface may do
 * batch operations and prefetching to improve performance.
 */
public interface SyscallCache {
  SyscallCache NO_CACHE =
      new SyscallCache() {
        @Override
        public Collection<Dirent> readdir(Path path) throws IOException {
          return path.readdir(Symlinks.NOFOLLOW);
        }

        @Override
        public FileStatus statIfFound(Path path, Symlinks symlinks) throws IOException {
          return path.statIfFound(symlinks);
        }

        @Override
        public Dirent.Type getType(Path path, Symlinks symlinks) throws IOException {
          return statusToDirentType(statIfFound(path, symlinks));
        }

        @Override
        public void clear() {}
      };

  /** Gets directory entries and their types. Does not follow symlinks. */
  Collection<Dirent> readdir(Path path) throws IOException;

  /** Returns the stat() for the given path, or null. */
  FileStatus statIfFound(Path path, Symlinks symlinks) throws IOException;

  /**
   * Returns the type of a specific file. This may be answered using stat() or readdir(). Returns
   * null if the path does not exist.
   */
  Dirent.Type getType(Path path, Symlinks symlinks) throws IOException;

  default byte[] getFastDigest(Path path) throws IOException {
    return path.getFastDigest();
  }

  /** Called before each build. Implementations should flush their caches at that point. */
  void clear();

  /**
   * Called at the end of the analysis phase (if not doing merged analysis/execution). Cache may
   * choose to drop some data then.
   */
  default void noteAnalysisPhaseEnded() {
    clear();
  }

  static Dirent.Type statusToDirentType(FileStatus status) {
    if (status == null) {
      return null;
    } else if (status.isFile()) {
      return Dirent.Type.FILE;
    } else if (status.isDirectory()) {
      return Dirent.Type.DIRECTORY;
    } else if (status.isSymbolicLink()) {
      return Dirent.Type.SYMLINK;
    }
    return Dirent.Type.UNKNOWN;
  }
}
