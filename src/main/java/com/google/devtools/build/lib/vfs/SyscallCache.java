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

import static com.google.common.base.Preconditions.checkState;

import java.io.IOException;
import java.util.Collection;
import javax.annotation.Nullable;

/**
 * Centralized point to perform filesystem calls, to promote caching. Ideally all filesystem
 * operations would be cached in Skyframe, but even then, implementations of this interface may do
 * batch operations and prefetching to improve performance.
 *
 * <p>There is typically one {@link SyscallCache} instance in effect for the lifetime of the Bazel
 * server, set in {@link com.google.devtools.build.lib.runtime.WorkspaceBuilder}. Between commands,
 * {@link #clear} is called to drop cached data from the previous command.
 *
 * <p>See the note in {@link XattrProvider} about caching in implementations. Do not call the
 * methods in this interface on files that may change <em>during</em> a build, like outputs or
 * external repository files. Calling these methods on source files is allowed.
 */
public interface SyscallCache extends XattrProvider {
  SyscallCache NO_CACHE =
      new SyscallCache() {
        @Override
        public Collection<Dirent> readdir(Path path) throws IOException {
          return path.readdir(Symlinks.NOFOLLOW);
        }

        @Nullable
        @Override
        public FileStatus statIfFound(Path path, Symlinks symlinks) throws IOException {
          return path.statIfFound(symlinks);
        }

        @Override
        public DirentTypeWithSkip getType(Path path, Symlinks symlinks) {
          return DirentTypeWithSkip.FILESYSTEM_OP_SKIPPED;
        }

        @Override
        public void clear() {}
      };

  /** Gets directory entries and their types. Does not follow symlinks. */
  Collection<Dirent> readdir(Path path) throws IOException;

  /** Returns the stat() for the given path, or null. */
  @Nullable
  FileStatus statIfFound(Path path, Symlinks symlinks) throws IOException;

  /**
   * Returns the type of a specific file. This may be answered using stat() or readdir(). Returns
   * null if the path does not exist. Returns {@link DirentTypeWithSkip#FILESYSTEM_OP_SKIPPED} if
   * cache had no data for path and chose not to do filesystem access to determine the type. Callers
   * should call {@link #statIfFound} and then {@link #statusToDirentType} if needed in that case.
   */
  @Nullable
  DirentTypeWithSkip getType(Path path, Symlinks symlinks) throws IOException;

  /** Called before each build. Implementations should flush their caches at that point. */
  void clear();

  /**
   * Called at the end of the analysis phase (if not doing merged analysis/execution). Cache may
   * choose to drop some data then.
   */
  default void noteAnalysisPhaseEnded() {
    clear();
  }

  /**
   * A {@link Dirent.Type} with an additional element signifying that the type is unknown because
   * this {@link SyscallCache} implementation skipped filesystem access.
   */
  enum DirentTypeWithSkip {
    FILE(Dirent.Type.FILE),
    DIRECTORY(Dirent.Type.DIRECTORY),
    SYMLINK(Dirent.Type.SYMLINK),
    UNKNOWN(Dirent.Type.UNKNOWN),
    FILESYSTEM_OP_SKIPPED(null);

    @Nullable private final Dirent.Type type;

    DirentTypeWithSkip(@Nullable Dirent.Type type) {
      this.type = type;
    }

    public Dirent.Type getType() {
      checkState(this != FILESYSTEM_OP_SKIPPED, "No type if filesystem op skipped");
      return type;
    }

    @Nullable
    public static DirentTypeWithSkip of(@Nullable Dirent.Type type) {
      if (type == null) {
        return null;
      }
      switch (type) {
        case FILE:
          return FILE;
        case DIRECTORY:
          return DIRECTORY;
        case SYMLINK:
          return SYMLINK;
        case UNKNOWN:
          return UNKNOWN;
      }
      throw new IllegalStateException("Got unrecognized type " + type);
    }
  }

  @Nullable
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
