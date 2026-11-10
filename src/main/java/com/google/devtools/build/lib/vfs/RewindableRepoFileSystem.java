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
import javax.annotation.Nullable;

/**
 * Implemented by {@link FileSystem}s that serve the contents of external repositories from a remote
 * cache and support recovering from the remote cache losing the contents of individual files by
 * fetching the repository again.
 */
public interface RewindableRepoFileSystem {

  /**
   * Returns the given file system if it can recover lost repository files during a command,
   * otherwise {@code null}.
   */
  @Nullable
  static RewindableRepoFileSystem of(FileSystem fileSystem) {
    return fileSystem instanceof RewindableRepoFileSystem repoFileSystem ? repoFileSystem : null;
  }

  /** Returns whether the given path lies within the contents of a repository. */
  boolean isRepoPath(PathFragment path);

  /**
   * Returns the repository whose contents the given path lies in, which requires {@link
   * #isRepoPath} to hold for it.
   */
  RepositoryName repoContaining(PathFragment path);

  /**
   * Records that a file in the given repository is no longer available in the remote cache, so that
   * rewinding the fetch of that repository recovers it.
   */
  void markLostRepoFile(RepositoryName repo);

  /**
   * Returns whether the contents of the given repo have been retrieved from the remote repo contents
   * cache during this command, so that a file of it that the remote cache has lost can be restored
   * by rewinding its fetch.
   */
  boolean isServedFromCache(RepositoryName repo);

  /**
   * Records that the given repository has been fetched anew by running its repo rule, which
   * replaces any contents that referenced files the remote cache has lost.
   */
  void repoRefetched(RepositoryName repo);

  /**
   * Records that fetching the given repository anew by running its repo rule has failed, so that
   * its contents are looked up in the remote cache again: they may only have been temporarily
   * unavailable.
   */
  void repoRefetchFailed(RepositoryName repo);
}
