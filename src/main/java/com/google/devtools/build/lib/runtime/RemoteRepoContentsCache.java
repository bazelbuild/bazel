// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime;

import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction;
import java.io.IOException;
import javax.annotation.Nullable;

/** A remote cache for the contents of external repositories. */
public interface RemoteRepoContentsCache {
  /** Adds a repository that has been fetched locally to the remote cache. */
  void addToCache(
      RepositoryName repoName,
      Path fetchedRepoDir,
      Path fetchedRepoMarkerFile,
      String predeclaredInputHash,
      ExtendedEventHandler reporter)
      throws InterruptedException;

  /**
   * Retrieves a repository from the remote cache if possible.
   *
   * <p>Callers have to check {@code env.valuesMissing()} after this method returns.
   *
   * @return true if there was a cache hit and the repository has been fetched into the given
   *     directory.
   */
  boolean lookupCache(
      RepositoryName repoName,
      Path repoDir,
      String predeclaredInputHash,
      SkyFunction.Environment env)
      throws IOException, InterruptedException;

  /**
   * Returns the contents of the marker file of the cache entry that the given repository has been
   * retrieved from if the remote cache has since lost the contents of some of its files while the
   * repository is still served from memory, otherwise null.
   *
   * <p>Such a repository has to be fetched into a different directory and passed to {@link
   * #restoreLostFiles} rather than fetched in place, as its remaining contents may be in use.
   */
  @Nullable
  String getLostFilesMarkerFile(RepositoryName repoName, Path repoDir);

  /**
   * Restores the files of a repository that the remote cache has lost from a fresh fetch of the
   * repository into a different directory, which is consumed in the process.
   *
   * @throws NonReproducibleRepoException if the fetched contents differ from those that have been
   *     retrieved from the remote cache
   */
  void restoreLostFiles(
      RepositoryName repoName,
      Path repoDir,
      Path fetchedRepoDir,
      Path fetchedRepoMarkerFile,
      String predeclaredInputHash,
      ExtendedEventHandler reporter)
      throws IOException, InterruptedException;

  /**
   * Thrown if a repository whose repo rule declared it as reproducible turned out to have different
   * contents when it was fetched again.
   */
  final class NonReproducibleRepoException extends IOException {
    /**
     * @param difference describes how the contents differ
     */
    public NonReproducibleRepoException(RepositoryName repoName, String difference) {
      super(
          ("the repo rule declares the contents of repository %s to be reproducible, but fetching"
                  + " it again to restore files lost by the remote cache resulted in different"
                  + " contents: %s")
              .formatted(repoName, difference));
    }
  }
}
