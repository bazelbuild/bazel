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

package com.google.devtools.build.lib.bazel.repository.cache;

import com.google.devtools.build.lib.vfs.Path;
import javax.annotation.Nullable;

/**
 * A cache directory related to repositories, containing both the {@link DownloadCache} and the
 * {@link RepoContentsCache}.
 */
public class RepositoryCache {
  // Repository cache subdirectories
  private static final String CAS_DIR = "content_addressable";
  private static final String CONTENTS_DIR = "contents";

  private final DownloadCache downloadCache;
  private final RepoContentsCache repoContentsCache;

  @Nullable private Path path;

  public RepositoryCache() {
    downloadCache = new DownloadCache();
    repoContentsCache = new RepoContentsCache();
  }

  public void setPath(@Nullable Path path) {
    this.path = path;
    if (path != null) {
      downloadCache.setPath(path.getRelative(CAS_DIR));
      repoContentsCache.setPath(path.getRelative(CONTENTS_DIR));
    } else {
      downloadCache.setPath(null);
      repoContentsCache.setPath(null);
    }
  }

  public DownloadCache getDownloadCache() {
    return downloadCache;
  }

  public RepoContentsCache getRepoContentsCache() {
    return repoContentsCache;
  }

  @Nullable
  public Path getPath() {
    return path;
  }
}
