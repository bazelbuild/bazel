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

package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import java.util.Objects;

/**
 * A RecursivePkgKey is a tuple of a {@link RootedPath}, {@code rootedPath}, defining the directory
 * to recurse beneath in search of packages, and an {@link ImmutableSet} of {@link PathFragment}s,
 * {@code excludedPaths}, relative to {@code rootedPath.getRoot}, defining the set of subdirectories
 * strictly beneath {@code rootedPath} to skip.
 *
 * <p>Throws {@link IllegalArgumentException} if {@code excludedPaths} contains any paths that are
 * equal to {@code rootedPath} or that are not beneath {@code rootedPath}.
 */
@ThreadSafe
public class RecursivePkgKey {
  @VisibleForSerialization final RepositoryName repositoryName;
  @VisibleForSerialization final RootedPath rootedPath;
  @VisibleForSerialization final ImmutableSet<PathFragment> excludedPaths;

  public RecursivePkgKey(
      RepositoryName repositoryName,
      RootedPath rootedPath,
      ImmutableSet<PathFragment> excludedPaths) {
    PathFragment.checkAllPathsAreUnder(excludedPaths, rootedPath.getRootRelativePath());
    this.repositoryName = repositoryName;
    this.rootedPath = Preconditions.checkNotNull(rootedPath);
    this.excludedPaths = Preconditions.checkNotNull(excludedPaths);
  }

  public RepositoryName getRepositoryName() {
    return repositoryName;
  }

  public RootedPath getRootedPath() {
    return rootedPath;
  }

  public ImmutableSet<PathFragment> getExcludedPaths() {
    return excludedPaths;
  }

  @Override
  public String toString() {
    return "rootedPath=" + rootedPath + ", excludedPaths=<omitted>";
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof RecursivePkgKey that)) {
      return false;
    }

    return excludedPaths.equals(that.excludedPaths)
        && rootedPath.equals(that.rootedPath)
        && repositoryName.equals(that.repositoryName);
  }

  @Override
  public int hashCode() {
    return Objects.hash(rootedPath, excludedPaths, repositoryName);
  }
}
