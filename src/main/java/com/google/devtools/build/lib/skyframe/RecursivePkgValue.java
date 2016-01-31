// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.Serializable;
import java.util.Objects;

/**
 * This value represents the result of looking up all the packages under a given package path root,
 * starting at a given directory.
 */
@Immutable
@ThreadSafe
public class RecursivePkgValue implements SkyValue {
  static final RecursivePkgValue EMPTY =
      new RecursivePkgValue(NestedSetBuilder.<String>emptySet(Order.STABLE_ORDER));

  private final NestedSet<String> packages;

  private RecursivePkgValue(NestedSet<String> packages) {
    this.packages = packages;
  }

  static RecursivePkgValue create(NestedSetBuilder<String> packages) {
    if (packages.isEmpty()) {
      return EMPTY;
    }
    return new RecursivePkgValue(packages.build());
  }

  /**
   * Create a transitive package lookup request.
   */
  @ThreadSafe
  public static SkyKey key(RepositoryName repositoryName, RootedPath rootedPath,
      ImmutableSet<PathFragment> excludedPaths) {
    return new SkyKey(SkyFunctions.RECURSIVE_PKG,
        new RecursivePkgKey(repositoryName, rootedPath, excludedPaths));
  }

  public NestedSet<String> getPackages() {
    return packages;
  }

  /**
   * A RecursivePkgKey is a tuple of a {@link RootedPath}, {@code rootedPath}, defining the
   * directory to recurse beneath in search of packages, and an {@link ImmutableSet} of {@link
   * PathFragment}s, {@code excludedPaths}, relative to {@code rootedPath.getRoot}, defining the
   * set of subdirectories beneath {@code rootedPath} to skip.
   *
   * <p>Throws {@link IllegalArgumentException} if {@code excludedPaths} contains any paths that
   * are equal to {@code rootedPath} or that are not beneath {@code rootedPath}.
   */
  @ThreadSafe
  public static final class RecursivePkgKey implements Serializable {
    private final RepositoryName repositoryName;
    private final RootedPath rootedPath;
    private final ImmutableSet<PathFragment> excludedPaths;

    public RecursivePkgKey(RepositoryName repositoryName, RootedPath rootedPath,
        ImmutableSet<PathFragment> excludedPaths) {
      PathFragment.checkAllPathsAreUnder(excludedPaths,
          rootedPath.getRelativePath());
      this.repositoryName = repositoryName;
      this.rootedPath = Preconditions.checkNotNull(rootedPath);
      this.excludedPaths = Preconditions.checkNotNull(excludedPaths);
    }

    public RepositoryName getRepository() {
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
      return "rootedPath=" + rootedPath + ", excludedPaths=<omitted>)";
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof RecursivePkgKey)) {
        return false;
      }

      RecursivePkgKey that = (RecursivePkgKey) o;
      return excludedPaths.equals(that.excludedPaths) && rootedPath.equals(that.rootedPath);
    }

    @Override
    public int hashCode() {
      return Objects.hash(rootedPath, excludedPaths);
    }
  }
}
