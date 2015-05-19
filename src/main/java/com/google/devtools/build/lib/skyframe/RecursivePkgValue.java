// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
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

  private final NestedSet<String> packages;

  public RecursivePkgValue(NestedSet<String> packages) {
    this.packages = packages;
  }

  /**
   * Create a transitive package lookup request.
   */
  @ThreadSafe
  public static SkyKey key(RootedPath rootedPath, ImmutableSet<PathFragment> excludedPaths) {
    return new SkyKey(SkyFunctions.RECURSIVE_PKG, new RecursivePkgKey(rootedPath, excludedPaths));
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
   * are not beneath {@code rootedPath}.
   */
  @ThreadSafe
  public static final class RecursivePkgKey implements Serializable {
    private final RootedPath rootedPath;
    private final ImmutableSet<PathFragment> excludedPaths;

    private RecursivePkgKey(RootedPath rootedPath, ImmutableSet<PathFragment> excludedPaths) {
      this.rootedPath = Preconditions.checkNotNull(rootedPath);
      this.excludedPaths = Preconditions.checkNotNull(excludedPaths);

      PathFragment rootedPathFragment = rootedPath.getRelativePath();
      for (PathFragment excludedPath : excludedPaths) {
        Preconditions.checkArgument(!excludedPath.equals(rootedPathFragment)
            && excludedPath.startsWith(rootedPathFragment), "%s is not beneath %s", excludedPath,
            rootedPathFragment);
      }
    }

    public RootedPath getRootedPath() {
      return rootedPath;
    }

    public ImmutableSet<PathFragment> getExcludedPaths() {
      return excludedPaths;
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
