// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;

/**
 * Dummy value that is the result of {@link PrepareDepsOfTargetsUnderDirectoryFunction}.
 *
 * <p>Note that even though the {@link PrepareDepsOfTargetsUnderDirectoryFunction} is evaluated
 * entirely because of its side effects (i.e. loading transitive dependencies of targets), this
 * value interacts safely with change pruning, despite the fact that this value is a singleton. When
 * the targets in a package change, the {@link PackageValue} that
 * {@link PrepareDepsOfTargetsUnderDirectoryFunction} depends on will be invalidated, and the
 * PrepareDeps function for that package's directory will be re-evaluated, loading any new
 * transitive dependencies. Change pruning may prevent the re-evaluation of PrepareDeps for
 * directories above that one, but they don't need to be re-run.
 */
public final class PrepareDepsOfTargetsUnderDirectoryValue implements SkyValue {
  @AutoCodec
  public static final PrepareDepsOfTargetsUnderDirectoryValue INSTANCE =
      new PrepareDepsOfTargetsUnderDirectoryValue();

  private PrepareDepsOfTargetsUnderDirectoryValue() {}

  /** Create a prepare deps of targets under directory request. */
  @ThreadSafe
  public static SkyKey key(RepositoryName repository, RootedPath rootedPath,
      ImmutableSet<PathFragment> excludedPaths) {
    return key(repository, rootedPath, excludedPaths, FilteringPolicies.NO_FILTER);
  }

  /**
   * Create a prepare deps of targets under directory request, specifying a filtering policy for
   * targets.
   */
  @ThreadSafe
  public static PrepareDepsOfTargetsUnderDirectoryKey key(
      RepositoryName repository,
      RootedPath rootedPath,
      ImmutableSet<PathFragment> excludedPaths,
      FilteringPolicy filteringPolicy) {
    return PrepareDepsOfTargetsUnderDirectoryKey.create(
        new RecursivePkgKey(repository, rootedPath, excludedPaths), filteringPolicy);
  }

  /**
   * The argument value for {@link SkyKey}s of {@link PrepareDepsOfTargetsUnderDirectoryFunction}.
   */
  @AutoCodec
  public static final class PrepareDepsOfTargetsUnderDirectoryKey implements SkyKey {
    private static final Interner<PrepareDepsOfTargetsUnderDirectoryKey> interners =
        BlazeInterners.newWeakInterner();

    private final RecursivePkgKey recursivePkgKey;
    private final FilteringPolicy filteringPolicy;

    private PrepareDepsOfTargetsUnderDirectoryKey(
        RecursivePkgKey recursivePkgKey, FilteringPolicy filteringPolicy) {
      this.recursivePkgKey = Preconditions.checkNotNull(recursivePkgKey);
      this.filteringPolicy = Preconditions.checkNotNull(filteringPolicy);
    }

    @AutoCodec.Instantiator
    public static PrepareDepsOfTargetsUnderDirectoryKey create(
        RecursivePkgKey recursivePkgKey, FilteringPolicy filteringPolicy) {
      return interners.intern(
          new PrepareDepsOfTargetsUnderDirectoryKey(recursivePkgKey, filteringPolicy));
    }

    public RecursivePkgKey getRecursivePkgKey() {
      return recursivePkgKey;
    }

    public FilteringPolicy getFilteringPolicy() {
      return filteringPolicy;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.PREPARE_DEPS_OF_TARGETS_UNDER_DIRECTORY;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof PrepareDepsOfTargetsUnderDirectoryKey)) {
        return false;
      }

      PrepareDepsOfTargetsUnderDirectoryKey that = (PrepareDepsOfTargetsUnderDirectoryKey) o;
      return Objects.equals(recursivePkgKey, that.recursivePkgKey)
          && Objects.equals(filteringPolicy, that.filteringPolicy);
    }

    @Override
    public int hashCode() {
      return Objects.hash(recursivePkgKey, filteringPolicy);
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(PrepareDepsOfTargetsUnderDirectoryKey.class)
              .add("pkg-key", recursivePkgKey)
              .add("filtering policy", filteringPolicy)
              .toString();
    }
  }
}
