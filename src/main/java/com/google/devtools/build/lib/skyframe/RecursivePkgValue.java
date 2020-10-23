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
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * This value represents the result of looking up all the packages under a given package path root,
 * starting at a given directory.
 */
@Immutable
@ThreadSafe
public class RecursivePkgValue implements SkyValue {
  @AutoCodec
  static final RecursivePkgValue EMPTY =
      new RecursivePkgValue(NestedSetBuilder.<String>emptySet(Order.STABLE_ORDER), false);

  private final NestedSet<String> packages;
  private final boolean hasErrors;

  private RecursivePkgValue(NestedSet<String> packages, boolean hasErrors) {
    this.packages = packages;
    this.hasErrors = hasErrors;
  }

  static RecursivePkgValue create(NestedSetBuilder<String> packages, boolean hasErrors) {
    if (packages.isEmpty() && !hasErrors) {
      return EMPTY;
    }
    return new RecursivePkgValue(packages.build(), hasErrors);
  }

  /** Create a transitive package lookup request. */
  @ThreadSafe
  public static Key key(
      RepositoryName repositoryName,
      RootedPath rootedPath,
      ImmutableSet<PathFragment> excludedPaths) {
    return Key.create(repositoryName, rootedPath, excludedPaths);
  }

  public NestedSet<String> getPackages() {
    return packages;
  }

  public boolean hasErrors() {
    return hasErrors;
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class Key extends RecursivePkgSkyKey {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    private Key(
        RepositoryName repositoryName,
        RootedPath rootedPath,
        ImmutableSet<PathFragment> excludedPaths) {
      super(repositoryName, rootedPath, excludedPaths);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static Key create(
        RepositoryName repositoryName,
        RootedPath rootedPath,
        ImmutableSet<PathFragment> excludedPaths) {
      return interner.intern(new Key(repositoryName, rootedPath, excludedPaths));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.RECURSIVE_PKG;
    }
  }
}
