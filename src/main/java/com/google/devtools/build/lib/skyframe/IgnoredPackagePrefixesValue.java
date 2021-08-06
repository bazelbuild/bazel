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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/** An immutable set of package name prefixes that should be ignored. */
@AutoCodec
public class IgnoredPackagePrefixesValue implements SkyValue {
  private final ImmutableSet<PathFragment> ignoredPrefixes;
  private final ImmutableSet<String> wildcardPatterns;

  @AutoCodec @AutoCodec.VisibleForSerialization
  public static final IgnoredPackagePrefixesValue EMPTY_LIST =
      new IgnoredPackagePrefixesValue(ImmutableSet.of());

  private IgnoredPackagePrefixesValue(ImmutableSet<PathFragment> patterns) {
    Preconditions.checkNotNull(patterns);

    ImmutableSet.Builder<String> wildcardPatternsBuilder = ImmutableSet.builder();
    ImmutableSet.Builder<PathFragment> ignoredPrefixesBuilder = ImmutableSet.builder();

    for (PathFragment pattern: patterns) {
      String path = pattern.getPathString();
      if (UnixGlob.isWildcardFree(path)) {
        ignoredPrefixesBuilder.add(pattern);
      } else {
        wildcardPatternsBuilder.add(path);
      }
    }

    wildcardPatterns = wildcardPatternsBuilder.build();
    ignoredPrefixes = ignoredPrefixesBuilder.build();
  }

  @AutoCodec.Instantiator
  public static IgnoredPackagePrefixesValue of(ImmutableSet<PathFragment> patterns) {
    return patterns.isEmpty() ? EMPTY_LIST : new IgnoredPackagePrefixesValue(patterns);
  }

  /** Creates a key from the main repository. */
  public static SkyKey key() {
    return Key.create(RepositoryName.MAIN);
  }

  /** Creates a key from the given repository name. */
  public static SkyKey key(RepositoryName repository) {
    return Key.create(repository);
  }

  public ImmutableSet<PathFragment> getPatterns() {
    return ignoredPrefixes;
  }

  public boolean isPathFragmentIgnored(PathFragment path) {
    for (PathFragment ignoredPrefix : ignoredPrefixes) {
      if (path.startsWith(ignoredPrefix)) {
        return true;
      }
    }
    for (String pattern : wildcardPatterns) {
      if (UnixGlob.matches(pattern + "/**", path.getPathString(), null)) {
        return true;
      }
    }
    return false;
  }

  @Override
  public int hashCode() {
    return ignoredPrefixes.hashCode() ;
  }

  @Override
  public boolean equals(Object obj) {
    if (obj instanceof IgnoredPackagePrefixesValue) {
      IgnoredPackagePrefixesValue other = (IgnoredPackagePrefixesValue) obj;
      return this.ignoredPrefixes.equals(other.ignoredPrefixes) && this.wildcardPatterns.equals(other.wildcardPatterns);
    }
    return false;
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class Key extends AbstractSkyKey<RepositoryName> {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    private Key(RepositoryName arg) {
      super(arg);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static Key create(RepositoryName arg) {
      return interner.intern(new Key(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.IGNORED_PACKAGE_PREFIXES;
    }
  }
}
