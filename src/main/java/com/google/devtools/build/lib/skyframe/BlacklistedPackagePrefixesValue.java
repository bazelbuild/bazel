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
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/** An immutable set of package name prefixes that should be blacklisted. */
@AutoCodec
public class BlacklistedPackagePrefixesValue implements SkyValue {
  private final ImmutableSet<PathFragment> patterns;

  @AutoCodec @AutoCodec.VisibleForSerialization
  public static final BlacklistedPackagePrefixesValue NO_BLACKLIST =
      new BlacklistedPackagePrefixesValue(ImmutableSet.of());

  private BlacklistedPackagePrefixesValue(ImmutableSet<PathFragment> patterns) {
    this.patterns = Preconditions.checkNotNull(patterns);
  }

  @AutoCodec.Instantiator
  public static BlacklistedPackagePrefixesValue of(ImmutableSet<PathFragment> patterns) {
    return patterns.isEmpty() ? NO_BLACKLIST : new BlacklistedPackagePrefixesValue(patterns);
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
    return patterns;
  }

  @Override
  public int hashCode() {
    return patterns.hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if (obj instanceof BlacklistedPackagePrefixesValue) {
      BlacklistedPackagePrefixesValue other = (BlacklistedPackagePrefixesValue) obj;
      return this.patterns.equals(other.patterns);
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
      return SkyFunctions.BLACKLISTED_PACKAGE_PREFIXES;
    }
  }
}
