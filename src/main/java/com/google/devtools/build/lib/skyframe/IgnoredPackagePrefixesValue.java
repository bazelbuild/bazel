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
import com.google.devtools.build.lib.cmdline.IgnoredSubdirectories;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/** An immutable set of package name prefixes that should be ignored. */
public class IgnoredPackagePrefixesValue implements SkyValue {
  private final IgnoredSubdirectories ignoredSubdirectories;

  @SerializationConstant @VisibleForSerialization
  public static final IgnoredPackagePrefixesValue EMPTY_LIST =
      new IgnoredPackagePrefixesValue(ImmutableSet.of());

  private IgnoredPackagePrefixesValue(ImmutableSet<PathFragment> patterns) {
    this.ignoredSubdirectories = IgnoredSubdirectories.of(Preconditions.checkNotNull(patterns));
  }

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

  public IgnoredSubdirectories asIgnoredSubdirectories() {
    return ignoredSubdirectories;
  }

  public ImmutableSet<PathFragment> getPatterns() {
    return ignoredSubdirectories.prefixes();
  }

  @Override
  public int hashCode() {
    return ignoredSubdirectories.hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if (obj instanceof IgnoredPackagePrefixesValue other) {
      return this.ignoredSubdirectories.equals(other.ignoredSubdirectories);
    }
    return false;
  }

  @Override
  public String toString() {
    return ignoredSubdirectories.toString();
  }

  @VisibleForSerialization
  @AutoCodec
  static class Key extends AbstractSkyKey<RepositoryName> {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    private Key(RepositoryName arg) {
      super(arg);
    }

    static Key create(RepositoryName arg) {
      return interner.intern(new Key(arg));
    }

    @VisibleForSerialization
    @AutoCodec.Interner
    static Key intern(Key key) {
      return interner.intern(key);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.IGNORED_PACKAGE_PREFIXES;
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }
}
