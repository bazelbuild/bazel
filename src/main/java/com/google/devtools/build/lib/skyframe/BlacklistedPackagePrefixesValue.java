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
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/** An immutable set of package name prefixes that should be blacklisted. */
@AutoCodec
public class BlacklistedPackagePrefixesValue implements SkyValue {
  public static final BlacklistedPackagePrefixesValue EMPTY =
      new BlacklistedPackagePrefixesValue(ImmutableSet.of());

  private final ImmutableSet<PathFragment> patterns;

  @AutoCodec.VisibleForSerialization @AutoCodec
  static final SkyKey BLACKLIST_KEY = () -> SkyFunctions.BLACKLISTED_PACKAGE_PREFIXES;

  public BlacklistedPackagePrefixesValue(ImmutableSet<PathFragment> patterns) {
    this.patterns = Preconditions.checkNotNull(patterns);
  }

  public static SkyKey key() {
    return BLACKLIST_KEY;
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

  public boolean isUnderBlacklisted(final RootedPath rootedPath) {
    for (PathFragment pattern : getPatterns()) {
      if (startsWithFragment(rootedPath, pattern)) {
        return Boolean.TRUE;
      }
    }
    return Boolean.FALSE;
  }

  private static boolean startsWithFragment(RootedPath rootedPath, PathFragment fragment) {
    return rootedPath.getRootRelativePath().startsWith(fragment) ||
        fragment.isAbsolute() && rootedPath.asPath().asFragment().startsWith(fragment);
  }
}
