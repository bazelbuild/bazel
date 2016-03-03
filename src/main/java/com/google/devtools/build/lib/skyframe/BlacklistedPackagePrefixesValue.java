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
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * An immutable set of package name prefixes that should be blacklisted.
 */
public class BlacklistedPackagePrefixesValue implements SkyValue {
  private final ImmutableSet<PathFragment> patterns;

  private static final SkyKey BLACKLIST_KEY =
      SkyKey.create(SkyFunctions.BLACKLISTED_PACKAGE_PREFIXES, "");

  public BlacklistedPackagePrefixesValue(ImmutableSet<PathFragment> exclusions) {
    this.patterns = Preconditions.checkNotNull(exclusions);
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
}
