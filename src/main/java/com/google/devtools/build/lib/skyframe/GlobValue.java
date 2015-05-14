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
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.PackageIdentifier;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/** A value corresponding to a glob. */
@Immutable
@ThreadSafe
public final class GlobValue implements SkyValue {

  static final GlobValue EMPTY = new GlobValue(
      NestedSetBuilder.<PathFragment>emptySet(Order.STABLE_ORDER));

  private final NestedSet<PathFragment> matches;

  GlobValue(NestedSet<PathFragment> matches) {
    this.matches = Preconditions.checkNotNull(matches);
  }

  /**
   * Returns glob matches.
   */
  public NestedSet<PathFragment> getMatches() {
    return matches;
  }

  @Override
  public boolean equals(Object other) {
    if (other == this) {
      return true;
    }
    if (!(other instanceof GlobValue)) {
      return false;
    }
    // shallowEquals() may fail to detect that two equivalent (according to toString())
    // NestedSets are equal, but will always detect when two NestedSets are different.
    // This makes this implementation of equals() overly strict, but we only call this
    // method when doing change pruning, which can accept false negatives.
    return getMatches().shallowEquals(((GlobValue) other).getMatches());
  }

  @Override
  public int hashCode() {
    return matches.shallowHashCode();
  }

  /**
   * Constructs a {@link SkyKey} for a glob lookup. {@code packageName} is assumed to be an
   * existing package. Trying to glob into a non-package is undefined behavior.
   *
   * @throws InvalidGlobPatternException if the pattern is not valid.
   */
  @ThreadSafe
  public static SkyKey key(PackageIdentifier packageId, String pattern, boolean excludeDirs,
      PathFragment subdir)
      throws InvalidGlobPatternException {
    if (pattern.indexOf('?') != -1) {
      throw new InvalidGlobPatternException(pattern, "wildcard ? forbidden");
    }

    String error = UnixGlob.checkPatternForError(pattern);
    if (error != null) {
      throw new InvalidGlobPatternException(pattern, error);
    }

    return internalKey(packageId, subdir, pattern, excludeDirs);
  }

  /**
   * Constructs a {@link SkyKey} for a glob lookup.
   *
   * <p>Do not use outside {@code GlobFunction}.
   */
  @ThreadSafe
  static SkyKey internalKey(PackageIdentifier packageId, PathFragment subdir, String pattern,
      boolean excludeDirs) {
    return new SkyKey(SkyFunctions.GLOB,
        new GlobDescriptor(packageId, subdir, pattern, excludeDirs));
  }

  /**
   * Constructs a {@link SkyKey} for a glob lookup.
   *
   * <p>Do not use outside {@code GlobFunction}.
   */
  @ThreadSafe
  static SkyKey internalKey(GlobDescriptor glob, String subdirName) {
    return internalKey(glob.packageId, glob.subdir.getRelative(subdirName),
        glob.pattern, glob.excludeDirs);
  }

  /**
   * An exception that indicates that a glob pattern is syntactically invalid.
   */
  @ThreadSafe
  public static final class InvalidGlobPatternException extends Exception {
    private final String pattern;

    InvalidGlobPatternException(String pattern, String error) {
      super(error);
      this.pattern = pattern;
    }

    @Override
    public String toString() {
      return String.format("invalid glob pattern '%s': %s", pattern, getMessage());
    }
  }
}
