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
import com.google.devtools.build.lib.util.StringCanonicalizer;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.Serializable;
import java.util.Objects;

/**
 * A value corresponding to a glob.
 */
@Immutable
@ThreadSafe
final class GlobValue implements SkyValue {

  static final GlobValue EMPTY = new GlobValue(
      NestedSetBuilder.<PathFragment>emptySet(Order.STABLE_ORDER));

  private final NestedSet<PathFragment> matches;

  GlobValue(NestedSet<PathFragment> matches) {
    this.matches = Preconditions.checkNotNull(matches);
  }

  /**
   * Returns glob matches.
   */
  NestedSet<PathFragment> getMatches() {
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
  static SkyKey key(PathFragment packageName, String pattern, boolean excludeDirs)
      throws InvalidGlobPatternException {
    if (pattern.indexOf('?') != -1) {
      throw new InvalidGlobPatternException(pattern, "wildcard ? forbidden");
    }

    String error = UnixGlob.checkPatternForError(pattern);
    if (error != null) {
      throw new InvalidGlobPatternException(pattern, error);
    }

    return internalKey(packageName, PathFragment.EMPTY_FRAGMENT, pattern, excludeDirs);
  }

  /**
   * Constructs a {@link SkyKey} for a glob lookup.
   *
   * <p>Do not use outside {@code GlobFunction}.
   */
  @ThreadSafe
  static SkyKey internalKey(PathFragment packageName, PathFragment subdir, String pattern,
      boolean excludeDirs) {
    return new SkyKey(SkyFunctions.GLOB,
        new GlobDescriptor(packageName, subdir, pattern, excludeDirs));
  }

  /**
   * Constructs a {@link SkyKey} for a glob lookup.
   *
   * <p>Do not use outside {@code GlobFunction}.
   */
  @ThreadSafe
  static SkyKey internalKey(GlobDescriptor glob, String subdirName) {
    return internalKey(glob.packageName, glob.subdir.getRelative(subdirName),
        glob.pattern, glob.excludeDirs);
  }

  /**
   * A descriptor for a glob request.
   *
   * <p>{@code subdir} must be empty or point to an existing directory.</p>
   *
   * <li>{@code pattern} must be valid, as indicated by {@code UnixGlob#checkPatternForError}.
   */
  @ThreadSafe
  static final class GlobDescriptor implements Serializable {
    private final PathFragment packageName;
    private final PathFragment subdir;
    private final String pattern;
    private final boolean excludeDirs;

    /**
     * Constructs a GlobDescriptor.
     *
     * @param packageName the name of the owner package (must be an existing package)
     * @param subdir the subdirectory being looked at (must exist and must be a directory. It's
     *               assumed that there are no other packages between {@code packageName} and
     *               {@code subdir}.
     * @param pattern a valid glob pattern
     * @param excludeDirs true if directories should be excluded from results
     */
    private GlobDescriptor(PathFragment packageName, PathFragment subdir, String pattern,
        boolean excludeDirs) {
      this.packageName = Preconditions.checkNotNull(packageName);
      this.subdir = Preconditions.checkNotNull(subdir);
      this.pattern = Preconditions.checkNotNull(StringCanonicalizer.intern(pattern));
      this.excludeDirs = excludeDirs;
    }

    @Override
    public String toString() {
      return String.format("<GlobDescriptor packageName=%s subdir=%s pattern=%s excludeDirs=%s>",
          packageName, subdir, pattern, excludeDirs);
    }

    /**
     * Returns the package that "owns" this glob.
     *
     * <p>The glob evaluation code ensures that the boundaries of this package are not crossed.
     */
    PathFragment getPackageName() {
      return packageName;
    }

    /**
     * Returns the subdirectory of the package under consideration.
     */
    PathFragment getSubdir() {
      return subdir;
    }

    /**
     * Returns the glob pattern under consideration. May contain wildcards.
     *
     * <p>As the glob evaluator traverses deeper into the file tree, components are added at the
     * beginning of {@code subdir} and removed from the beginning of {@code pattern}.
     */
    String getPattern() {
      return pattern;
    }

    /**
     * Returns true if directories should be excluded from results.
     */
    boolean excludeDirs() {
      return excludeDirs;
    }

    @Override
    public int hashCode() {
      return Objects.hash(packageName, subdir, pattern, excludeDirs);
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (obj == null) {
        return false;
      }
      if (!(obj instanceof GlobDescriptor)) {
        return false;
      }
      GlobDescriptor other = (GlobDescriptor) obj;
      return packageName.equals(other.packageName) && subdir.equals(other.subdir)
          && pattern.equals(other.pattern) && excludeDirs == other.excludeDirs;
    }
  }

  /**
   * An exception that indicates that a glob pattern is syntactically invalid.
   */
  @ThreadSafe
  static final class InvalidGlobPatternException extends Exception {
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
