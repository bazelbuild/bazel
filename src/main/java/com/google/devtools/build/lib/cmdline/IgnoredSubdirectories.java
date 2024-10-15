// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.cmdline;

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * A set of subdirectories to ignore during target pattern matching or globbing.
 *
 * <p>This is currently just a prefix, but will eventually support glob-style wildcards.
 */
public final class IgnoredSubdirectories {
  private final ImmutableSet<PathFragment> prefixes;

  public IgnoredSubdirectories(ImmutableSet<PathFragment> prefixes) {
    this.prefixes = prefixes;
  }

  public ImmutableSet<PathFragment> prefixes() {
    return prefixes;
  }

  public static final IgnoredSubdirectories EMPTY = new IgnoredSubdirectories(ImmutableSet.of());

  /**
   * Checks whether every path in this instance can conceivably match something under {@code
   * directory}.
   */
  public boolean allPathsAreUnder(PathFragment directory) {
    for (PathFragment prefix : prefixes) {
      if (!prefix.startsWith(directory)) {
        return false;
      }

      if (prefix.equals(directory)) {
        return false;
      }
    }

    return true;
  }

  /** Returns the entry that matches a given directory or {@code null} if none. */
  @Nullable
  public PathFragment matchingEntry(PathFragment directory) {
    for (PathFragment prefix : prefixes) {
      if (directory.startsWith(prefix)) {
        return prefix;
      }
    }

    return null;
  }

  /** Filters out entries that cannot match anything under {@code directory}. */
  public IgnoredSubdirectories filterForDirectory(PathFragment directory) {
    ImmutableSet<PathFragment> filteredPrefixes =
        prefixes.stream().filter(p -> p.startsWith(directory)).collect(toImmutableSet());

    return new IgnoredSubdirectories(filteredPrefixes);
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof IgnoredSubdirectories)) {
      return false;
    }

    IgnoredSubdirectories that = (IgnoredSubdirectories) other;
    return Objects.equals(this.prefixes, that.prefixes);
  }

  @Override
  public int hashCode() {
    return prefixes.hashCode();
  }
}
