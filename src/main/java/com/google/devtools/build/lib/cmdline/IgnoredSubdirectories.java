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

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.ObjectArrays;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.UnixGlob;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * A set of subdirectories to ignore during target pattern matching or globbing.
 *
 * <p>This is currently just a prefix, but will eventually support glob-style wildcards.
 */
public final class IgnoredSubdirectories {
  public static final IgnoredSubdirectories EMPTY = new IgnoredSubdirectories(
      ImmutableSet.of(), ImmutableList.of(), ImmutableList.of());

  private static final Splitter SLASH_SPLITTER = Splitter.on("/");

  private final ImmutableSet<PathFragment> prefixes;

  // String[] is mutable; we keep the split version because that's faster to match and the non-split
  // one because that allows for simpler equality checking and then matchingEntry() doesn't need to
  // allocate new objects.
  private final ImmutableList<String> patterns;
  private final ImmutableList<String[]> splitPatterns;

  private IgnoredSubdirectories(
      ImmutableSet<PathFragment> prefixes,
      ImmutableList<String> patterns,
      ImmutableList<String[]> splitPatterns) {
    Preconditions.checkArgument(patterns.size() == splitPatterns.size());

    this.prefixes = prefixes;
    this.patterns = patterns;
    this.splitPatterns = splitPatterns;
  }

  public static IgnoredSubdirectories of(ImmutableSet<PathFragment> prefixes) {
    return of(prefixes, ImmutableList.of());
  }

  public static IgnoredSubdirectories of(ImmutableSet<PathFragment> prefixes, ImmutableList<String> patterns) {
    if (prefixes.isEmpty() && patterns.isEmpty()) {
      return EMPTY;
    }

    for (PathFragment prefix : prefixes) {
      Preconditions.checkArgument(!prefix.isAbsolute());
    }

    ImmutableList<String[]> splitPatterns = patterns.stream()
        .map(p -> Iterables.toArray(SLASH_SPLITTER.split(p), String.class))
        .collect(ImmutableList.toImmutableList());

    return new IgnoredSubdirectories(prefixes, patterns, splitPatterns);
  }

  public IgnoredSubdirectories withPrefix(PathFragment prefix) {
    Preconditions.checkArgument(!prefix.isAbsolute());

    ImmutableSet<PathFragment> prefixedPrefixes =
        prefixes.stream().map(prefix::getRelative).collect(toImmutableSet());

    ImmutableList<String> prefixedPatterns = patterns.stream()
        .map(p -> prefix + "/" + p)
        .collect(ImmutableList.toImmutableList());

    String[] splitPrefix = Iterables.toArray(prefix.segments(), String.class);
    ImmutableList<String[]> prefixedSplitPatterns = splitPatterns.stream()
        .map(p -> ObjectArrays.concat(splitPrefix, p, String.class))
        .collect(ImmutableList.toImmutableList());

    return new IgnoredSubdirectories(prefixedPrefixes, prefixedPatterns, prefixedSplitPatterns);
  }

  public IgnoredSubdirectories union(IgnoredSubdirectories other) {
    return new IgnoredSubdirectories(
        ImmutableSet.<PathFragment>builder().addAll(prefixes).addAll(other.prefixes).build(),
        ImmutableList.<String>builder().addAll(patterns).addAll(other.patterns).build(),
        ImmutableList.<String[]>builder().addAll(splitPatterns).addAll(other.splitPatterns).build());
  }

  /** Filters out entries that cannot match anything under {@code directory}. */
  public IgnoredSubdirectories filterForDirectory(PathFragment directory) {
    ImmutableSet<PathFragment> filteredPrefixes =
        prefixes.stream().filter(p -> p.startsWith(directory)).collect(toImmutableSet());

    String[] directorySegments = Iterables.toArray(directory.segments(), String.class);

    ImmutableList.Builder<String> filteredPatterns = ImmutableList.builder();
    ImmutableList.Builder<String[]> filteredSplitPatterns = ImmutableList.builder();
    for (int i = 0; i < patterns.size(); i++) {
      if (UnixGlob.canMatchChild(splitPatterns.get(i), directorySegments)){
        filteredPatterns.add(patterns.get(i));
        filteredSplitPatterns.add(splitPatterns.get(i));
      }
    }
    return new IgnoredSubdirectories(filteredPrefixes, filteredPatterns.build(), filteredSplitPatterns.build());
  }

  public ImmutableSet<PathFragment> prefixes() {
    return prefixes;
  }

  public boolean isEmpty() {
    return this.prefixes.isEmpty();
  }

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
  public String matchingEntry(PathFragment directory) {
    for (PathFragment prefix : prefixes) {
      if (directory.startsWith(prefix)) {
        return prefix.getPathString();
      }
    }

    String[] segmentArray = Iterables.toArray(directory.segments(), String.class);
    for (int i = 0; i < patterns.size(); i++) {
      if (UnixGlob.matchesPrefix(splitPatterns.get(i), segmentArray)) {
        return patterns.get(i);
      }
    }

    return null;
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof IgnoredSubdirectories)) {
      return false;
    }

    // splitPatterns is a function of patterns so it's enough to check if patterns is equal
    IgnoredSubdirectories that = (IgnoredSubdirectories) other;
    return Objects.equals(this.prefixes, that.prefixes) && Objects.equals(this.patterns, that.patterns);
  }

  @Override
  public int hashCode() {
    return Objects.hash(prefixes, patterns);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper("IgnoredSubdirectories")
        .add("prefixes", prefixes)
        .add("patterns", patterns)
        .toString();
  }
}
