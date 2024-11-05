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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * A set of subdirectories to ignore during target pattern matching or globbing.
 */
public final class IgnoredSubdirectories {
  public static final IgnoredSubdirectories EMPTY =
      new IgnoredSubdirectories(ImmutableSet.of(), ImmutableList.of());

  private static final Splitter SLASH_SPLITTER = Splitter.on("/");

  private final ImmutableSet<PathFragment> prefixes;

  // String[] is mutable; we keep the split version because that's faster to match and the non-split
  // one because that allows for simpler equality checking and then matchingEntry() doesn't need to
  // allocate new objects.
  private final ImmutableList<String> patterns;
  private final ImmutableList<String[]> splitPatterns;

  private static class Codec implements ObjectCodec<IgnoredSubdirectories> {
    private static final Codec INSTANCE = new Codec();

    @Override
    public Class<? extends IgnoredSubdirectories> getEncodedClass() {
      return IgnoredSubdirectories.class;
    }

    @Override
    public void serialize(
        SerializationContext context, IgnoredSubdirectories obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serialize(obj.prefixes, codedOut);
      context.serialize(obj.patterns, codedOut);
    }

    @Override
    public IgnoredSubdirectories deserialize(
        DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      ImmutableSet<PathFragment> prefixes = context.deserialize(codedIn);
      ImmutableList<String> patterns = context.deserialize(codedIn);

      return new IgnoredSubdirectories(prefixes, patterns);
    }
  }

  private IgnoredSubdirectories(
      ImmutableSet<PathFragment> prefixes, ImmutableList<String> patterns) {
    this.prefixes = prefixes;
    this.patterns = patterns;
    this.splitPatterns =
        patterns.stream()
            .map(p -> Iterables.toArray(SLASH_SPLITTER.split(p), String.class))
            .collect(toImmutableList());
  }

  public static IgnoredSubdirectories of(ImmutableSet<PathFragment> prefixes) {
    return of(prefixes, ImmutableList.of());
  }

  public static IgnoredSubdirectories of(
      ImmutableSet<PathFragment> prefixes, ImmutableList<String> patterns) {
    if (prefixes.isEmpty() && patterns.isEmpty()) {
      return EMPTY;
    }

    for (PathFragment prefix : prefixes) {
      Preconditions.checkArgument(!prefix.isAbsolute());
    }

    return new IgnoredSubdirectories(prefixes, patterns);
  }

  public IgnoredSubdirectories withPrefix(PathFragment prefix) {
    Preconditions.checkArgument(!prefix.isAbsolute());

    ImmutableSet<PathFragment> prefixedPrefixes =
        prefixes.stream().map(prefix::getRelative).collect(toImmutableSet());

    ImmutableList<String> prefixedPatterns =
        patterns.stream().map(p -> prefix + "/" + p).collect(toImmutableList());

    return new IgnoredSubdirectories(prefixedPrefixes, prefixedPatterns);
  }

  public IgnoredSubdirectories union(IgnoredSubdirectories other) {
    return new IgnoredSubdirectories(
        ImmutableSet.<PathFragment>builder().addAll(prefixes).addAll(other.prefixes).build(),
        ImmutableList.copyOf(
            ImmutableSet.<String>builder().addAll(patterns).addAll(other.patterns).build()));
  }

  /** Filters out entries that cannot match anything under {@code directory}. */
  public IgnoredSubdirectories filterForDirectory(PathFragment directory) {
    ImmutableSet<PathFragment> filteredPrefixes =
        prefixes.stream().filter(p -> p.startsWith(directory)).collect(toImmutableSet());

    String[] splitDirectory =
        Iterables.toArray(SLASH_SPLITTER.split(directory.getPathString()), String.class);
    ImmutableList.Builder<String> filteredPatterns = ImmutableList.builder();
    for (int i = 0; i < patterns.size(); i++) {
      if (UnixGlob.canMatchChild(splitPatterns.get(i), splitDirectory)) {
        filteredPatterns.add(patterns.get(i));
      }
    }

    return new IgnoredSubdirectories(filteredPrefixes, filteredPatterns.build());
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
    return Objects.equals(this.prefixes, that.prefixes)
        && Objects.equals(this.patterns, that.patterns);
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
