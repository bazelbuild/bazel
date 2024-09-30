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
package com.google.devtools.build.lib.collect;


import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A thread-safe PathFragment segment-based trie for inclusion checks.
 *
 * <p>The {@code put} operation is synchronized on the object, whereas the {@code includes}
 * retrieval operation do not block, and may overlap with {@code} put, and will reflect the results
 * of the most recently completed {@code put} operation. That is, if a {@code put} overlaps with an
 * {@code includes}, {@code includes} will return consistent results, either with the state before
 * or after the {@code put}.
 */
@ThreadSafe
public final class PathFragmentPrefixTrie {

  private final Set<PathFragment> includedPaths = new HashSet<>();
  private final Set<PathFragment> excludedPaths = new HashSet<>();

  private abstract static sealed class Segment
      permits InterimSegment, ExcludedSegment, IncludedSegment {
    private final Map<String, Segment> segmentMap;

    private Segment() {
      this.segmentMap = new ConcurrentHashMap<>();
    }

    private Segment(Map<String, Segment> segmentMap) {
      this.segmentMap = segmentMap;
    }

    Map<String, Segment> getSegmentMap() {
      return segmentMap;
    }
  }

  /** An interim segment. This segment has not been explicitly marked as included or excluded. */
  private static final class InterimSegment extends Segment {}

  /** A segment that has been explicitly marked as excluded. */
  private static final class ExcludedSegment extends Segment {
    private ExcludedSegment() {
      super();
    }

    private ExcludedSegment(Map<String, Segment> segmentMap) {
      super(segmentMap);
    }
  }

  /** A segment that has been explicitly marked as included. */
  private static final class IncludedSegment extends Segment {
    private IncludedSegment() {
      super();
    }

    private IncludedSegment(Map<String, Segment> segmentMap) {
      super(segmentMap);
    }
  }

  private Segment root;

  public PathFragmentPrefixTrie() {
    root = new InterimSegment();
  }

  public static PathFragmentPrefixTrie of(Collection<String> paths) {
    PathFragmentPrefixTrie trie = new PathFragmentPrefixTrie();
    paths.forEach(
        p -> {
          if (p.startsWith("-")) {
            // Exclusion
            trie.put(PathFragment.create(p.substring(1)), false);
          } else {
            // Inclusion
            trie.put(PathFragment.create(p), true);
          }
        });
    return trie;
  }

  public static ImmutableMap<String, PathFragmentPrefixTrie> transformValues(
      Map<String, Collection<String>> map) {
    return ImmutableMap.copyOf(Maps.transformValues(map, PathFragmentPrefixTrie::of));
  }

  /** Puts the explicit inclusion or exclusion state for a {@link PathFragment} into the trie. */
  public synchronized void put(PathFragment pathFragment, boolean included) {
    Preconditions.checkArgument(
        !pathFragment.equals(PathFragment.EMPTY_FRAGMENT),
        "path fragment cannot be the empty fragment.");

    if (included) {
      includedPaths.add(pathFragment);
    } else {
      excludedPaths.add(pathFragment);
    }

    Segment current = root;

    Iterator<String> segments = pathFragment.segments().iterator();
    while (segments.hasNext()) {
      String nextSegment = segments.next();
      if (segments.hasNext()) {
        current =
            current
                .getSegmentMap()
                .computeIfAbsent(nextSegment.intern(), unused -> new InterimSegment());
        continue;
      }

      // This is the last segment.
      Segment newChild =
          switch (current.getSegmentMap().get(nextSegment)) {
            case InterimSegment segment ->
                included
                    ? new IncludedSegment(segment.getSegmentMap())
                    : new ExcludedSegment(segment.getSegmentMap());
            case null -> included ? new IncludedSegment() : new ExcludedSegment();
            case ExcludedSegment unused ->
                throw new IllegalArgumentException(
                    pathFragment + " has already been explicitly marked as excluded.");
            case IncludedSegment unused ->
                throw new IllegalArgumentException(
                    pathFragment + " has already been explicitly marked as included.");
          };
      current.getSegmentMap().put(nextSegment.intern(), newChild);
    }
  }

  /**
   * Checks if a PathFragment is included, after applying exclusion checks.
   *
   * <p>If there is an exact match, its inclusion state will be returned.
   *
   * <p>Otherwise, the result corresponds to the longest prefix's inclusion state explicitly defined
   * in the trie. If the state is inconclusive (i.e. none of its ancestors are explicitly defined),
   * then the default is false / excluded.
   */
  public boolean includes(PathFragment pathFragment) {
    if (pathFragment.equals(PathFragment.EMPTY_FRAGMENT)) {
      return false;
    }

    Segment current = root;
    Segment lastSegment = current;

    for (String nextSegment : pathFragment.segments()) {
      current = current.getSegmentMap().get(nextSegment);
      if (current == null) {
        break;
      }

      if (!(current instanceof InterimSegment)) {
        lastSegment = current; // either Included or Excluded
      }
    }
    return lastSegment instanceof IncludedSegment;
  }

  @Override
  public String toString() {
    return "[included: "
        + includedPaths.stream().sorted().toList()
        + ", excluded: "
        + excludedPaths.stream().sorted().toList()
        + "]";
  }
}
