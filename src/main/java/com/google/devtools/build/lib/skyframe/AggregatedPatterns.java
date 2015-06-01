// Copyright 2015 Google Inc. All rights reserved.
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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.cmdline.TargetPattern.Type;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternKey;

import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.NavigableMap;
import java.util.Set;
import java.util.TreeMap;

/**
 * This is used by {@link TargetPatternValue#keys} to help it convert a sequence of target patterns
 * into a sequence of {@link TargetPatternKey}s that, when evaluated and combined together, yield
 * the targets matched by the target pattern sequence.
 *
 * <p>Its main strategy for accomplishing this is to sort the target patterns by the directories
 * that contain the targets they match, and take advantage of that sorting to efficiently discover
 * redundant patterns (e.g. the two patterns "//foo/..." and "//foo/bar/..." don't each need to be
 * evaluated if they appear side-by-side in the sequence) and exclusionary patterns (e.g. the two
 * patterns "//foo/..." and "-//foo/bar/..." don't each need to be evaluated--instead, while
 * evaluating the targets below "foo", the subdirectory "bar" can be skipped).
 *
 * <p>An {@link AggregatedPatterns} instance expects its {@link #addPattern} method to be called
 * for each target pattern in the sequence in a left-to-right order. This is because a pattern of
 * type {@code Type.TARGETS_BELOW_DIRECTORY} overrides everything encountered so far in and below
 * the directory it specifies, and this class implements that by removing all target patterns
 * that were already added in and below that directory.
 *
 * <p>The class takes a {@link FilteringPolicy} and a ({@link String}) offset in its constructor, as
 * the {@link TargetPatternKey}s it constructs from a target pattern sequence include those values.
 *
 * <p>The creation of {@link TargetPatternKey}s begins when {@link AggregatedPatterns#build} is
 * called. It processes the sorted target patterns recursively, breaking the problem of
 * converting the sequence of target patterns to a sequence of {@link TargetPatternKey}s into a
 * set of sub-problems, where each sub-problem consists of converting the patterns beneath each
 * {@code Type.TARGETS_BELOW_DIRECTORY} pattern into a sequence of {@link TargetPatternKey}s.
 * This gets off the ground because one can think of every target pattern sequence as implicitly
 * starting with the "-//..." pattern.
 */
class AggregatedPatterns {

  // Keys in this ordered map are strings representing directories. These strings are either "" (to
  // indicate the depot root) or end with a trailing '/'.
  //
  // They end with '/' to ensure that an iteration through the map corresponds to a depth-first
  // traversal through the directory tree, and that an entry for a directory directly precedes the
  // entries for its subdirectories.
  //
  // For example, consider the three directories "a/", "a/b/", and "a.1/". When sorted into the
  // map, their entries will be visited in this order:
  //
  //     "a.1/"
  //     "a/"
  //     "a/b/"
  //
  // If the keys were directories in normal form (i.e. lacking a trailing '/') they would be
  // visited in this order:
  //
  //     "a"
  //     "a.1"
  //     "a/b"
  //
  // (Note how "a" and "a/b" are sadly separated in this latter example.)
  //
  // If {@code l = orderedPatternsByDirectory.get(d)}, then {@code l} is the list of all target
  // patterns {@code p} such that {@code maybeAddTrailingSlash(p.getDirectory()).equals(d)}, in
  // order of insertion via {@link AggregatedPatterns#addPattern} (which is assumed to correspond
  // to logic order in the target pattern sequence), which were not inserted before any
  // {@code Type.TARGETS_BELOW_DIRECTORY} patterns with directories equal to or above {@code d}
  // (because later patterns take precedence over earlier patterns).
  private final NavigableMap<String, List<SignedPattern>> orderedPatternsByDirectory =
      new TreeMap<>();
  private final FilteringPolicy policy;
  private final String offset;

  AggregatedPatterns(FilteringPolicy policy, String offset) {
    this.policy = policy;
    this.offset = offset;
  }

  /** Must be called for each target pattern in the sequence in order from left to right. */
  AggregatedPatterns addPattern(SignedPattern signedPattern) {
    Preconditions.checkNotNull(signedPattern);
    String directoryWithoutTrailingSlash = signedPattern.getPattern().getDirectory();

    // If the new pattern is a TargetsBelowDirectory, then it overrides everything in and below its
    // directory, so we can remove those keys.
    if (isTargetsBelowDirectory(signedPattern)) {
      // The set returned by SortedMap.keySet supports element removal via the clear operation.
      getDirectoriesBelow(directoryWithoutTrailingSlash).clear();
    }

    // If the pattern's directory is non-empty, we add a trailing "/" to the directory key. See the
    // comment on orderedPatternsByDirectory for the reason why.
    String directoryKey = getDirectoryKey(directoryWithoutTrailingSlash);
    List<SignedPattern> patternsMaybe = orderedPatternsByDirectory.get(directoryKey);
    if (patternsMaybe == null) {
      patternsMaybe = Lists.newArrayList();
      orderedPatternsByDirectory.put(directoryKey, patternsMaybe);
    }
    patternsMaybe.add(signedPattern);
    return this;
  }

  Iterable<TargetPatternKey> build() {
    return getKeysAndExcludedDirsUnder(orderedPatternsByDirectory, false).getKeys();
  }

  private enum DirectoryInclusion {
    INCLUDED,
    EXCLUDED,
    MIXED;

    static DirectoryInclusion from(boolean isIncluded) {
      return isIncluded ? INCLUDED : EXCLUDED;
    }
  }

  private static class KeysAndExcludedDirectories {
    private final ImmutableList<TargetPatternKey> keys;
    private final ImmutableSet<String> excludedDirectories;

    private KeysAndExcludedDirectories(ImmutableList<TargetPatternKey> keys,
        ImmutableSet<String> excludedDirectories) {
      this.keys = Preconditions.checkNotNull(keys);
      this.excludedDirectories = Preconditions.checkNotNull(excludedDirectories);
    }

    public ImmutableList<TargetPatternKey> getKeys() {
      return keys;
    }

    public ImmutableSet<String> getExcludedDirectories() {
      return excludedDirectories;
    }
  }

  private static class SubdirectoriesAndRemainder {
    private final NavigableMap<String, List<SignedPattern>> subdirectoriesMap;
    private final Iterator<Entry<String, List<SignedPattern>>> remainder;

    public SubdirectoriesAndRemainder(
        NavigableMap<String, List<SignedPattern>> subdirectoriesMap,
        Iterator<Entry<String, List<SignedPattern>>> remainder) {
      this.subdirectoriesMap = Preconditions.checkNotNull(subdirectoriesMap);
      this.remainder = Preconditions.checkNotNull(remainder);
    }

    public NavigableMap<String, List<SignedPattern>> getSubdirectoriesMap() {
      return subdirectoriesMap;
    }

    public Iterator<Entry<String, List<SignedPattern>>> getRemainder() {
      return remainder;
    }
  }

  /**
   * Calculates a list of {@link TargetPatternKey}s and a set of excluded directories
   * (represented by {@link String}s in normal form, relative to the workspace root)
   * corresponding to the {@link SignedPattern}s included in {@code directories}.
   *
   * <p>The function tries to eliminate redundant {@link TargetPatternKey}s from the returned value.
   * To help it discover these redundancies it must be told whether targets in
   * {@code directories} are considered included by default, via {@code includedByDefault}. For
   * example, if all the directories in {@code directories} are subdirectories below "some/path",
   * and we had already seen a target pattern corresponding to "//some/path/...", then {@code
   * includedByDefault} would be true.
   *
   * @param directories an ordered map of directories to ordered lists of target patterns,
   *     sharing the same requirements as {@code orderedPatternsByDirectory}
   * @param includedByDefault whether targets in {@code directories} should be considered
   *     included by default
   */
  private KeysAndExcludedDirectories getKeysAndExcludedDirsUnder(
      NavigableMap<String, List<SignedPattern>> directories, boolean includedByDefault) {
    ImmutableList.Builder<TargetPatternKey> keysBuilder = ImmutableList.builder();
    ImmutableSet.Builder<String> excludedDirsBuilder = ImmutableSet.builder();

    // The following {@code iterator} variable is used to iterate through the directories in
    // {@code orderedPatternsByDirectory}. When it encounters a pattern of type
    // {@code Type.TARGETS_BELOW_DIRECTORY} in a directory, it recursively calls itself to evaluate
    // the interval of entries corresponding to the subdirectories beneath that directory.
    // Afterwards, it skips past that already-processed interval by *assigning to*
    // {@code iterator}, to make it point at the first element after the interval (or to an "empty"
    // iterator if there is no such element). That assignment is why this while loop isn't a
    // standard "for(T e : iterable)" loop.
    Iterator<Entry<String, List<SignedPattern>>> iterator = directories.entrySet().iterator();
    while (iterator.hasNext()) {
      Entry<String, List<SignedPattern>> directoryEntry = iterator.next();
      // The following {@code directoryInclusion} variable represents whether targets in this
      // directory are included by default. It's initialized to INCLUDED or EXCLUDED based on the
      // value of {@code includedByDefault}. It will be set to INCLUDED if we encounter a positive
      // TargetsBelowDirectory (TBD) pattern in the directory, EXCLUDED if we encounter a
      // negative TBD pattern in the directory, and will be set to MIXED if we encounter a
      // non-TBD pattern to be subtracted while in the INCLUDED state or if we encounter a
      // non-TBD pattern to be added while in the EXCLUDED state. (Note that only the first
      // pattern in the directory can be a TBD pattern because adding a TBD pattern removes all
      // existing patterns in and below its directory.)
      //
      // It's used to discover redundant patterns, i.e. patterns that can be skipped because a
      // prior pattern already includes it (and it's positive) or because a prior pattern
      // already excluded it (and it's negative).
      //
      // Currently, the only kind of prior patterns that this code is sensitive to are those of
      // type TBD. This code doesn't keep track of targets in a more fine-grained way for
      // simplicity. (See the final comment in the
      // TargetPatternValueTest#testSubtractingThenAddingInIncludedDirectory unit test for more
      // information.) Still, this approach offers the benefit of skipping positive targets under
      // a positive TBD pattern and negative targets under a negative TBD pattern.
      DirectoryInclusion directoryInclusion = DirectoryInclusion.from(includedByDefault);

      Iterable<SignedPattern> directoryPatterns = directoryEntry.getValue();
      for (SignedPattern signedPattern : directoryPatterns) {
        if (isTargetsBelowDirectory(signedPattern)) {
          if (!signedPattern.isPositive()) {
            // Add its directory to the excluded directories.
            excludedDirsBuilder.add(signedPattern.getPattern().getDirectory());
          }

          // If it's a TBD pattern with sign opposite from the current default inclusion policy,
          // then it isn't redundant, and will contribute to the returned value as either a
          // TargetPatternKey (if the pattern's sign is positive) or as an excluded directory (if
          // the pattern's sign is negative).
          if (signedPattern.isPositive() != includedByDefault) {
            // 1) Remember the new effective inclusion policy for this directory.
            directoryInclusion = DirectoryInclusion.from(signedPattern.isPositive());

            // 2) Collect any additional keys and excluded directories from subdirectories below
            // this one.
            SubdirectoriesAndRemainder subdirectoriesAndRemainder =
                getSubdirectoriesAndRemainder(directories, directoryEntry.getKey());
            NavigableMap<String, List<SignedPattern>> subMapForSubdirectories =
                subdirectoriesAndRemainder.getSubdirectoriesMap();
            KeysAndExcludedDirectories keysAndExcludedDirs =
                getKeysAndExcludedDirsUnder(subMapForSubdirectories,
                    /*includedByDefault=*/signedPattern.isPositive());

            if (signedPattern.isPositive()) {
              // 2.5) If this TBD pattern is positive, specify the excluded subdirectories in the
              // TargetPatternKey for this pattern and add the key to the keysBuilder.
              keysBuilder.add(createTargetPatternKey(signedPattern, policy, offset,
                  keysAndExcludedDirs.getExcludedDirectories()));
            }

            // 3) Include the keys collected from subdirectories.
            // Note that this statement means that the number of references this algorithm copies
            // is potentially quadratic. An adversary could engineer a target pattern sequence
            // that degrades performance.
            keysBuilder.addAll(keysAndExcludedDirs.getKeys());

            // 4) Skip past the keys from subdirectories below this one.
            iterator = subdirectoriesAndRemainder.getRemainder();
          }
          // Else, it's a TBD pattern with sign equal to the current default inclusion policy.
          // Skip it because it's redundant.
        } else { // It's not a TBD pattern.
          boolean subtractingFromIncludedDirectory =
              directoryInclusion.equals(DirectoryInclusion.INCLUDED) && !signedPattern.isPositive();
          boolean addingToExcludedDirectory =
              directoryInclusion.equals(DirectoryInclusion.EXCLUDED) && signedPattern.isPositive();
          boolean alreadyMixed = directoryInclusion.equals(DirectoryInclusion.MIXED);
          if (subtractingFromIncludedDirectory || addingToExcludedDirectory || alreadyMixed) {
            // If we're subtracting from an included directory, or adding to an excluded
            // directory, or if the directory is already mixed, add the pattern to the keysBuilder.
            keysBuilder.add(createTargetPatternKey(signedPattern, policy, offset));
            // Also remember that this directory is now both including and excluding some targets,
            // so all subsequent patterns must be evaluated.
            directoryInclusion = DirectoryInclusion.MIXED;
          }
        }
      }
    }
    return new KeysAndExcludedDirectories(keysBuilder.build(), excludedDirsBuilder.build());
  }

  private static String getDirectoryKey(String directoryWithoutTrailingSlash) {
    return directoryWithoutTrailingSlash.length() == 0 ? "" : directoryWithoutTrailingSlash + "/";
  }

  private static TargetPatternKey createTargetPatternKey(SignedPattern pattern,
      FilteringPolicy policy, String offset) {
    return createTargetPatternKey(pattern, policy, offset, ImmutableSet.<String>of());
  }

  private static TargetPatternKey createTargetPatternKey(SignedPattern signedPattern,
      FilteringPolicy policy, String offset, ImmutableSet<String> excludedDirectories) {
    if (signedPattern.isPositive()) {
      return new TargetPatternKey(signedPattern.getPattern(), policy, /*isNegative=*/false, offset,
          excludedDirectories);
    } else {
      return new TargetPatternKey(signedPattern.getPattern(), FilteringPolicies.NO_FILTER,
          /*isNegative=*/true, offset, excludedDirectories);
    }
  }

  private static SubdirectoriesAndRemainder getSubdirectoriesAndRemainder(
      NavigableMap<String, List<SignedPattern>> directories, String directoryKey) {
    NavigableMap<String, List<SignedPattern>> subdirectories;
    Iterator<Entry<String, List<SignedPattern>>> remainder;
    if (directoryKey.length() == 0) {
      subdirectories = directories.tailMap(directoryKey,
          /*inclusive, referring to the directoryKey endpoint=*/false);
      remainder = Collections.emptyIterator();
    } else {
      String afterKey = getKeyAfterAllPathsBelowDirectoryWithoutTrailingSlash(
          directoryKey.substring(0, directoryKey.length() - 1));
      subdirectories = directories.subMap(directoryKey,
          /*inclusive, referring to the directoryKey endpoint=*/false, afterKey,
          /*inclusive, referring to the afterKey endpoint=*/false);
      remainder = directories.tailMap(afterKey).entrySet().iterator();
    }
    return new SubdirectoriesAndRemainder(subdirectories, remainder);
  }

  private static String getKeyAfterAllPathsBelowDirectoryWithoutTrailingSlash(
      String directoryWithoutTrailingSlash) {
    // We calculate the least character greater than '/' (which happens to be '0'). We use this
    // to collect all paths in subdirectories of the current one, using NavigableMap's subMap
    // method, which takes an interval.
    return directoryWithoutTrailingSlash + ((char) ('/' + 1));
  }

  private Set<String> getDirectoriesBelow(String directoryWithoutTrailingSlash) {
    // If the empty string is passed in, collect all the directories.
    if (directoryWithoutTrailingSlash.length() == 0) {
      return orderedPatternsByDirectory.keySet();
    }
    // Otherwise, collect the keyset belonging to the passed-in directory and its subdirectories.
    String afterKey = getKeyAfterAllPathsBelowDirectoryWithoutTrailingSlash(
        directoryWithoutTrailingSlash);
    // Note that the subMap method defaults to interpreting the first, lesser, argument inclusively
    // and the second, greater, argument exclusively.
    return orderedPatternsByDirectory.subMap(getDirectoryKey(directoryWithoutTrailingSlash),
        afterKey).keySet();
  }

  private static boolean isTargetsBelowDirectory(SignedPattern signedPattern) {
    return signedPattern.getPattern().getType().equals(Type.TARGETS_BELOW_DIRECTORY);
  }
}
