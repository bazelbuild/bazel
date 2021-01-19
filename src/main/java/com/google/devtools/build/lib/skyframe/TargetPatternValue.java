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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.cmdline.TargetPattern.TargetsBelowDirectory;
import com.google.devtools.build.lib.cmdline.TargetPattern.TargetsBelowDirectory.ContainsResult;
import com.google.devtools.build.lib.cmdline.TargetPattern.Type;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.supplier.InterruptibleSupplier;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * A value referring to a computed set of resolved targets. This is used for the results of target
 * pattern parsing.
 */
@Immutable
@ThreadSafe
public final class TargetPatternValue implements SkyValue {

  private ResolvedTargets<Label> targets;

  TargetPatternValue(ResolvedTargets<Label> targets) {
    this.targets = Preconditions.checkNotNull(targets);
  }

  private void writeObject(ObjectOutputStream out) throws IOException {
    List<String> ts = new ArrayList<>();
    List<String> filteredTs = new ArrayList<>();
    for (Label target : targets.getTargets()) {
      ts.add(target.toString());
    }
    for (Label target : targets.getFilteredTargets()) {
      filteredTs.add(target.toString());
    }

    out.writeObject(ts);
    out.writeObject(filteredTs);
  }

  private Label labelFromString(String labelString) {
    try {
      return Label.parseAbsolute(labelString, ImmutableMap.of());
    } catch (LabelSyntaxException e) {
      throw new IllegalStateException(e);
    }
  }

  @SuppressWarnings("unchecked")
  private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
    List<String> ts = (List<String>) in.readObject();
    List<String> filteredTs = (List<String>) in.readObject();

    ResolvedTargets.Builder<Label> builder = ResolvedTargets.<Label>builder();
    for (String labelString : ts) {
      builder.add(labelFromString(labelString));
    }

    for (String labelString : filteredTs) {
      builder.remove(labelFromString(labelString));
    }
    this.targets = builder.build();
  }

  @SuppressWarnings("unused")
  private void readObjectNoData() {
    throw new IllegalStateException();
  }

  /**
   * Create a target pattern {@link SkyKey}. Throws {@link TargetParsingException} if the provided
   * {@code pattern} cannot be parsed.
   *
   * @param pattern The pattern, eg "-foo/biz...". If the first character is "-", the pattern is
   *     treated as a negative pattern.
   * @param policy The filtering policy, eg "only return test targets"
   * @param offset The offset to apply to relative target patterns.
   */
  @ThreadSafe
  public static TargetPatternKey key(String pattern, FilteringPolicy policy, PathFragment offset)
      throws TargetParsingException {
    return Iterables.getOnlyElement(keys(ImmutableList.of(pattern), policy, offset)).getSkyKey();
  }

  /**
   * Returns an iterable of {@link TargetPatternSkyKeyOrException}, with {@link TargetPatternKey}
   * arguments, in the same order as the list of patterns provided as input. If a provided pattern
   * fails to parse, the element in the returned iterable corresponding to it will throw when its
   * {@link TargetPatternSkyKeyOrException#getSkyKey} method is called.
   *
   * @param patterns The list of patterns, eg "-foo/biz...". If a pattern's first character is "-",
   *     it is treated as a negative pattern.
   * @param policy The filtering policy, eg "only return test targets"
   * @param offset The offset to apply to relative target patterns.
   */
  @ThreadSafe
  public static Iterable<TargetPatternSkyKeyOrException> keys(
      List<String> patterns, FilteringPolicy policy, PathFragment offset) {
    TargetPattern.Parser parser = new TargetPattern.Parser(offset);
    ImmutableList.Builder<TargetPatternSkyKeyOrException> builder = ImmutableList.builder();
    for (String pattern : patterns) {
      boolean positive = !pattern.startsWith("-");
      String absoluteValueOfPattern = positive ? pattern : pattern.substring(1);
      TargetPattern targetPattern;
      try {
        targetPattern = parser.parse(absoluteValueOfPattern);
      } catch (TargetParsingException e) {
        builder.add(new TargetPatternSkyKeyException(e, absoluteValueOfPattern));
        continue;
      }
      TargetPatternKey targetPatternKey =
          new TargetPatternKey(
              targetPattern,
              positive ? policy : FilteringPolicies.NO_FILTER,
              /*isNegative=*/ !positive,
              offset);
      builder.add(new TargetPatternSkyKeyValue(targetPatternKey));
    }
    return builder.build();
  }

  @ThreadSafe
  public static ImmutableList<TargetPatternKey> combineTargetsBelowDirectoryWithNegativePatterns(
      List<TargetPatternKey> keys, boolean excludeSingleTargets) {
    ImmutableList.Builder<TargetPatternKey> builder = ImmutableList.builder();
    // We use indicesOfNegativePatternsThatNeedToBeIncluded to avoid adding negative TBD or single
    // target patterns that have already been combined with previous patterns as an excluded
    // directory or excluded single target.
    HashSet<Integer> indicesOfNegativePatternsThatNeedToBeIncluded = new HashSet<>();
    boolean positivePatternSeen = false;
    for (int i = 0; i < keys.size(); i++) {
      TargetPatternKey targetPatternKey = keys.get(i);
      if (targetPatternKey.isNegative()) {
        if (indicesOfNegativePatternsThatNeedToBeIncluded.contains(i) || !positivePatternSeen) {
          builder.add(targetPatternKey);
        }
      } else {
        positivePatternSeen = true;
        TargetPatternKeyWithExclusionsResult result =
            computeTargetPatternKeyWithExclusions(targetPatternKey, i, keys, excludeSingleTargets);
        result.targetPatternKeyMaybe.ifPresent(builder::add);
        indicesOfNegativePatternsThatNeedToBeIncluded.addAll(
            result.indicesOfNegativePatternsThatNeedToBeIncluded);
      }
    }
    return builder.build();
  }

  private static TargetPatternKey setExcludedDirectoriesAndTargets(
      TargetPatternKey original,
      ImmutableSet<PathFragment> excludedSubdirectories,
      ImmutableSet<Label> excludedSingleTargets) {
    FilteringPolicy policy = original.getPolicy();
    if (!excludedSingleTargets.isEmpty()) {
      policy =
          FilteringPolicies.and(policy, new TargetExcludingFilteringPolicy(excludedSingleTargets));
    }
    return new TargetPatternKey(
        original.getParsedPattern(),
        policy,
        original.isNegative(),
        original.getOffset(),
        excludedSubdirectories);
  }

  private static class TargetPatternKeyWithExclusionsResult {
    private final Optional<TargetPatternKey> targetPatternKeyMaybe;
    private final ImmutableList<Integer> indicesOfNegativePatternsThatNeedToBeIncluded;

    private TargetPatternKeyWithExclusionsResult(
        Optional<TargetPatternKey> targetPatternKeyMaybe,
        ImmutableList<Integer> indicesOfNegativePatternsThatNeedToBeIncluded) {
      this.targetPatternKeyMaybe = targetPatternKeyMaybe;
      this.indicesOfNegativePatternsThatNeedToBeIncluded =
          indicesOfNegativePatternsThatNeedToBeIncluded;
    }
  }

  private static TargetPatternKeyWithExclusionsResult computeTargetPatternKeyWithExclusions(
      TargetPatternKey targetPatternKey,
      int position,
      List<TargetPatternKey> keys,
      boolean excludeSingleTargets) {
    TargetPattern targetPattern = targetPatternKey.getParsedPattern();
    ImmutableSet.Builder<PathFragment> excludedDirectoriesBuilder = ImmutableSet.builder();
    ImmutableSet.Builder<Label> excludedSingleTargetsBuilder = ImmutableSet.builder();
    ImmutableList.Builder<Integer> indicesOfNegativePatternsThatNeedToBeIncludedBuilder =
        ImmutableList.builder();
    for (int j = position + 1; j < keys.size(); j++) {
      TargetPatternKey laterTargetPatternKey = keys.get(j);
      TargetPattern laterParsedPattern = laterTargetPatternKey.getParsedPattern();
      if (!laterTargetPatternKey.isNegative()) {
        continue;
      }
      if (laterParsedPattern.getType() == Type.TARGETS_BELOW_DIRECTORY) {
        TargetsBelowDirectory laterParsedTargetsBelowDirectory =
            (TargetsBelowDirectory) laterParsedPattern;
        if (targetPattern.getType() == Type.TARGETS_BELOW_DIRECTORY) {
          TargetsBelowDirectory targetsBelowDirectory = (TargetsBelowDirectory) targetPattern;
          if (laterParsedTargetsBelowDirectory.contains(targetsBelowDirectory)
              == ContainsResult.DIRECTORY_EXCLUSION_WOULD_BE_EXACT) {
            return new TargetPatternKeyWithExclusionsResult(Optional.empty(), ImmutableList.of());
          } else {
            switch (targetsBelowDirectory.contains(laterParsedTargetsBelowDirectory)) {
              case DIRECTORY_EXCLUSION_WOULD_BE_EXACT:
                excludedDirectoriesBuilder.add(
                    laterParsedTargetsBelowDirectory.getDirectory().getPackageFragment());
                break;
              case DIRECTORY_EXCLUSION_WOULD_BE_TOO_BROAD:
                indicesOfNegativePatternsThatNeedToBeIncludedBuilder.add(j);
                break;
              case NOT_CONTAINED:
                // Nothing to do with this pattern.
            }
          }
        }
      } else if (excludeSingleTargets && laterParsedPattern.getType() == Type.SINGLE_TARGET) {
        try {
          Label label =
              Label.parseAbsolute(
                  laterParsedPattern.getSingleTargetPath(),
                  /*repositoryMapping=*/ ImmutableMap.of());
          excludedSingleTargetsBuilder.add(label);
        } catch (LabelSyntaxException e) {
          indicesOfNegativePatternsThatNeedToBeIncludedBuilder.add(j);
        }
      } else {
        indicesOfNegativePatternsThatNeedToBeIncludedBuilder.add(j);
      }
    }
    return new TargetPatternKeyWithExclusionsResult(
        Optional.of(
            setExcludedDirectoriesAndTargets(
                targetPatternKey,
                excludedDirectoriesBuilder.build(),
                excludedSingleTargetsBuilder.build())),
        indicesOfNegativePatternsThatNeedToBeIncludedBuilder.build());
  }

  public ResolvedTargets<Label> getTargets() {
    return targets;
  }

  /**
   * A TargetPatternKey is a tuple of pattern (eg, "foo/..."), filtering policy, a relative pattern
   * offset, whether it is a positive or negative match, and a set of excluded subdirectories.
   */
  @ThreadSafe
  public static class TargetPatternKey implements SkyKey, Serializable {
    private final TargetPattern parsedPattern;
    private final FilteringPolicy policy;
    private final boolean isNegative;

    private final PathFragment offset;
    /**
     * Must be "compatible" with {@link #parsedPattern}: if {@link #parsedPattern} is a {@link
     * TargetsBelowDirectory} object, then no element of {@code excludedSubdirectories} fully
     * contains it.
     */
    private final ImmutableSet<PathFragment> excludedSubdirectories;

    public TargetPatternKey(
        TargetPattern parsedPattern,
        FilteringPolicy policy,
        boolean isNegative,
        PathFragment offset) {
      this(parsedPattern, policy, isNegative, offset, ImmutableSet.of());
    }

    private TargetPatternKey(
        TargetPattern parsedPattern,
        FilteringPolicy policy,
        boolean isNegative,
        PathFragment offset,
        ImmutableSet<PathFragment> excludedSubdirectories) {
      this.parsedPattern = Preconditions.checkNotNull(parsedPattern);
      this.policy = Preconditions.checkNotNull(policy);
      this.isNegative = isNegative;
      this.offset = offset;
      this.excludedSubdirectories = Preconditions.checkNotNull(excludedSubdirectories);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.TARGET_PATTERN;
    }

    public String getPattern() {
      return parsedPattern.getOriginalPattern();
    }

    public TargetPattern getParsedPattern() {
      return parsedPattern;
    }

    public boolean isNegative() {
      return isNegative;
    }

    public FilteringPolicy getPolicy() {
      return policy;
    }

    public PathFragment getOffset() {
      return offset;
    }

    public ImmutableSet<PathFragment> getExcludedSubdirectories() {
      return excludedSubdirectories;
    }

    ImmutableSet<PathFragment> getAllSubdirectoriesToExclude(
        ImmutableSet<PathFragment> ignoredPackagePrefixes) throws InterruptedException {
      ImmutableSet.Builder<PathFragment> excludedPathsBuilder = ImmutableSet.builder();
      excludedPathsBuilder.addAll(getExcludedSubdirectories());
      excludedPathsBuilder.addAll(
          getAllIgnoredSubdirectoriesToExclude(() -> ignoredPackagePrefixes));
      return excludedPathsBuilder.build();
    }

    public ImmutableSet<PathFragment> getAllIgnoredSubdirectoriesToExclude(
        InterruptibleSupplier<ImmutableSet<PathFragment>> ignoredPackagePrefixes)
        throws InterruptedException {
      return getAllIgnoredSubdirectoriesToExclude(parsedPattern, ignoredPackagePrefixes);
    }

    public static ImmutableSet<PathFragment> getAllIgnoredSubdirectoriesToExclude(
        TargetPattern pattern,
        InterruptibleSupplier<ImmutableSet<PathFragment>> ignoredPackagePrefixes)
        throws InterruptedException {
      if (pattern.getType() != Type.TARGETS_BELOW_DIRECTORY) {
        return ImmutableSet.of();
      }
      TargetsBelowDirectory targetsBelowDirectory = (TargetsBelowDirectory) pattern;
      ImmutableSet<PathFragment> ignoredPaths = ignoredPackagePrefixes.get();
      ImmutableSet.Builder<PathFragment> ignoredPathsBuilder =
          ImmutableSet.builderWithExpectedSize(0);
      for (PathFragment ignoredPackagePrefix : ignoredPaths) {
        PackageIdentifier pkgIdForIgnoredDirectorPrefix =
            PackageIdentifier.create(
                targetsBelowDirectory.getDirectory().getRepository(), ignoredPackagePrefix);
        if (targetsBelowDirectory.containsAllTransitiveSubdirectories(
            pkgIdForIgnoredDirectorPrefix)) {
          ignoredPathsBuilder.add(ignoredPackagePrefix);
        }
      }
      return ignoredPathsBuilder.build();
    }

    @Override
    public String toString() {
      return String.format(
          "%s, excludedSubdirs=%s, filteringPolicy=%s",
          (isNegative ? "-" : "") + parsedPattern.getOriginalPattern(),
          excludedSubdirectories,
          getPolicy());
    }

    @Override
    public int hashCode() {
      return Objects.hash(parsedPattern, isNegative, policy, offset,
          excludedSubdirectories);
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof TargetPatternKey)) {
        return false;
      }
      TargetPatternKey other = (TargetPatternKey) obj;

      return other.isNegative == this.isNegative && other.parsedPattern.equals(this.parsedPattern)
          && other.offset.equals(this.offset) && other.policy.equals(this.policy)
          && other.excludedSubdirectories.equals(this.excludedSubdirectories);
    }
  }

  /**
   * Wrapper for a target pattern {@link SkyKey} or the {@link TargetParsingException} thrown when
   * trying to compute it.
   */
  public interface TargetPatternSkyKeyOrException {

    /**
     * Returns the stored {@link SkyKey} or throws {@link TargetParsingException} if one was thrown
     * when computing the key.
     */
    TargetPatternKey getSkyKey() throws TargetParsingException;

    /**
     * Returns the pattern that resulted in the stored {@link SkyKey} or {@link
     * TargetParsingException}.
     */
    String getOriginalPattern();
  }

  private static final class TargetPatternSkyKeyValue implements TargetPatternSkyKeyOrException {

    private final TargetPatternKey value;

    private TargetPatternSkyKeyValue(TargetPatternKey value) {
      this.value = value;
    }

    @Override
    public TargetPatternKey getSkyKey() throws TargetParsingException {
      return value;
    }

    @Override
    public String getOriginalPattern() {
      return ((TargetPatternKey) value.argument()).getPattern();
    }
  }

  private static final class TargetPatternSkyKeyException implements
      TargetPatternSkyKeyOrException {

    private final TargetParsingException exception;
    private final String originalPattern;

    private TargetPatternSkyKeyException(TargetParsingException exception, String originalPattern) {
      this.exception = exception;
      this.originalPattern = originalPattern;
    }

    @Override
    public TargetPatternKey getSkyKey() throws TargetParsingException {
      throw exception;
    }

    @Override
    public String getOriginalPattern() {
      return originalPattern;
    }
  }
}
