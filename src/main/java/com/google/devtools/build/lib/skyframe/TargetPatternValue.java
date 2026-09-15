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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.cmdline.SignedTargetPattern;
import com.google.devtools.build.lib.cmdline.SignedTargetPattern.Sign;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.cmdline.TargetPattern.TargetsBelowDirectory;
import com.google.devtools.build.lib.cmdline.TargetPattern.TargetsBelowDirectory.ContainsResult;
import com.google.devtools.build.lib.cmdline.TargetPattern.Type;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
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

  private final ResolvedTargets<Label> targets;

  TargetPatternValue(ResolvedTargets<Label> targets) {
    this.targets = Preconditions.checkNotNull(targets);
  }

  /**
   * Create a target pattern {@link SkyKey}.
   *
   * @param pattern The pattern, eg "-foo/biz...".
   * @param policy The filtering policy, eg "only return test targets"
   */
  @ThreadSafe
  public static TargetPatternKey key(SignedTargetPattern pattern, FilteringPolicy policy) {
    return new TargetPatternKey(
        pattern, pattern.sign() == Sign.POSITIVE ? policy : FilteringPolicies.NO_FILTER);
  }

  /**
   * Returns an iterable of {@link TargetPatternKey}s, in the same order as the list of patterns
   * provided as input.
   *
   * @param patterns The list of patterns, eg "-foo/biz...".
   * @param policy The filtering policy, eg "only return test targets"
   */
  @ThreadSafe
  public static Iterable<TargetPatternKey> keys(
      List<SignedTargetPattern> patterns, FilteringPolicy policy) {
    return patterns.stream().map(pattern -> key(pattern, policy)).collect(toImmutableList());
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
    return new TargetPatternKey(original.getSignedParsedPattern(), policy, excludedSubdirectories);
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
        excludedSingleTargetsBuilder.add(laterParsedPattern.getSingleTargetLabel());
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
  public static class TargetPatternKey implements SkyKey {
    private final SignedTargetPattern signedParsedPattern;
    private final FilteringPolicy policy;

    /**
     * Must be "compatible" with {@link #signedParsedPattern}: if {@link #signedParsedPattern} is a
     * {@link TargetsBelowDirectory} object, then {@link TargetsBelowDirectory#containedIn} is false
     * for every element of {@code excludedSubdirectories}.
     */
    private final ImmutableSet<PathFragment> excludedSubdirectories;

    public TargetPatternKey(SignedTargetPattern signedParsedPattern, FilteringPolicy policy) {
      this(signedParsedPattern, policy, ImmutableSet.of());
    }

    private TargetPatternKey(
        SignedTargetPattern signedParsedPattern,
        FilteringPolicy policy,
        ImmutableSet<PathFragment> excludedSubdirectories) {
      this.signedParsedPattern = Preconditions.checkNotNull(signedParsedPattern);
      this.policy = Preconditions.checkNotNull(policy);
      this.excludedSubdirectories = Preconditions.checkNotNull(excludedSubdirectories);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.TARGET_PATTERN;
    }

    public String getPattern() {
      return signedParsedPattern.pattern().getOriginalPattern();
    }

    public TargetPattern getParsedPattern() {
      return signedParsedPattern.pattern();
    }

    private SignedTargetPattern getSignedParsedPattern() {
      return signedParsedPattern;
    }

    public boolean isNegative() {
      return signedParsedPattern.sign() == Sign.NEGATIVE;
    }

    public FilteringPolicy getPolicy() {
      return policy;
    }

    public ImmutableSet<PathFragment> getExcludedSubdirectories() {
      return excludedSubdirectories;
    }

    @Override
    public String toString() {
      return String.format(
          "%s, excludedSubdirs=%s, filteringPolicy=%s",
          (isNegative() ? "-" : "") + signedParsedPattern.pattern().getOriginalPattern(),
          excludedSubdirectories,
          policy);
    }

    @Override
    public int hashCode() {
      return Objects.hash(signedParsedPattern, policy, excludedSubdirectories);
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof TargetPatternKey other)) {
        return false;
      }

      return other.signedParsedPattern.equals(this.signedParsedPattern)
          && other.policy.equals(this.policy)
          && other.excludedSubdirectories.equals(this.excludedSubdirectories);
    }
  }
}
