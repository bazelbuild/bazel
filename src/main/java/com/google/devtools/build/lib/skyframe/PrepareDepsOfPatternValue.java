// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.cmdline.TargetPattern.Type;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternKey;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternSkyKeyOrException;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.List;

/**
 * The value returned by {@link PrepareDepsOfPatternFunction}. Because that function is
 * invoked only for its side effect (i.e. ensuring the graph contains targets matching the
 * pattern and its transitive dependencies), this value carries no information.
 *
 * <p>Because the returned value is always equal to objects that share its type, this value and the
 * {@link PrepareDepsOfPatternFunction} which computes it are incompatible with change pruning. It
 * should only be requested by consumers who do not require reevaluation when
 * {@link PrepareDepsOfPatternFunction} is reevaluated. Safe consumers include, e.g., top-level
 * consumers, and other functions which invoke {@link PrepareDepsOfPatternFunction} solely for its
 * side-effects.
 */
public class PrepareDepsOfPatternValue implements SkyValue {
  // Note that this value does not guarantee singleton-like reference equality because we use Java
  // deserialization. Java deserialization can create other instances.
  public static final PrepareDepsOfPatternValue INSTANCE = new PrepareDepsOfPatternValue();

  private PrepareDepsOfPatternValue() {
  }

  @Override
  public boolean equals(Object o) {
    return o instanceof PrepareDepsOfPatternValue;
  }

  @Override
  public int hashCode() {
    return 42;
  }

  /**
   * Returns an iterable of {@link PrepareDepsOfPatternSkyKeyOrException}, with {@link
   * TargetPatternKey} arguments. Negative target patterns of type other than {@link
   * Type#TARGETS_BELOW_DIRECTORY} are not permitted. If a provided pattern fails to parse or is
   * negative but not a {@link Type#TARGETS_BELOW_DIRECTORY}, an element in the returned iterable
   * will throw when its {@link PrepareDepsOfPatternSkyKeyOrException#getSkyKey} method is called
   * and will return the failing pattern when its {@link
   * PrepareDepsOfPatternSkyKeyOrException#getOriginalPattern} method is called.
   *
   * <p>There may be fewer returned elements than patterns provided as input. This function will
   * combine negative {@link Type#TARGETS_BELOW_DIRECTORY} patterns with preceding patterns to
   * return an iterable of SkyKeys that avoids loading excluded directories during evaluation.
   *
   * @param patterns The list of patterns, e.g. [//foo/..., -//foo/biz/...]. If a pattern's first
   *     character is "-", it is treated as a negative pattern.
   * @param offset The offset to apply to relative target patterns.
   */
  @ThreadSafe
  public static Iterable<PrepareDepsOfPatternSkyKeyOrException> keys(List<String> patterns,
      String offset) {
    List<TargetPatternSkyKeyOrException> keysMaybe =
        ImmutableList.copyOf(TargetPatternValue.keys(patterns, FilteringPolicies.NO_FILTER,
            offset));

    // This code path is evaluated only for query universe preloading, and the quadratic cost of
    // the code below (i.e. for each pattern, consider each later pattern as a candidate for
    // subdirectory exclusion) is only acceptable because all the use cases for query universe
    // preloading involve short (<10 items) pattern sequences.
    ImmutableList.Builder<PrepareDepsOfPatternSkyKeyOrException> builder = ImmutableList.builder();
    for (int i = 0; i < keysMaybe.size(); i++) {
      TargetPatternSkyKeyOrException keyMaybe = keysMaybe.get(i);
      SkyKey skyKey;
      try {
        skyKey = keyMaybe.getSkyKey();
      } catch (TargetParsingException e) {
        // keyMaybe.getSkyKey() may throw TargetParsingException if its corresponding pattern
        // failed to parse. If so, wrap the exception and return it, so that our caller can
        // deal with it.
        skyKey = null;
        builder.add(new PrepareDepsOfPatternSkyKeyException(e, keyMaybe.getOriginalPattern()));
      }
      if (skyKey != null) {
        TargetPatternKey targetPatternKey = (TargetPatternKey) skyKey.argument();
        if (targetPatternKey.isNegative()) {
          if (!targetPatternKey.getParsedPattern().getType().equals(Type.TARGETS_BELOW_DIRECTORY)) {
            builder.add(
                new PrepareDepsOfPatternSkyKeyException(
                    new TargetParsingException(
                        "Negative target patterns of types other than \"targets below directory\""
                            + " are not permitted."), targetPatternKey.toString()));
          }
          // Otherwise it's a negative TBD pattern which was combined with previous patterns as an
          // excluded directory. These can be skipped because there's no PrepareDepsOfPattern work
          // to be done for them.
        } else {
          builder.add(new PrepareDepsOfPatternSkyKeyValue(setExcludedDirectories(targetPatternKey,
              excludedDirectoriesBeneath(targetPatternKey, i, keysMaybe))));
        }
      }
    }
    return builder.build();
  }

  private static TargetPatternKey setExcludedDirectories(
      TargetPatternKey original, ImmutableSet<PathFragment> excludedSubdirectories) {
    return new TargetPatternKey(original.getParsedPattern(), original.getPolicy(),
        original.isNegative(), original.getOffset(), excludedSubdirectories);
  }

  private static ImmutableSet<PathFragment> excludedDirectoriesBeneath(
      TargetPatternKey targetPatternKey,
      int position,
      List<TargetPatternSkyKeyOrException> keysMaybe) {
    ImmutableSet.Builder<PathFragment> excludedDirectoriesBuilder = ImmutableSet.builder();
    for (int j = position + 1; j < keysMaybe.size(); j++) {
      TargetPatternSkyKeyOrException laterPatternMaybe = keysMaybe.get(j);
      SkyKey laterSkyKey;
      try {
        laterSkyKey = laterPatternMaybe.getSkyKey();
      } catch (TargetParsingException ignored) {
        laterSkyKey = null;
      }
      if (laterSkyKey != null) {
        TargetPatternKey laterTargetPatternKey = (TargetPatternKey) laterSkyKey.argument();
        TargetPattern laterParsedPattern = laterTargetPatternKey.getParsedPattern();
        if (laterTargetPatternKey.isNegative()
            && targetPatternKey.getParsedPattern().containsBelowDirectory(laterParsedPattern)) {
          excludedDirectoriesBuilder.add(laterParsedPattern.getDirectory().getPackageFragment());
        }
      }
    }
    return excludedDirectoriesBuilder.build();
  }

  /**
   * Wrapper for a prepare deps of pattern {@link SkyKey} or the {@link TargetParsingException}
   * thrown when trying to create it.
   */
  public interface PrepareDepsOfPatternSkyKeyOrException {

    /**
     * Returns the stored {@link SkyKey} or throws {@link TargetParsingException} if one was thrown
     * when creating the key.
     */
    SkyKey getSkyKey() throws TargetParsingException;

    /**
     * Returns the pattern that resulted in the stored {@link SkyKey} or {@link
     * TargetParsingException}.
     */
    String getOriginalPattern();
  }


  private static class PrepareDepsOfPatternSkyKeyException implements
      PrepareDepsOfPatternSkyKeyOrException {

    private final TargetParsingException exception;
    private final String originalPattern;

    public PrepareDepsOfPatternSkyKeyException(TargetParsingException exception,
        String originalPattern) {
      this.exception = exception;
      this.originalPattern = originalPattern;
    }

    @Override
    public SkyKey getSkyKey() throws TargetParsingException {
      throw exception;
    }

    @Override
    public String getOriginalPattern() {
      return originalPattern;
    }
  }

  private static class PrepareDepsOfPatternSkyKeyValue implements
      PrepareDepsOfPatternSkyKeyOrException {

    private final TargetPatternKey targetPatternKey;

    public PrepareDepsOfPatternSkyKeyValue(TargetPatternKey targetPatternKey) {
      this.targetPatternKey = targetPatternKey;
    }

    @Override
    public SkyKey getSkyKey() throws TargetParsingException {
      return new SkyKey(SkyFunctions.PREPARE_DEPS_OF_PATTERN, targetPatternKey);
    }

    @Override
    public String getOriginalPattern() {
      return targetPatternKey.getPattern();
    }
  }
}
