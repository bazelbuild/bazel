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
import com.google.devtools.build.lib.cmdline.SignedTargetPattern;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.cmdline.TargetPattern.Type;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.server.FailureDetails.TargetPatterns;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternKey;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.List;

/**
 * The value returned by {@link PrepareDepsOfPatternFunction}. Because that function is invoked only
 * for its side effect (i.e. ensuring the graph contains targets matching the pattern and its
 * transitive dependencies), this value carries no information.
 *
 * <p>Because the returned value is always equal to objects that share its type, this value and the
 * {@link PrepareDepsOfPatternFunction} which computes it are incompatible with change pruning. It
 * should only be requested by consumers who do not require reevaluation when {@link
 * PrepareDepsOfPatternFunction} is reevaluated. Safe consumers include, e.g., top-level consumers,
 * and other functions which invoke {@link PrepareDepsOfPatternFunction} solely for its
 * side-effects.
 */
public class PrepareDepsOfPatternValue implements SkyValue {
  // Note that this value does not guarantee singleton-like reference equality because we use Java
  // deserialization. Java deserialization can create other instances.
  @SerializationConstant
  public static final PrepareDepsOfPatternValue INSTANCE = new PrepareDepsOfPatternValue();

  private PrepareDepsOfPatternValue() {}

  @Override
  public boolean equals(Object o) {
    return o instanceof PrepareDepsOfPatternValue;
  }

  @Override
  public int hashCode() {
    return 42;
  }

  /**
   * Returns a {@link PrepareDepsOfPatternSkyKeysAndExceptions}, containing {@link
   * PrepareDepsOfPatternSkyKeyValue} and {@link PrepareDepsOfPatternSkyKeyException} instances that
   * have {@link TargetPatternKey} arguments. Negative target patterns of type other than {@link
   * Type#TARGETS_BELOW_DIRECTORY} are not permitted. If a provided pattern fails to parse or is
   * negative but not a {@link Type#TARGETS_BELOW_DIRECTORY}, there will be a corresponding {@link
   * PrepareDepsOfPatternSkyKeyException} in the iterable returned by {@link
   * PrepareDepsOfPatternSkyKeysAndExceptions#getExceptions} whose {@link
   * PrepareDepsOfPatternSkyKeyException#getException} and {@link
   * PrepareDepsOfPatternSkyKeyException#getOriginalPattern} methods return the {@link
   * TargetParsingException} and original pattern, respectively.
   *
   * <p>There may be fewer returned elements in {@link
   * PrepareDepsOfPatternSkyKeysAndExceptions#getValues} than patterns provided as input. This
   * function will combine negative {@link Type#TARGETS_BELOW_DIRECTORY} patterns with preceding
   * patterns to return an iterable of SkyKeys that avoids loading excluded directories during
   * evaluation.
   *
   * @param patterns The list of patterns, e.g. [//foo/..., -//foo/biz/...]. If a pattern's first
   *     character is "-", it is treated as a negative pattern.
   * @param mainRepoTargetParser The target pattern parser configured with the specified offset and
   *     the main repository mapping.
   */
  @ThreadSafe
  public static PrepareDepsOfPatternSkyKeysAndExceptions keys(
      List<String> patterns, TargetPattern.Parser mainRepoTargetParser) {
    ImmutableList.Builder<PrepareDepsOfPatternSkyKeyValue> resultValuesBuilder =
        ImmutableList.builder();
    ImmutableList.Builder<PrepareDepsOfPatternSkyKeyException> resultExceptionsBuilder =
        ImmutableList.builder();
    ImmutableList.Builder<TargetPatternKey> targetPatternKeysBuilder = ImmutableList.builder();
    for (String pattern : patterns) {
      try {
        targetPatternKeysBuilder.add(
            TargetPatternValue.key(
                SignedTargetPattern.parse(pattern, mainRepoTargetParser),
                FilteringPolicies.NO_FILTER));
      } catch (TargetParsingException e) {
        resultExceptionsBuilder.add(new PrepareDepsOfPatternSkyKeyException(e, pattern));
      }
    }
    // This code path is evaluated only for query universe preloading, and the quadratic cost of
    // the code below (i.e. for each pattern, consider each later pattern as a candidate for
    // subdirectory exclusion) is only acceptable because all the use cases for query universe
    // preloading involve short (<10 items) pattern sequences.
    Iterable<TargetPatternKey> combinedTargetPatternKeys =
        TargetPatternValue.combineTargetsBelowDirectoryWithNegativePatterns(
            targetPatternKeysBuilder.build(), /*excludeSingleTargets=*/ false);
    for (TargetPatternKey targetPatternKey : combinedTargetPatternKeys) {
      if (targetPatternKey.isNegative()
          && !targetPatternKey
              .getParsedPattern()
              .getType()
              .equals(TargetPattern.Type.TARGETS_BELOW_DIRECTORY)) {
        resultExceptionsBuilder.add(
            new PrepareDepsOfPatternSkyKeyException(
                new TargetParsingException(
                    "Negative target patterns of types other than \"targets below directory\""
                        + " are not permitted.",
                    TargetPatterns.Code.NEGATIVE_TARGET_PATTERN_NOT_ALLOWED),
                targetPatternKey.toString()));
      } else {
        resultValuesBuilder.add(new PrepareDepsOfPatternSkyKeyValue(targetPatternKey));
      }
    }
    return new PrepareDepsOfPatternSkyKeysAndExceptions(
        resultValuesBuilder.build(), resultExceptionsBuilder.build());
  }

  /**
   * A pair of {@link Iterable<PrepareDepsOfPatternSkyKeyValue>} and {@link
   * Iterable<PrepareDepsOfPatternSkyKeyException>}.
   */
  public static class PrepareDepsOfPatternSkyKeysAndExceptions {
    private final Iterable<PrepareDepsOfPatternSkyKeyValue> values;
    private final Iterable<PrepareDepsOfPatternSkyKeyException> exceptions;

    public PrepareDepsOfPatternSkyKeysAndExceptions(
        Iterable<PrepareDepsOfPatternSkyKeyValue> values,
        Iterable<PrepareDepsOfPatternSkyKeyException> exceptions) {
      this.values = values;
      this.exceptions = exceptions;
    }

    public Iterable<PrepareDepsOfPatternSkyKeyValue> getValues() {
      return values;
    }

    public Iterable<PrepareDepsOfPatternSkyKeyException> getExceptions() {
      return exceptions;
    }
  }

  /** Represents a {@link TargetParsingException} when parsing a target pattern string. */
  public static class PrepareDepsOfPatternSkyKeyException {

    private final TargetParsingException exception;
    private final String originalPattern;

    public PrepareDepsOfPatternSkyKeyException(
        TargetParsingException exception, String originalPattern) {
      this.exception = exception;
      this.originalPattern = originalPattern;
    }

    public TargetParsingException getException() {
      return exception;
    }

    public String getOriginalPattern() {
      return originalPattern;
    }
  }

  /**
   * Represents the successful parsing of a target pattern string into a {@link TargetPatternKey}.
   */
  public static class PrepareDepsOfPatternSkyKeyValue {

    private final TargetPatternKey targetPatternKey;

    PrepareDepsOfPatternSkyKeyValue(TargetPatternKey targetPatternKey) {
      this.targetPatternKey = targetPatternKey;
    }

    public Key getSkyKey() {
      return Key.create(targetPatternKey);
    }

    @AutoCodec
    static class Key extends AbstractSkyKey<TargetPatternKey> {
      private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

      private Key(TargetPatternKey arg) {
        super(arg);
      }

      @VisibleForSerialization
      @AutoCodec.Instantiator
      static Key create(TargetPatternKey arg) {
        return interner.intern(new Key(arg));
      }

      @Override
      public SkyFunctionName functionName() {
        return SkyFunctions.PREPARE_DEPS_OF_PATTERN;
      }

      @Override
      public SkyKeyInterner<Key> getSkyKeyInterner() {
        return interner;
      }
    }
  }
}
