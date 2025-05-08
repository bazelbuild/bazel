// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.SignedTargetPattern;
import com.google.devtools.build.lib.cmdline.SignedTargetPattern.Sign;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternKey;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/** Utility class to help with evaluating target patterns. */
public class TargetPatternUtil {

  /**
   * Expand the given {@code targetPatterns}, using the {@code filteringPolicy}. This handles the
   * needed underlying Skyframe calls (via {@code env}), and will return {@code null} to signal a
   * Skyframe restart.
   */
  @Nullable
  public static ImmutableList<Label> expandTargetPatterns(
      Environment env, List<SignedTargetPattern> targetPatterns, FilteringPolicy filteringPolicy)
      throws InvalidTargetPatternException, InterruptedException {

    if (targetPatterns.isEmpty()) {
      return ImmutableList.of();
    }

    Iterable<TargetPatternKey> targetPatternKeys =
        TargetPatternValue.keys(targetPatterns, filteringPolicy);
    SkyframeLookupResult resolvedPatterns = env.getValuesAndExceptions(targetPatternKeys);
    boolean valuesMissing = env.valuesMissing();
    // Use an ArrayList so that we can add and remove results based on negative patterns.
    List<Label> labels = valuesMissing ? null : new ArrayList<>();

    for (TargetPatternKey pattern : targetPatternKeys) {
      try {
        TargetPatternValue value =
            (TargetPatternValue) resolvedPatterns.getOrThrow(pattern, TargetParsingException.class);
        if (valuesMissing || value == null) {
          continue;
        }
        if (pattern.isNegative()) {
          // Remove from the results.
          labels.removeAll(value.getTargets().getTargets());
        } else {
          // Add to results.
          labels.addAll(value.getTargets().getTargets());
        }
      } catch (TargetParsingException e) {
        throw new InvalidTargetPatternException(pattern.getPattern(), e);
      }
    }

    if (env.valuesMissing()) {
      if (valuesMissing != env.valuesMissing()) {
        BugReport.logUnexpected(
            "Some value from '%s' was missing, this should never happen", targetPatternKeys);
      }
      return null;
    }

    return ImmutableList.copyOf(labels);
  }

  // TODO(bazel-team): look into moving this into SignedTargetPattern itself.
  public static ImmutableList<SignedTargetPattern> parseAllSigned(
      List<String> patterns, TargetPattern.Parser parser) throws InvalidTargetPatternException {
    ImmutableList.Builder<SignedTargetPattern> parsedPatterns = ImmutableList.builder();
    for (String pattern : patterns) {
      try {
        parsedPatterns.add(SignedTargetPattern.parse(pattern, parser));
      } catch (TargetParsingException e) {
        throw new InvalidTargetPatternException(pattern, e);
      }
    }
    return parsedPatterns.build();
  }

  /** Converts patterns to signed patterns, considering all input patterns positive. */
  public static ImmutableList<SignedTargetPattern> toSigned(List<TargetPattern> patterns) {
    return patterns.stream()
        .map(pattern -> SignedTargetPattern.create(pattern, Sign.POSITIVE))
        .collect(toImmutableList());
  }

  /** Exception used when an error occurs in {@link #expandTargetPatterns}. */
  // TODO(bazel-team): Consolidate this and TargetParsingException. Just have the latter store the
  //   original unparsed pattern too.
  public static final class InvalidTargetPatternException extends Exception {
    private final String invalidPattern;
    private final TargetParsingException tpe;

    public InvalidTargetPatternException(String invalidPattern, TargetParsingException tpe) {
      super(tpe);
      this.invalidPattern = invalidPattern;
      this.tpe = tpe;
    }

    public String getInvalidPattern() {
      return invalidPattern;
    }

    public TargetParsingException getTpe() {
      return tpe;
    }
  }
}
