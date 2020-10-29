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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.ValueOrException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Utility class to help with evaluating target patterns.
 */
public class TargetPatternUtil {

  /**
   * Expand the given {@code targetPatterns}. This handles the needed underlying Skyframe calls (via
   * {@code env}), and will return {@code null} to signal a Skyframe restart.
   */
  @Nullable
  public static ImmutableList<Label> expandTargetPatterns(
      Environment env, List<String> targetPatterns)
      throws InvalidTargetPatternException, InterruptedException {

    return expandTargetPatterns(env, targetPatterns, FilteringPolicies.NO_FILTER);
  }

  /**
   * Expand the given {@code targetPatterns}, using the {@code filteringPolicy}. This handles the
   * needed underlying Skyframe calls (via {@code env}), and will return {@code null} to signal a
   * Skyframe restart.
   */
  @Nullable
  public static ImmutableList<Label> expandTargetPatterns(
      Environment env, List<String> targetPatterns, FilteringPolicy filteringPolicy)
      throws InvalidTargetPatternException, InterruptedException {

    if (targetPatterns.isEmpty()) {
      return ImmutableList.of();
    }

    // First parse the patterns, and throw any errors immediately.
    List<TargetPatternValue.TargetPatternKey> patternKeys = new ArrayList<>();
    for (TargetPatternValue.TargetPatternSkyKeyOrException keyOrException :
        TargetPatternValue.keys(targetPatterns, filteringPolicy, PathFragment.EMPTY_FRAGMENT)) {

      try {
        patternKeys.add(keyOrException.getSkyKey());
      } catch (TargetParsingException e) {
        throw new InvalidTargetPatternException(keyOrException.getOriginalPattern(), e);
      }
    }

    // Then, resolve the patterns.
    Map<SkyKey, ValueOrException<TargetParsingException>> resolvedPatterns =
        env.getValuesOrThrow(patternKeys, TargetParsingException.class);
    boolean valuesMissing = env.valuesMissing();
    ImmutableList.Builder<Label> labels = valuesMissing ? null : new ImmutableList.Builder<>();

    for (TargetPatternValue.TargetPatternKey pattern : patternKeys) {
      TargetPatternValue value;
      try {
        value = (TargetPatternValue) resolvedPatterns.get(pattern).get();
        if (!valuesMissing && value != null) {
          labels.addAll(value.getTargets().getTargets());
        }
      } catch (TargetParsingException e) {
        throw new InvalidTargetPatternException(pattern.getPattern(), e);
      }
    }

    if (valuesMissing) {
      return null;
    }

    return labels.build();
  }

  /** Exception used when an error occurs in {@link #expandTargetPatterns}. */
  static final class InvalidTargetPatternException extends Exception {
    private String invalidPattern;
    private TargetParsingException tpe;

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
