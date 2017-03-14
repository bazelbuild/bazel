// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.config;

import com.google.common.base.Preconditions;
import com.google.common.base.Verify;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.Attribute.SplitTransition;
import com.google.devtools.build.lib.packages.Attribute.Transition;
import java.util.List;

/**
 * A configuration transition that composes two other transitions in an ordered sequence.
 *
 * <p>Example:
 * <pre>
 *   transition1: { someSetting = $oldVal + " foo" }
 *   transition2: { someSetting = $oldVal + " bar" }
 *   ComposingSplitTransition(transition1, transition2): { someSetting = $oldVal + " foo bar" }
 * </pre>
 *
 * <p>Child transitions can be {@link SplitTransition}s, {@link PatchTransition}s, or any
 * combination thereof. We implement this class as a {@link SplitTransition} since that abstraction
 * captures all possible combinations.
 */
public class ComposingSplitTransition implements SplitTransition<BuildOptions> {
  private Transition transition1;
  private Transition transition2;

  /**
   * Creates a {@link ComposingSplitTransition} that applies the sequence:
   * {@code fromOptions -> transition1 -> transition2 -> toOptions  }.
   */
  public ComposingSplitTransition(Transition transition1, Transition transition2) {
    this.transition1 = verifySupported(transition1);
    this.transition2 = verifySupported(transition2);
  }

  @Override
  public List<BuildOptions> split(BuildOptions buildOptions) {
    ImmutableList.Builder<BuildOptions> toOptions = ImmutableList.builder();
    for (BuildOptions transition1Options : apply(buildOptions, transition1)) {
      toOptions.addAll(apply(transition1Options, transition2));
    }
    return toOptions.build();
  }

  /**
   * Verifies support for the given transition type. Throws an {@link IllegalArgumentException} if
   * unsupported.
   */
  private Transition verifySupported(Transition transition) {
    Preconditions.checkArgument(transition instanceof PatchTransition
        || transition instanceof SplitTransition<?>);
    return transition;
  }

  /**
   * Applies the given transition over the given {@link BuildOptions}, returns the result.
   */
  // TODO(gregce): move this somewhere more general. This isn't intrinsic to composed splits.
  static List<BuildOptions> apply(BuildOptions fromOptions, Transition transition) {
    if (transition == ConfigurationTransition.NONE) {
      return ImmutableList.<BuildOptions>of(fromOptions);
    } else if (transition instanceof PatchTransition) {
      return ImmutableList.<BuildOptions>of(((PatchTransition) transition).apply(fromOptions));
    } else if (transition instanceof SplitTransition) {
      SplitTransition split = (SplitTransition<BuildOptions>) transition;
      List<BuildOptions> splitOptions = split.split(fromOptions);
      if (splitOptions.isEmpty()) {
        Verify.verify(split.defaultsToSelf());
        return ImmutableList.<BuildOptions>of(fromOptions);
      } else {
        return splitOptions;
      }
    } else {
      throw new IllegalStateException(
          String.format("Unsupported composite transition type: %s",
              transition.getClass().getName()));
    }
  }

  @Override
  public boolean defaultsToSelf() {
    return true;
  }
}

