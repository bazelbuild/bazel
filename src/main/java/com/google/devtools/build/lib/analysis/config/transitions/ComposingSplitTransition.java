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

package com.google.devtools.build.lib.analysis.config.transitions;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.util.List;
import java.util.Objects;

/**
 * A configuration transition that composes two other transitions in an ordered sequence.
 *
 * <p>Example:
 *
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
@AutoCodec
public class ComposingSplitTransition implements SplitTransition {
  private ConfigurationTransition transition1;
  private ConfigurationTransition transition2;

  @Override
  public String getName() {
    return "(" + transition1.getName() + " + " + transition2.getName() + ")";
  }

  /**
   * Creates a {@link ComposingSplitTransition} that applies the sequence: {@code fromOptions ->
   * transition1 -> transition2 -> toOptions }.
   *
   * <p>Note that it's possible to create silly transitions with this constructor (e.g., if one or
   * both of the transitions is NoTransition). Use composeTransitions instead, which checks for
   * these states and avoids instantiation appropriately.
   *
   * @see TransitionResolver#composeTransitions
   */
  @AutoCodec.Instantiator
  public ComposingSplitTransition(
      ConfigurationTransition transition1, ConfigurationTransition transition2) {
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
  private ConfigurationTransition verifySupported(ConfigurationTransition transition) {
    Preconditions.checkArgument(transition instanceof PatchTransition
        || transition instanceof SplitTransition);
    return transition;
  }

  @Override
  public int hashCode() {
    return Objects.hash(transition1, transition2);
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof ComposingSplitTransition
        && ((ComposingSplitTransition) other).transition1.equals(this.transition1)
        && ((ComposingSplitTransition) other).transition2.equals(this.transition2);
  }

  /**
   * Returns whether this transition contains only patches (and is thus suitable as a delegate
   * for {@link ComposingPatchTransition}).
   */
  public boolean isPatchOnly() {
    return transition1 instanceof PatchTransition && transition2 instanceof PatchTransition;
  }

  /**
   * Allows this transition to be used in patch-only contexts if it contains only
   * {@link PatchTransition}s.
   *
   * <p>Can only be called if {@link #isPatchOnly()} returns true.
   */
  public ComposingPatchTransition asPatch() {
    return new ComposingPatchTransition(this);
  }

  /**
   * Applies the given transition over the given {@link BuildOptions}, returns the result.
   */
  // TODO(gregce): move this somewhere more general. This isn't intrinsic to composed splits.
  static List<BuildOptions> apply(BuildOptions fromOptions, ConfigurationTransition transition) {
    if (transition instanceof PatchTransition) {
      return ImmutableList.<BuildOptions>of(((PatchTransition) transition).apply(fromOptions));
    } else if (transition instanceof SplitTransition) {
      SplitTransition split = (SplitTransition) transition;
      List<BuildOptions> splitOptions = split.split(fromOptions);
      if (splitOptions.isEmpty()) {
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
}
