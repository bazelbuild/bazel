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
import java.util.ArrayList;
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
 *   ComposingTransition(transition1, transition2): { someSetting = $oldVal + " foo bar" }
 * </pre>
 */
@AutoCodec
public class ComposingTransition implements ConfigurationTransition {
  private ConfigurationTransition transition1;
  private ConfigurationTransition transition2;

  /**
   * Creates a {@link ComposingTransition} that applies the sequence: {@code fromOptions ->
   * transition1 -> transition2 -> toOptions }.
   */
  @AutoCodec.Instantiator
  ComposingTransition(ConfigurationTransition transition1, ConfigurationTransition transition2) {
    this.transition1 = transition1;
    this.transition2 = transition2;
  }

  @Override
  public List<BuildOptions> apply(BuildOptions buildOptions) {
    ImmutableList.Builder<BuildOptions> toOptions = ImmutableList.builder();
    for (BuildOptions transition1Options : transition1.apply(buildOptions)) {
      toOptions.addAll(transition2.apply(transition1Options));
    }
    return toOptions.build();
  }

  @Override
  public String reasonForOverride() {
    return "Basic abstraction for combining other transitions";
  }

  @Override
  public String getName() {
    return "(" + transition1.getName() + " + " + transition2.getName() + ")";
  }

  @Override
  public int hashCode() {
    return Objects.hash(transition1, transition2);
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof ComposingTransition
        && ((ComposingTransition) other).transition1.equals(this.transition1)
        && ((ComposingTransition) other).transition2.equals(this.transition2);
  }

  /**
   * Creates a {@link ComposingTransition} that applies the sequence: {@code fromOptions ->
   * transition1 -> transition2 -> toOptions }.
   *
   * <p>Note that this method checks for transitions that cannot be composed, such as if one of the
   * transitions is {@link NoTransition} or the host transition, and returns an efficiently composed
   * transition.
   */
  public static ConfigurationTransition of(
      ConfigurationTransition transition1, ConfigurationTransition transition2) {
    Preconditions.checkNotNull(transition1);
    Preconditions.checkNotNull(transition2);

    if (transition1 == NullTransition.INSTANCE || transition1.isHostTransition()) {
      // Since no other transition can be composed with transition1, use it directly.
      return transition1;
    } else if (transition1 == NoTransition.INSTANCE) {
      // Since transition1 causes no changes, use transition2 directly.
      return transition2;
    }

    if (transition2 == NoTransition.INSTANCE) {
      // Since transition2 causes no changes, use transition 1 directly.
      return transition1;
    } else if (transition2 == NullTransition.INSTANCE || transition2.isHostTransition()) {
      // When the second transition is null or a HOST transition, there's no need to compose. But
      // this also
      // improves performance: host transitions are common, and ConfiguredTargetFunction has special
      // optimized logic to handle them. If they were buried in the last segment of a
      // ComposingTransition, those optimizations wouldn't trigger.
      return transition2;
    }

    return new ComposingTransition(transition1, transition2);
  }

  /**
   * Recursively decompose a composing transition into all the {@link ConfigurationTransition}
   * instances that it holds.
   *
   * @param root {@link ComposingTransition} to decompose
   */
  public static ImmutableList<ConfigurationTransition> decomposeTransition(
      ConfigurationTransition root) {
    ArrayList<ConfigurationTransition> toBeInspected = new ArrayList<>();
    ImmutableList.Builder<ConfigurationTransition> transitions = new ImmutableList.Builder<>();
    toBeInspected.add(root);
    ConfigurationTransition current;
    while (!toBeInspected.isEmpty()) {
      current = toBeInspected.remove(0);
      if (current instanceof ComposingTransition) {
        ComposingTransition composingCurrent = (ComposingTransition) current;
        toBeInspected.addAll(
            ImmutableList.of(composingCurrent.transition1, composingCurrent.transition2));
      } else {
        transitions.add(current);
      }
    }
    return transitions.build();
  }
}
