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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.util.Map;
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
  public ImmutableSet<String> requiresOptionFragments(BuildOptions options) {
    // At first glance this code looks wrong. A composing transition applies transition2 over
    // transition1's outputs, not the original options. We don't have to worry about that here
    // because the reason we pass the options is so Starlark transitions can map individual flags
    // like "//command_line_option:copts" to the fragments that own them. This doesn't depend on the
    // flags' values. This is fortunate, because it producers simpler, faster code and cleaner
    // interfaces.
    return ImmutableSet.<String>builder()
        .addAll(transition1.requiresOptionFragments(options))
        .addAll(transition2.requiresOptionFragments(options))
        .build();
  }

  @Override
  public Map<String, BuildOptions> apply(BuildOptionsView buildOptions, EventHandler eventHandler)
      throws InterruptedException {
    ImmutableMap.Builder<String, BuildOptions> toOptions = ImmutableMap.builder();
    Map<String, BuildOptions> transition1Output =
        transition1.apply(
            TransitionUtil.restrict(transition1, buildOptions.underlying()), eventHandler);
    for (Map.Entry<String, BuildOptions> entry1 : transition1Output.entrySet()) {
      Map<String, BuildOptions> transition2Output =
          transition2.apply(TransitionUtil.restrict(transition2, entry1.getValue()), eventHandler);
      for (Map.Entry<String, BuildOptions> entry2 : transition2Output.entrySet()) {
        toOptions.put(composeKeys(entry1.getKey(), entry2.getKey()), entry2.getValue());
      }
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
  public boolean isHostTransition() {
    return transition1.isHostTransition() || transition2.isHostTransition();
  }

  // Override to allow recursive visiting.
  @Override
  public <E extends Exception> void visit(Visitor<E> visitor) throws E {
    this.transition1.visit(visitor);
    this.transition2.visit(visitor);
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

    if (isFinal(transition1)) {
      // Since no other transition can be composed with transition1, use it directly.
      return transition1;
    } else if (transition1 == NoTransition.INSTANCE) {
      // Since transition1 causes no changes, use transition2 directly.
      return transition2;
    }

    if (transition2 == NoTransition.INSTANCE) {
      // Since transition2 causes no changes, use transition 1 directly.
      return transition1;
    } else if (isFinal(transition2)) {
      // When the second transition is null or a HOST transition, there's no need to compose. But
      // this also
      // improves performance: host transitions are common, and ConfiguredTargetFunction has special
      // optimized logic to handle them. If they were buried in the last segment of a
      // ComposingTransition, those optimizations wouldn't trigger.
      return transition2;
    }

    return new ComposingTransition(transition1, transition2);
  }

  private static boolean isFinal(ConfigurationTransition transition) {
    return transition == NullTransition.INSTANCE || transition.isHostTransition();
  }

  /**
   * Composes a new key out of two given keys. Composing two split transitions is not allowed at the
   * moment, so what this essentially does are (1) make sure not both transitions are split and (2)
   * choose one from a split transition, if there's any, or return {@code PATCH_TRANSITION_KEY),
   * if there isn't.
   */
  private String composeKeys(String key1, String key2) {
    if (!key1.equals(PATCH_TRANSITION_KEY)) {
      if (!key2.equals(PATCH_TRANSITION_KEY)) {
        throw new IllegalStateException(
            String.format(
                "can't compose two split transitions %s and %s",
                transition1.getName(), transition2.getName()));
      }
      return key1;
    }
    return key2;
  }
}
