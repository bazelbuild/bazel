// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.RequiredConfigFragmentsProvider;
import com.google.devtools.build.lib.analysis.config.BuildOptionDetails;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.events.EventHandler;
import java.util.Map;
import java.util.Objects;

/**
 * A transition factory that composes two other transition factories in an ordered sequence.
 *
 * <p>Example:
 *
 * <pre>
 *   transitionFactory1: { someSetting = $oldVal + " foo" }
 *   transitionFactory2: { someSetting = $oldVal + " bar" }
 *   ComposingTransitionFactory(transitionFactory1, transitionFactory2):
 *     { someSetting = $oldVal + " foo bar" }
 * </pre>
 */
@AutoValue
public abstract class ComposingTransitionFactory<T extends TransitionFactory.Data>
    implements TransitionFactory<T> {

  /**
   * Creates a {@link ComposingTransitionFactory} that applies the given factories in sequence:
   * {@code fromOptions -> transition1 -> transition2 -> toOptions }.
   *
   * <p>Note that this method checks for transition factories that cannot be composed, such as if
   * one of the transitions is {@link NoTransition}, and returns an efficiently composed transition.
   */
  public static <T extends TransitionFactory.Data> TransitionFactory<T> of(
      TransitionFactory<T> transitionFactory1, TransitionFactory<T> transitionFactory2) {

    Preconditions.checkNotNull(transitionFactory1);
    Preconditions.checkNotNull(transitionFactory2);
    Preconditions.checkArgument(
        transitionFactory1.transitionType().isCompatibleWith(transitionFactory2.transitionType()),
        "transition factory types must be compatible");
    Preconditions.checkArgument(
        !transitionFactory1.isSplit() || !transitionFactory2.isSplit(),
        "can't compose two split transition factories");

    if (NoTransition.isInstance(transitionFactory1)) {
      // Since transitionFactory1 causes no changes, use transitionFactory2 directly.
      return transitionFactory2;
    } else if (NoTransition.isInstance(transitionFactory2)) {
      // Since transitionFactory2 causes no changes, use transitionFactory1 directly.
      return transitionFactory1;
    }

    return create(transitionFactory1, transitionFactory2);
  }

  private static <T extends TransitionFactory.Data> TransitionFactory<T> create(
      TransitionFactory<T> transitionFactory1, TransitionFactory<T> transitionFactory2) {
    return new AutoValue_ComposingTransitionFactory<T>(transitionFactory1, transitionFactory2);
  }

  @Override
  public ConfigurationTransition create(T data) {
    ConfigurationTransition transition1 = transitionFactory1().create(data);
    ConfigurationTransition transition2 = transitionFactory2().create(data);
    return new ComposingTransition(transition1, transition2);
  }

  @Override
  public TransitionType transitionType() {
    // Both types must match so this is correct.
    return transitionFactory1().transitionType();
  }

  abstract TransitionFactory<T> transitionFactory1();

  abstract TransitionFactory<T> transitionFactory2();

  @Override
  public boolean isTool() {
    return transitionFactory1().isTool() || transitionFactory2().isTool();
  }

  @Override
  public boolean isSplit() {
    return transitionFactory1().isSplit() || transitionFactory2().isSplit();
  }

  @Override
  public void visit(Visitor<T> visitor) {
    this.transitionFactory1().visit(visitor);
    this.transitionFactory2().visit(visitor);
  }

  /** A configuration transition that composes two other transitions in an ordered sequence. */
  private static final class ComposingTransition implements ConfigurationTransition {
    private final ConfigurationTransition transition1;
    private final ConfigurationTransition transition2;

    /**
     * Creates a {@link ComposingTransition} that applies the sequence: {@code fromOptions ->
     * transition1 -> transition2 -> toOptions }.
     */
    private ComposingTransition(
        ConfigurationTransition transition1, ConfigurationTransition transition2) {
      this.transition1 = transition1;
      this.transition2 = transition2;
    }

    @Override
    public void addRequiredFragments(
        RequiredConfigFragmentsProvider.Builder requiredFragments,
        BuildOptionDetails optionDetails) {
      // At first glance this code looks wrong. A composing transition applies transition2 over
      // transition1's outputs, not the original options. We don't have to worry about that here
      // because the reason we pass the options is so Starlark transitions can map individual flags
      // like "//command_line_option:copts" to the fragments that own them. This doesn't depend on
      // the
      // flags' values. This is fortunate, because it producers simpler, faster code and cleaner
      // interfaces.
      transition1.addRequiredFragments(requiredFragments, optionDetails);
      transition2.addRequiredFragments(requiredFragments, optionDetails);
    }

    @Override
    public ImmutableMap<String, BuildOptions> apply(
        BuildOptionsView buildOptions, EventHandler eventHandler) throws InterruptedException {
      ImmutableMap.Builder<String, BuildOptions> toOptions = ImmutableMap.builder();
      Map<String, BuildOptions> transition1Output =
          transition1.apply(
              TransitionUtil.restrict(transition1, buildOptions.underlying()), eventHandler);
      for (Map.Entry<String, BuildOptions> entry1 : transition1Output.entrySet()) {
        Map<String, BuildOptions> transition2Output =
            transition2.apply(
                TransitionUtil.restrict(transition2, entry1.getValue()), eventHandler);
        for (Map.Entry<String, BuildOptions> entry2 : transition2Output.entrySet()) {
          toOptions.put(composeKeys(entry1.getKey(), entry2.getKey()), entry2.getValue());
        }
      }
      return toOptions.buildOrThrow();
    }

    @Override
    public String reasonForOverride() {
      return "Basic abstraction for combining other transitions";
    }

    @Override
    public String getName() {
      return "(" + transition1.getName() + " + " + transition2.getName() + ")";
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
      return other instanceof ComposingTransition composingTransition
          && composingTransition.transition1.equals(this.transition1)
          && ((ComposingTransition) other).transition2.equals(this.transition2);
    }

    /**
     * Composes a new key out of two given keys. Composing two split transitions is not allowed at
     * the moment, so what this essentially does are (1) make sure not both transitions are split
     * and (2) choose one from a split transition, if there's any, or return {@code
     * PATCH_TRANSITION_KEY}, if there isn't.
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
}
