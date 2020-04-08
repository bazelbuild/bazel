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

import com.google.devtools.build.lib.analysis.config.transitions.ComposingTransition;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NullTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import javax.annotation.Nullable;

/**
 * Tool for evaluating which {@link ConfigurationTransition}(s) should be applied to given targets.
 *
 * <p>For the work of turning these transitions into actual configurations, see {@link
 * ConfigurationResolver}.
 *
 * <p>This is the "generic engine" for configuration selection. It doesn't know anything about
 * specific rules or their requirements. Rule writers decide those with appropriately placed {@link
 * com.google.devtools.build.lib.analysis.config.transitions.PatchTransition} declarations. This
 * class then processes those declarations to determine final transitions.
 */
public final class TransitionResolver {
  /**
   * Given an original configuration and a base transition, determines the configuration a target
   * should have.
   *
   * @param fromConfig the original configuration
   * @param baseTransition the base configuration transitions computed by this method should be
   *     composed with (the configuration transition of the attribute through which this dependency
   *     happens or {@code NoTransition.INSTANCE} if this is a top-level target)
   * @param toTarget the target whose configuration should be computed (may or may not be a rule)
   * @param trimmingTransitionFactory the transition factory used to trim rules (note: this is a
   *     temporary feature; see the corresponding methods in {@code ConfiguredRuleClassProvider})
   * @return the target's configuration(s), expressed as a diff from the original configuration.
   */
  public static ConfigurationTransition evaluateTransition(
      BuildConfiguration fromConfig,
      ConfigurationTransition baseTransition,
      Target toTarget,
      @Nullable TransitionFactory<Rule> trimmingTransitionFactory) {

    // I. The null configuration always remains the null configuration. We could fold this into
    // (III), but NoTransition doesn't work if the source is the null configuration.
    if (fromConfig == null) {
      return NullTransition.INSTANCE;
    }

    // II. Input files and package groups have no configurations. We don't want to duplicate them.
    if (!toTarget.isConfigurable()) {
      return NullTransition.INSTANCE;
    }

    // III. Host configurations never switch to another. All prerequisites of host targets have the
    // same host configuration.
    if (fromConfig.isHostConfiguration()) {
      return NoTransition.INSTANCE;
    }

    // The current transition to apply. When multiple transitions are requested, this is a
    // ComposingTransition, which encapsulates them into a single object so calling code
    // doesn't need special logic for combinations.
    // IV. Apply whatever transition the attribute requires.
    ConfigurationTransition currentTransition = baseTransition;

    // V. Applies any rule transitions associated with the dep target and composes their
    // transitions with a passed-in existing transition.
    currentTransition = applyRuleTransition(currentTransition, toTarget);

    // VI. Applies a transition to trim the result and returns it. (note: this is a temporary
    // feature; see the corresponding methods in ConfiguredRuleClassProvider)
    return applyTransitionFromFactory(currentTransition, toTarget, trimmingTransitionFactory);
  }

  /**
   * @param currentTransition a pre-existing transition to be composed with
   * @param toTarget target whose associated rule's incoming transition should be applied
   */
  private static ConfigurationTransition applyRuleTransition(
      ConfigurationTransition currentTransition, Target toTarget) {
    Rule associatedRule = toTarget.getAssociatedRule();
    TransitionFactory<Rule> transitionFactory =
        associatedRule.getRuleClassObject().getTransitionFactory();
    return applyTransitionFromFactory(currentTransition, toTarget, transitionFactory);
  }

  /**
   * @param currentTransition a pre-existing transition to be composed with
   * @param toTarget target whose associated rule's incoming transition should be applied
   * @param transitionFactory a rule transition factory to apply, or null to do nothing
   */
  private static ConfigurationTransition applyTransitionFromFactory(
      ConfigurationTransition currentTransition,
      Target toTarget,
      @Nullable TransitionFactory<Rule> transitionFactory) {
    if (transitionFactory != null) {
      return ComposingTransition.of(
          currentTransition, transitionFactory.create(toTarget.getAssociatedRule()));
    }
    return currentTransition;
  }
}
