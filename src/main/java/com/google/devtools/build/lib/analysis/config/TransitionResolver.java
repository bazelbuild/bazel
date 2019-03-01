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
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.config.transitions.ComposingTransition;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NullTransition;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.SplitTransition;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleTransitionFactory;
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
 * PatchTransition} declarations. This class then processes those declarations to determine final
 * transitions.
 */
public final class TransitionResolver {
  /**
   * Given a parent rule and configuration depending on a child through an attribute, determines the
   * configuration the child should take.
   *
   * @param fromConfig the parent rule's configuration
   * @param attributeTransition the configuration transition of the attribute of this dependency
   *     edge
   * @param toTarget the child target (which may or may not be a rule)
   * @param trimmingTransitionFactory the transition factory used to trim rules (note: this is a
   *     temporary feature; see the corresponding methods in ConfiguredRuleClassProvider)
   * @return the child's configuration(s), expressed as a diff from the parent's configuration.
   */
  public static ConfigurationTransition evaluateTransition(
      BuildConfiguration fromConfig,
      ConfigurationTransition attributeTransition,
      Target toTarget,
      @Nullable RuleTransitionFactory trimmingTransitionFactory) {

    // I. The null configuration always remains the null configuration. We could fold this into
    // (III), but NoTransition doesn't work if the source is the null configuration.
    if (fromConfig == null) {
      return NullTransition.INSTANCE;
    }

    // II. Input files and package groups have no configurations. We don't want to duplicate them.
    if (usesNullConfiguration(toTarget)) {
      return NullTransition.INSTANCE;
    }

    // III. Host configurations never switch to another. All prerequisites of host targets have the
    // same host configuration.
    if (fromConfig.isHostConfiguration()) {
      return NoTransition.INSTANCE;
    }

    // Make sure config_setting dependencies are resolved in the referencing rule's configuration,
    // unconditionally. For example, given:
    //
    // genrule(
    //     name = 'myrule',
    //     tools = select({ '//a:condition': [':sometool'] })
    //
    // all labels in "tools" get resolved in the host configuration (since the "tools" attribute
    // declares a host configuration transition). We want to explicitly exclude configuration labels
    // from these transitions, since their *purpose* is to do computation on the owning
    // rule's configuration.
    // TODO(bazel-team): don't require special casing here. This is far too hackish.
    if (toTarget instanceof Rule && ((Rule) toTarget).getRuleClassObject().isConfigMatcher()) {
      // TODO(gregce): see if this actually gets called
      return NoTransition.INSTANCE;
    }

    // The current transition to apply. When multiple transitions are requested, this is a
    // ComposingTransition, which encapsulates them into a single object so calling code
    // doesn't need special logic for combinations.
    // IV. Apply whatever transition the attribute requires.
    ConfigurationTransition currentTransition = attributeTransition;

    // V. Applies any rule transitions associated with the dep target and composes their
    // transitions with a passed-in existing transition.
    currentTransition = applyRuleTransition(currentTransition, toTarget);

    // VI. Applies a transition to trim the result and returns it. (note: this is a temporary
    // feature; see the corresponding methods in ConfiguredRuleClassProvider)
    return applyTransitionFromFactory(currentTransition, toTarget, trimmingTransitionFactory);
  }

  /**
   * Same as evaluateTransition except does not check for transitions coming from parents and
   * enables support for rule-triggered top-level configuration hooks.
   */
  public static ConfigurationTransition evaluateTopLevelTransition(
      TargetAndConfiguration targetAndConfig,
      @Nullable RuleTransitionFactory trimmingTransitionFactory) {
    Target target = targetAndConfig.getTarget();

    // Rule class transitions (chosen by rule class definitions):
    if (target.getAssociatedRule() == null) {
      return NoTransition.INSTANCE;
    }
    ConfigurationTransition ruleTransition = applyRuleTransition(NoTransition.INSTANCE, target);
    ConfigurationTransition trimmingTransition =
        applyTransitionFromFactory(ruleTransition, target, trimmingTransitionFactory);
    return trimmingTransition;
  }

  /**
   * Returns true if the given target should have a null configuration. This method is the
   * "source of truth" for this determination.
   */
  public static boolean usesNullConfiguration(Target target) {
    return target instanceof InputFile || target instanceof PackageGroup;
  }

  /**
   * Composes two transitions together efficiently.
   */
  public static ConfigurationTransition composeTransitions(ConfigurationTransition transition1,
      ConfigurationTransition transition2) {
    Preconditions.checkNotNull(transition1);
    Preconditions.checkNotNull(transition2);
    if (isFinal(transition1) || transition2 == NoTransition.INSTANCE) {
      return transition1;
    } else if (isFinal(transition2) || transition1 == NoTransition.INSTANCE) {
      // When the second transition is a HOST transition, there's no need to compose. But this also
      // improves performance: host transitions are common, and ConfiguredTargetFunction has special
      // optimized logic to handle them. If they were buried in the last segment of a
      // ComposingTransition, those optimizations wouldn't trigger.
      return transition2;
    } else {
      return new ComposingTransition(transition1, transition2);
    }
  }

  /**
   * Returns true if once the given transition is applied to a dep no followup transitions should
   * be composed after it.
   */
  private static boolean isFinal(ConfigurationTransition transition) {
    return (transition == NullTransition.INSTANCE
        || transition == HostTransition.INSTANCE);
  }

  /** Applies the given split and composes it after an existing transition. */
  public static ConfigurationTransition split(
      ConfigurationTransition currentTransition, SplitTransition split) {
    Preconditions.checkState(currentTransition != NullTransition.INSTANCE,
        "cannot apply splits after null transitions (null transitions are expected to be final)");
    Preconditions.checkState(currentTransition != HostTransition.INSTANCE,
        "cannot apply splits after host transitions (host transitions are expected to be final)");
    return composeTransitions(currentTransition, split);
  }

  /**
   * @param currentTransition a pre-existing transition to be composed with
   * @param toTarget target whose associated rule's incoming transition should be applied
   */
  private static ConfigurationTransition applyRuleTransition(
      ConfigurationTransition currentTransition, Target toTarget) {
    Rule associatedRule = toTarget.getAssociatedRule();
    RuleTransitionFactory transitionFactory =
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
      @Nullable RuleTransitionFactory transitionFactory) {
    if (isFinal(currentTransition)) {
      return currentTransition;
    }
    if (transitionFactory != null) {
      return composeTransitions(
          currentTransition, transitionFactory.buildTransitionFor(toTarget.getAssociatedRule()));
    }
    return currentTransition;
  }
}
