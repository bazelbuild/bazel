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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.Attribute.Configurator;
import com.google.devtools.build.lib.packages.Attribute.SplitTransition;
import com.google.devtools.build.lib.packages.Attribute.Transition;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleTransitionFactory;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.util.Preconditions;

/**
 * Determines which configurations targets should take.
 *
 * <p>This is the "generic engine" for configuration selection. It doesn't know anything about
 * specific rules or their requirements. Rule writers decide those with appropriately placed
 * {@link PatchTransition} declarations. This class then processes those declarations to determine
 * final configurations.
 */
public final class ConfigurationResolver {
  private final DynamicTransitionMapper transitionMapper;

  /**
   * Instantiates this resolver with a helper class that maps non-{@link PatchTransition}s to
   * {@link PatchTransition}s.
   */
  public ConfigurationResolver(DynamicTransitionMapper transitionMapper) {
    this.transitionMapper = transitionMapper;
  }

  /**
   * Given a parent rule and configuration depending on a child through an attribute, determines
   * the configuration the child should take.
   *
   * @param fromConfig the parent rule's configuration
   * @param fromRule the parent rule
   * @param attribute the attribute creating the dependency (e.g. "srcs")
   * @param toTarget the child target (which may or may not be a rule)
   *
   * @return the child's configuration, expressed as a diff from the parent's configuration. This
   *     is usually a {@PatchTransition} but exceptions apply (e.g.
   *     {@link Attribute.ConfigurationTransition}).
   */
  public Transition evaluateTransition(BuildConfiguration fromConfig, final Rule fromRule,
      final Attribute attribute, final Target toTarget) {

    // I. Input files and package groups have no configurations. We don't want to duplicate them.
    if (usesNullConfiguration(toTarget)) {
      return Attribute.ConfigurationTransition.NULL;
    }

    // II. Host configurations never switch to another. All prerequisites of host targets have the
    // same host configuration.
    if (fromConfig.isHostConfiguration()) {
      return Attribute.ConfigurationTransition.NONE;
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
      return Attribute.ConfigurationTransition.NONE;
    }

    // The current transition to apply. When multiple transitions are requested, this is a
    // ComposingSplitTransition, which encapsulates them into a single object so calling code
    // doesn't need special logic for combinations.
    Transition currentTransition = Attribute.ConfigurationTransition.NONE;

    // Apply the parent rule's outgoing transition if it has one.
    RuleTransitionFactory transitionFactory =
        fromRule.getRuleClassObject().getOutgoingTransitionFactory();
    if (transitionFactory != null) {
      Transition transition = transitionFactory.buildTransitionFor(toTarget.getAssociatedRule());
      if (transition != null) {
        currentTransition = composeTransitions(currentTransition, transition);
      }
    }

    // TODO(gregce): make the below transitions composable (i.e. take away the "else" clauses).
    // The "else" is a legacy restriction from static configurations.
    if (attribute.hasSplitConfigurationTransition()) {
      Preconditions.checkState(attribute.getConfigurator() == null);
      currentTransition = split(currentTransition,
          (SplitTransition<BuildOptions>) attribute.getSplitTransition(fromRule));
    } else {
      // III. Attributes determine configurations. The configuration of a prerequisite is determined
      // by the attribute.
      @SuppressWarnings("unchecked")
      Configurator<BuildOptions> configurator =
          (Configurator<BuildOptions>) attribute.getConfigurator();
      if (configurator != null) {
        // TODO(gregce): attribute configurators are a holdover from static configurations. Remove
        // them and remove this branch.
        currentTransition = applyAttributeConfigurator(currentTransition, configurator);
      } else {
        currentTransition = composeTransitions(currentTransition,
            attribute.getConfigurationTransition());
      }
    }

    return applyConfigurationHook(currentTransition, toTarget);
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
  @VisibleForTesting
  public Transition composeTransitions(Transition transition1, Transition transition2) {
    if (isFinal(transition1)) {
      return transition1;
    } else if (transition2 == Attribute.ConfigurationTransition.NONE) {
      return transition1;
    } else if (transition2 == Attribute.ConfigurationTransition.NULL) {
      // A NULL transition can just replace earlier transitions: no need to cfpose them.
      return Attribute.ConfigurationTransition.NULL;
    } else if (transition2 == Attribute.ConfigurationTransition.HOST) {
      // A HOST transition can just replace earlier transitions: no need to compose them.
      // But it also improves performance: host transitions are common, and
      // ConfiguredTargetFunction has special optimized logic to handle them. If they were buried
      // in the last segment of a ComposingSplitTransition, those optimizations wouldn't trigger.
      return HostTransition.INSTANCE;
    }

    // TODO(gregce): remove the below conversion when all transitions are patch transitions.
    Transition dynamicTransition = transitionMapper.map(transition2);
    return transition1 == Attribute.ConfigurationTransition.NONE
        ? dynamicTransition
        : new ComposingSplitTransition(transition1, dynamicTransition);
  }

  /**
   * Returns true if once the given transition is applied to a dep no followup transitions should
   * be composed after it.
   */
  private static boolean isFinal(Transition transition) {
    return (transition == Attribute.ConfigurationTransition.NULL
        || transition == HostTransition.INSTANCE);
  }

  /**
   * Applies the given split and composes it after an existing transition.
   */
  private static Transition split(Transition currentTransition,
      SplitTransition<BuildOptions> split) {
    Preconditions.checkState(currentTransition != Attribute.ConfigurationTransition.NULL,
        "cannot apply splits after null transitions (null transitions are expected to be final)");
    Preconditions.checkState(currentTransition != HostTransition.INSTANCE,
        "cannot apply splits after host transitions (host transitions are expected to be final)");
    return currentTransition == Attribute.ConfigurationTransition.NONE
        ? split
        : new ComposingSplitTransition(currentTransition, split);
  }

  /**
   * Applies the given attribute configurator and composes it after an existing transition.
   */
  @VisibleForTesting
  public Transition applyAttributeConfigurator(Transition currentTransition,
      Configurator<BuildOptions> configurator) {
    if (isFinal(currentTransition)) {
      return currentTransition;
    }
    return composeTransitions(currentTransition, new AttributeConfiguratorTransition(configurator));
  }

  /**
   * A {@link PatchTransition} that applies an attribute configurator over some input options
   * to determine which transition to use, then applies that transition over those options
   * for the final output.
   */
  private static final class AttributeConfiguratorTransition implements PatchTransition {
    private final Configurator<BuildOptions> configurator;

    AttributeConfiguratorTransition(Configurator<BuildOptions> configurator) {
      this.configurator = configurator;
    }

    @Override
    public BuildOptions apply(BuildOptions options) {
      return Iterables.getOnlyElement(
          ComposingSplitTransition.apply(options, configurator.apply(options)));
    }

    @Override
    public boolean defaultsToSelf() {
      return false;
    }
  }

  /**
   * Applies any configuration hooks associated with the dep target, composes their transitions
   * after an existing transition, and returns the composed result.
   */
  private Transition applyConfigurationHook(Transition currentTransition, Target toTarget) {
    if (isFinal(currentTransition)) {
      return currentTransition;
    }
    Rule associatedRule = toTarget.getAssociatedRule();
    RuleTransitionFactory transitionFactory =
        associatedRule.getRuleClassObject().getTransitionFactory();
    if (transitionFactory != null) {
      // transitionMapper is only needed because of Attribute.ConfigurationTransition.DATA: this is
      // C++-specific but non-C++ rules declare it. So they can't directly provide the C++-specific
      // patch transition that implements it.
      PatchTransition ruleClassTransition = (PatchTransition)
          transitionMapper.map(transitionFactory.buildTransitionFor(associatedRule));
      if (ruleClassTransition != null) {
        if (currentTransition == ConfigurationTransition.NONE) {
          return ruleClassTransition;
        } else {
          return new ComposingSplitTransition(currentTransition, ruleClassTransition);
        }
      }
    }
    return currentTransition;
  }
}
