// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BuildSettingProvider;
import com.google.devtools.build.lib.analysis.RequiredConfigFragmentsProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.config.CoreOptions.IncludeConfigFragmentsEnum;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.AttributeTransitionData;
import com.google.devtools.build.lib.packages.ConfigurationFragmentPolicy;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleTransitionData;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import java.util.Collection;
import java.util.Optional;
import javax.annotation.Nullable;

/**
 * Utility methods for determining what {@link Fragment}s are required to analyze targets.
 *
 * <p>For example if a target reads <code>--copt</code> as part of its analysis logic, it requires
 * the {@link com.google.devtools.build.lib.rules.cpp.CppConfiguration} fragment.
 *
 * <p>Used by {@link
 * com.google.devtools.build.lib.query2.cquery.CqueryOptions#showRequiredConfigFragments}.
 */
public class RequiredFragmentsUtil {

  /**
   * Returns a {@link RequiredConfigFragmentsProvider} identifying all pieces of configuration a
   * target requires, or {@code null} if required config fragments are not enabled (see {@link
   * CoreOptions#includeRequiredConfigFragmentsProvider}).
   *
   * <p>The returned config state includes things that are known to be required at the time when the
   * target's dependencies have already been analyzed but before it's been analyzed itself. See
   * {@link RuleConfiguredTargetBuilder#maybeAddRequiredConfigFragmentsProvider} for the remaining
   * pieces of config state.
   *
   * <p>If {@code configuration} is {@link CoreOptions.IncludeConfigFragmentsEnum#DIRECT}, the
   * result includes only the config state considered to be directly required by this target. If
   * it's {@link CoreOptions.IncludeConfigFragmentsEnum#TRANSITIVE}, it also includes config state
   * needed by transitive dependencies. If it's {@link CoreOptions.IncludeConfigFragmentsEnum#OFF},
   * this method returns {@code null}.
   *
   * <p>{@code select()}s and toolchain dependencies are considered when looking at what config
   * state is required.
   *
   * @param target the target
   * @param configuration the configuration for this target
   * @param universallyRequiredFragments fragments that are always required even if not explicitly
   *     specified for this target
   * @param configurationFragmentPolicy source of truth for the fragments required by this target
   *     class definition
   * @param configConditions <code>config_settings</code> required by this target's <code>select
   *     </code>s. Used for a) figuring out which options <code>select</code>s read and b) figuring
   *     out which transitions are attached to the target. {@link TransitionFactory}, which
   *     determines the transitions, may read the target's attributes.
   * @param prerequisites all prerequisites of this aspect
   * @return {@link RequiredConfigFragmentsProvider} or {@code null} if not enabled
   */
  @Nullable
  public static RequiredConfigFragmentsProvider getRequiredFragmentsIfEnabled(
      Rule target,
      BuildConfiguration configuration,
      FragmentClassSet universallyRequiredFragments,
      ConfigurationFragmentPolicy configurationFragmentPolicy,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      Iterable<ConfiguredTargetAndData> prerequisites) {
    if (!requiredFragmentsEnabled(configuration)) {
      return null;
    }
    return getRequiredFragments(
        target.isBuildSetting() ? Optional.of(target.getLabel()) : Optional.empty(),
        configuration,
        universallyRequiredFragments,
        configurationFragmentPolicy,
        configConditions.values(),
        getTransitions(target, configConditions, configuration.checksum()),
        prerequisites);
  }

  /**
   * Variation of {@link #getRequiredFragments} for aspects.
   *
   * @param aspect the aspect
   * @param associatedTarget the target this aspect is attached to
   * @param configuration the configuration for this aspect
   * @param universallyRequiredFragments fragments that are always required even if not explicitly
   *     specified for this aspect
   * @param configurationFragmentPolicy source of truth for the fragments required by this aspect
   *     class definition
   * @param configConditions <code>config_settings</code> required by <code>select</code>s on the
   *     associated target. Used for figuring out which transitions are attached to the target.
   * @param prerequisites all prerequisites of this aspect
   * @return {@link RequiredConfigFragmentsProvider} or {@code null} if not enabled
   */
  @Nullable
  public static RequiredConfigFragmentsProvider getRequiredFragmentsIfEnabled(
      Aspect aspect,
      Rule associatedTarget,
      BuildConfiguration configuration,
      FragmentClassSet universallyRequiredFragments,
      ConfigurationFragmentPolicy configurationFragmentPolicy,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      Iterable<ConfiguredTargetAndData> prerequisites) {
    if (!requiredFragmentsEnabled(configuration)) {
      return null;
    }
    return getRequiredFragments(
        /*buildSettingLabel=*/ Optional.empty(),
        configuration,
        universallyRequiredFragments,
        configurationFragmentPolicy,
        configConditions.values(),
        getTransitions(
            aspect,
            ConfiguredAttributeMapper.of(
                associatedTarget, configConditions, configuration.checksum())),
        prerequisites);
  }

  /** Internal implementation that handles any source (target or aspect). */
  private static RequiredConfigFragmentsProvider getRequiredFragments(
      Optional<Label> buildSettingLabel,
      BuildConfiguration configuration,
      FragmentClassSet universallyRequiredFragments,
      ConfigurationFragmentPolicy configurationFragmentPolicy,
      Collection<ConfigMatchingProvider> configConditions,
      Collection<ConfigurationTransition> associatedTransitions,
      Iterable<ConfiguredTargetAndData> prerequisites) {
    // Add directly required fragments:
    RequiredConfigFragmentsProvider.Builder requiredFragments =
        RequiredConfigFragmentsProvider.builder()
            // Fragments explicitly required by the native target/aspect definition API:
            .addFragmentClasses(configurationFragmentPolicy.getRequiredConfigurationFragments())
            // Fragments explicitly required by the Starlark target/aspect definition API:
            .addFragmentClasses(
                Collections2.transform(
                    configurationFragmentPolicy.getRequiredStarlarkFragments(),
                    configuration::getStarlarkFragmentByName))
            // Fragments universally required by everything:
            .addFragmentClasses(universallyRequiredFragments);
    // Fragments required by attached select()s.
    configConditions.forEach(
        configCondition -> requiredFragments.merge(configCondition.requiredFragmentOptions()));
    // We consider build settings (which are both targets and configuration) to require themselves:
    buildSettingLabel.ifPresent(requiredFragments::addStarlarkOption);

    // Fragments required by attached configuration transitions.
    for (ConfigurationTransition transition : associatedTransitions) {
      transition.addRequiredFragments(requiredFragments, configuration.getOptions());
    }

    // Optionally add transitively required fragments (only if --show_config_fragments=transitive):
    addRequiredFragmentsFromDeps(requiredFragments, configuration, prerequisites);
    return new RequiredConfigFragmentsProvider(requiredFragments.build());
  }

  /**
   * Returns the {@link ConfigurationTransition}s "attached" to a target.
   *
   * <p>"Attached" means the transition is attached to the target itself or one of its attributes.
   *
   * <p>These are the transitions required for a target to successfully analyze. Technically,
   * transitions attached to the target are evaluated during its parent's analysis, which is where
   * the configuration for the child is determined. We still consider these the child's requirements
   * because the child's properties determine that dependency.
   */
  private static ImmutableList<ConfigurationTransition> getTransitions(
      Rule target,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      String configHash) {
    ImmutableList.Builder<ConfigurationTransition> transitions = ImmutableList.builder();
    if (target.getRuleClassObject().getTransitionFactory() != null) {
      transitions.add(
          target
              .getRuleClassObject()
              .getTransitionFactory()
              .create(RuleTransitionData.create(target)));
    }
    // We don't set the execution platform in this data because a) that doesn't affect which
    // fragments are required and b) it's one less parameter we have to pass to
    // RequiredFragmenstUtil's public interface.
    AttributeTransitionData attributeTransitionData =
        AttributeTransitionData.builder()
            .attributes(ConfiguredAttributeMapper.of(target, configConditions, configHash))
            .build();
    for (Attribute attribute : target.getRuleClassObject().getAttributes()) {
      if (attribute.getTransitionFactory() != null) {
        transitions.add(attribute.getTransitionFactory().create(attributeTransitionData));
      }
    }
    return transitions.build();
  }

  /**
   * Returns the {@link ConfigurationTransition}s "attached" to an aspect.
   *
   * <p>"Attached" means the transition is attached to one of the aspect's attributes. Transitions
   * can'be attached directly to aspects themselves.
   */
  private static ImmutableList<ConfigurationTransition> getTransitions(
      Aspect aspect, AttributeMap attributeMap) {
    ImmutableList.Builder<ConfigurationTransition> transitions = ImmutableList.builder();
    AttributeTransitionData attributeTransitionData =
        AttributeTransitionData.builder().attributes(attributeMap).build();
    for (Attribute attribute : aspect.getDefinition().getAttributes().values()) {
      if (attribute.getTransitionFactory() != null) {
        transitions.add(attribute.getTransitionFactory().create(attributeTransitionData));
      }
    }
    return transitions.build();
  }

  /**
   * Returns fragments required by deps. This includes:
   *
   * <ul>
   *   <li>Requirements transitively required by deps iff {@link
   *       CoreOptions#includeRequiredConfigFragmentsProvider} is {@link
   *       CoreOptions.IncludeConfigFragmentsEnum#TRANSITIVE},
   *   <li>Dependencies on Starlark build settings iff {@link
   *       CoreOptions#includeRequiredConfigFragmentsProvider} is not {@link
   *       CoreOptions.IncludeConfigFragmentsEnum#OFF}. These are considered direct requirements on
   *       the rule.
   * </ul>
   */
  private static void addRequiredFragmentsFromDeps(
      RequiredConfigFragmentsProvider.Builder requiredFragments,
      BuildConfiguration configuration,
      Iterable<ConfiguredTargetAndData> prerequisites) {
    CoreOptions coreOptions = configuration.getOptions().get(CoreOptions.class);
    for (ConfiguredTargetAndData prereq : prerequisites) {
      // If the target depends on a Starlark build setting, conceptually that means it directly
      // requires that as an option (even though it's technically a dependency).
      BuildSettingProvider buildSettingProvider =
          prereq.getConfiguredTarget().getProvider(BuildSettingProvider.class);
      if (buildSettingProvider != null) {
        requiredFragments.addStarlarkOption(buildSettingProvider.getLabel());
      }
      if (coreOptions.includeRequiredConfigFragmentsProvider
          == CoreOptions.IncludeConfigFragmentsEnum.TRANSITIVE) {
        // Add fragments only required because transitive deps need them.
        RequiredConfigFragmentsProvider depProvider =
            prereq.getConfiguredTarget().getProvider(RequiredConfigFragmentsProvider.class);
        if (depProvider != null) {
          requiredFragments.merge(depProvider);
        }
      }
    }
  }

  private static boolean requiredFragmentsEnabled(BuildConfiguration config) {
    IncludeConfigFragmentsEnum setting =
        config.getOptions().get(CoreOptions.class).includeRequiredConfigFragmentsProvider;
    switch (setting) {
      case OFF:
        return false;
      case DIRECT_HOST_ONLY:
        return config.isHostConfiguration();
      case DIRECT:
      case TRANSITIVE:
        return true;
    }
    throw new AssertionError(setting);
  }

  private RequiredFragmentsUtil() {}
}
