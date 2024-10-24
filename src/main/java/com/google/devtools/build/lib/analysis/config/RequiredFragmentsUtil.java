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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.Collections2;
import com.google.devtools.build.lib.analysis.BuildSettingProvider;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RequiredConfigFragmentsProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.config.CoreOptions.IncludeConfigFragmentsEnum;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.analysis.starlark.StarlarkAttributeTransitionProvider;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeTransitionData;
import com.google.devtools.build.lib.packages.ConfigurationFragmentPolicy;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleTransitionData;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import java.util.Objects;
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
public final class RequiredFragmentsUtil {

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
   * @param configConditions <code>config_settings</code> required by this target's <code>select
   *     </code>s. Used for a) figuring out which options <code>select</code>s read and b) figuring
   *     out which transitions are attached to the target. {@link TransitionFactory}, which
   *     determines the transitions, may read the target's attributes.
   * @param prerequisites all prerequisites of {@code target}
   * @param starlarkExecTransition the Starlark transition implementing the exec transition
   * @return {@link RequiredConfigFragmentsProvider} or {@code null} if not enabled
   */
  @Nullable
  public static RequiredConfigFragmentsProvider getRuleRequiredFragmentsIfEnabled(
      Rule target,
      BuildConfigurationValue configuration,
      FragmentClassSet universallyRequiredFragments,
      ConfigConditions configConditions,
      Iterable<ConfiguredTarget> prerequisites,
      @Nullable StarlarkAttributeTransitionProvider starlarkExecTransition) {
    IncludeConfigFragmentsEnum mode = getRequiredFragmentsMode(configuration);
    if (mode == IncludeConfigFragmentsEnum.OFF) {
      return null;
    }
    RuleClass ruleClass = target.getRuleClassObject();
    ConfiguredAttributeMapper attributes =
        ConfiguredAttributeMapper.of(target, configConditions.asProviders(), configuration);
    RequiredConfigFragmentsProvider.Builder requiredFragments =
        getRequiredFragments(
            mode,
            configuration,
            universallyRequiredFragments,
            ruleClass.getConfigurationFragmentPolicy(),
            configConditions,
            prerequisites);
    if (!ruleClass.isStarlark()) {
      ruleClass
          .getConfiguredTargetFactory(RuleConfiguredTargetFactory.class)
          .addRuleImplSpecificRequiredConfigFragments(requiredFragments, attributes, configuration);
    }
    addRequiredFragmentsFromRuleTransitions(
        configConditions,
        configuration.checksum(),
        requiredFragments,
        target,
        attributes,
        configuration.getBuildOptionDetails(),
        starlarkExecTransition);

    // We consider build settings (which are both targets and configuration) to require themselves.
    if (target.isBuildSetting()) {
      requiredFragments.addStarlarkOption(target.getLabel());
    }

    return requiredFragments.build();
  }

  /**
   * Variation of {@link #getRuleRequiredFragmentsIfEnabled} for aspects.
   *
   * @param aspect the aspect
   * @param aspectFactory the corresponding {@link ConfiguredAspectFactory}
   * @param associatedTarget the target this aspect is attached to
   * @param configuration the configuration for this aspect
   * @param universallyRequiredFragments fragments that are always required even if not explicitly
   *     specified for this aspect
   * @param configConditions <code>config_settings</code> required by <code>select</code>s on the
   *     associated target. Used for figuring out which transitions are attached to the target.
   * @param prerequisites all prerequisites of {@code aspect}
   * @param starlarkExecTransition the Starlark transition implementing the exec transition
   * @return {@link RequiredConfigFragmentsProvider} or {@code null} if not enabled
   */
  @Nullable
  public static RequiredConfigFragmentsProvider getAspectRequiredFragmentsIfEnabled(
      Aspect aspect,
      ConfiguredAspectFactory aspectFactory,
      Rule associatedTarget,
      BuildConfigurationValue configuration,
      FragmentClassSet universallyRequiredFragments,
      ConfigConditions configConditions,
      Iterable<ConfiguredTarget> prerequisites,
      StarlarkAttributeTransitionProvider starlarkExecTransition) {
    IncludeConfigFragmentsEnum mode = getRequiredFragmentsMode(configuration);
    if (mode == IncludeConfigFragmentsEnum.OFF) {
      return null;
    }
    RequiredConfigFragmentsProvider.Builder requiredFragments =
        getRequiredFragments(
            mode,
            configuration,
            universallyRequiredFragments,
            aspect.getDefinition().getConfigurationFragmentPolicy(),
            configConditions,
            prerequisites);
    aspectFactory.addAspectImplSpecificRequiredConfigFragments(requiredFragments);
    addRequiredFragmentsFromAspectTransitions(
        requiredFragments,
        aspect,
        ConfiguredAttributeMapper.of(
            associatedTarget, configConditions.asProviders(), configuration),
        configuration.getBuildOptionDetails(),
        starlarkExecTransition);
    return requiredFragments.build();
  }

  /** Internal implementation that handles requirements common to both rules and aspects. */
  private static RequiredConfigFragmentsProvider.Builder getRequiredFragments(
      IncludeConfigFragmentsEnum mode,
      BuildConfigurationValue configuration,
      FragmentClassSet universallyRequiredFragments,
      ConfigurationFragmentPolicy configurationFragmentPolicy,
      ConfigConditions configConditions,
      Iterable<ConfiguredTarget> prerequisites) {
    RequiredConfigFragmentsProvider.Builder requiredFragments =
        RequiredConfigFragmentsProvider.builder();

    if (mode == IncludeConfigFragmentsEnum.TRANSITIVE) {
      // Add transitive requirements first, which results in better performance. See explanation on
      // RequiredConfigFragmentsProvider.Builder.
      addTransitivelyRequiredFragments(requiredFragments, prerequisites);
    } else {
      addStarlarkBuildSettings(requiredFragments, prerequisites);
    }

    // Add directly required fragments:
    requiredFragments
        // Fragments explicitly required by the native target/aspect definition API:
        .addFragmentClasses(configurationFragmentPolicy.getRequiredConfigurationFragments())
        // Fragments explicitly required by the Starlark target/aspect definition API (nulls are
        // filtered because the rule definition may reference non-existent fragments):
        .addFragmentClasses(
            Collections2.filter(
                Collections2.transform(
                    configurationFragmentPolicy.getRequiredStarlarkFragments(),
                    configuration::getStarlarkFragmentByName),
                Objects::nonNull))
        // Fragments universally required by everything:
        .addFragmentClasses(universallyRequiredFragments);
    // Fragments required by attached select()s. Propagating fragments from the config conditions as
    // configured targets (rather than as providers) is necessary in case of a dependency on an
    // alias that resolves to a config setting. Providers only reflect the resolved settings, which
    // won't include fragments required to resolve a select within the alias rule (b/237534193).
    for (ConfiguredTargetAndData targetAndData : configConditions.asConfiguredTargets().values()) {
      requiredFragments.merge(
          targetAndData.getConfiguredTarget().getProvider(RequiredConfigFragmentsProvider.class));
    }
    return requiredFragments;
  }

  /**
   * Adds required fragments from transitions "attached" to a target.
   *
   * <p>"Attached" means the transition is attached to the target itself or one of its attributes.
   *
   * <p>These are the transitions required for a target to successfully analyze. Technically,
   * transitions attached to the target are evaluated during its parent's analysis, which is where
   * the configuration for the child is determined. We still consider these the child's requirements
   * because the child's properties determine that dependency.
   */
  private static void addRequiredFragmentsFromRuleTransitions(
      ConfigConditions configConditions,
      String configHash,
      RequiredConfigFragmentsProvider.Builder requiredFragments,
      Rule target,
      ConfiguredAttributeMapper attributeMap,
      BuildOptionDetails optionDetails,
      @Nullable StarlarkAttributeTransitionProvider starlarkExecTransition) {
    target
        .getRuleClassObject()
        .getTransitionFactory()
        .create(RuleTransitionData.create(target, configConditions.asProviders(), configHash))
        .addRequiredFragments(requiredFragments, optionDetails);
    // We don't set the execution platform in this data because a) that doesn't affect which
    // fragments are required and b) it's one less parameter we have to pass to
    // RequiredFragmenstUtil's public interface.
    AttributeTransitionData attributeTransitionData =
        AttributeTransitionData.builder()
            .attributes(attributeMap)
            .analysisData(starlarkExecTransition)
            .build();
    for (Attribute attribute : target.getRuleClassObject().getAttributes()) {
      if (attribute.getTransitionFactory() != null) {
        attribute
            .getTransitionFactory()
            .create(attributeTransitionData)
            .addRequiredFragments(requiredFragments, optionDetails);
      }
    }
  }

  /**
   * Adds required fragments from transitions "attached" to an aspect.
   *
   * <p>"Attached" means the transition is attached to one of the aspect's attributes. Transitions
   * can't be attached directly to aspects themselves.
   */
  private static void addRequiredFragmentsFromAspectTransitions(
      RequiredConfigFragmentsProvider.Builder requiredFragments,
      Aspect aspect,
      ConfiguredAttributeMapper attributeMap,
      BuildOptionDetails optionDetails,
      StarlarkAttributeTransitionProvider starlarkExecTransition) {
    AttributeTransitionData attributeTransitionData =
        AttributeTransitionData.builder()
            .attributes(attributeMap)
            .analysisData(starlarkExecTransition)
            .build();
    for (Attribute attribute : aspect.getDefinition().getAttributes().values()) {
      if (attribute.getTransitionFactory() != null) {
        attribute
            .getTransitionFactory()
            .create(attributeTransitionData)
            .addRequiredFragments(requiredFragments, optionDetails);
      }
    }
  }

  private static void addTransitivelyRequiredFragments(
      RequiredConfigFragmentsProvider.Builder requiredFragments,
      Iterable<ConfiguredTarget> prerequisites) {
    for (ConfiguredTarget prereq : prerequisites) {
      RequiredConfigFragmentsProvider depProvider =
          prereq.getProvider(RequiredConfigFragmentsProvider.class);
      if (depProvider != null) {
        requiredFragments.merge(depProvider);
      }
    }
  }

  /**
   * Adds dependencies on Starlark build settings.
   *
   * <p>Starlark build settings are considered direct requirements on the rule even though they are
   * technically dependencies. Note that this method only needs to be called in {@link
   * IncludeConfigFragmentsEnum#DIRECT} mode, since {@link IncludeConfigFragmentsEnum#TRANSITIVE}
   * mode will already pick up these requirements from dependencies.
   */
  private static void addStarlarkBuildSettings(
      RequiredConfigFragmentsProvider.Builder requiredFragments,
      Iterable<ConfiguredTarget> prerequisites) {
    for (ConfiguredTarget prereq : prerequisites) {
      BuildSettingProvider buildSettingProvider = prereq.getProvider(BuildSettingProvider.class);
      if (buildSettingProvider != null) {
        requiredFragments.addStarlarkOption(buildSettingProvider.getLabel());
      }
    }
  }

  private static IncludeConfigFragmentsEnum getRequiredFragmentsMode(
      BuildConfigurationValue config) {
    return checkNotNull(
        config.getOptions().get(CoreOptions.class).includeRequiredConfigFragmentsProvider);
  }

  private RequiredFragmentsUtil() {}
}
