// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Verify;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.DependencyResolver.DependencyKind;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions.IncludeConfigFragmentsEnum;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.configuredtargets.EnvironmentGroupConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.InputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.OutputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.PackageGroupConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.skylark.SkylarkRuleConfiguredTargetUtil;
import com.google.devtools.build.lib.analysis.test.AnalysisFailure;
import com.google.devtools.build.lib.analysis.test.AnalysisFailureInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.AdvertisedProviderSet;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.ConfigurationFragmentPolicy;
import com.google.devtools.build.lib.packages.ConfigurationFragmentPolicy.MissingFragmentPolicy;
import com.google.devtools.build.lib.packages.ConstantRuleVisibility;
import com.google.devtools.build.lib.packages.EnvironmentGroup;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.PackageGroupsRuleVisibility;
import com.google.devtools.build.lib.packages.PackageSpecification;
import com.google.devtools.build.lib.packages.PackageSpecification.PackageGroupContents;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.RuleVisibility;
import com.google.devtools.build.lib.packages.SkylarkProviderIdentifier;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.profiler.memory.CurrentRuleTracker;
import com.google.devtools.build.lib.skyframe.AspectFunction.AspectFunctionException;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * This class creates {@link ConfiguredTarget} instances using a given {@link
 * ConfiguredRuleClassProvider}.
 */
@ThreadSafe
public final class ConfiguredTargetFactory {
  // This class is not meant to be outside of the analysis phase machinery and is only public
  // in order to be accessible from the .view.skyframe package.

  private final ConfiguredRuleClassProvider ruleClassProvider;

  public ConfiguredTargetFactory(ConfiguredRuleClassProvider ruleClassProvider) {
    this.ruleClassProvider = ruleClassProvider;
  }

  /**
   * Returns the visibility of the given target. Errors during package group resolution are reported
   * to the {@code AnalysisEnvironment}.
   */
  private NestedSet<PackageGroupContents> convertVisibility(
      OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> prerequisiteMap,
      EventHandler reporter,
      Target target,
      BuildConfiguration packageGroupConfiguration) {
    RuleVisibility ruleVisibility = target.getVisibility();
    if (ruleVisibility instanceof ConstantRuleVisibility) {
      return ((ConstantRuleVisibility) ruleVisibility).isPubliclyVisible()
          ? NestedSetBuilder.create(
              Order.STABLE_ORDER,
              PackageGroupContents.create(ImmutableList.of(PackageSpecification.everything())))
          : NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    } else if (ruleVisibility instanceof PackageGroupsRuleVisibility) {
      PackageGroupsRuleVisibility packageGroupsVisibility =
          (PackageGroupsRuleVisibility) ruleVisibility;

      NestedSetBuilder<PackageGroupContents> result = NestedSetBuilder.stableOrder();
      for (Label groupLabel : packageGroupsVisibility.getPackageGroups()) {
        // PackageGroupsConfiguredTargets are always in the package-group configuration.
        TransitiveInfoCollection group =
            findPrerequisite(prerequisiteMap, groupLabel, packageGroupConfiguration);
        PackageSpecificationProvider provider = null;
        // group == null can only happen if the package group list comes
        // from a default_visibility attribute, because in every other case,
        // this missing link is caught during transitive closure visitation or
        // if the RuleConfiguredTargetGraph threw out a visibility edge
        // because if would have caused a cycle. The filtering should be done
        // in a single place, ConfiguredTargetGraph, but for now, this is the
        // minimally invasive way of providing a sane error message in case a
        // cycle is created by a visibility attribute.
        if (group != null) {
          provider = group.getProvider(PackageSpecificationProvider.class);
        }
        if (provider != null) {
          result.addTransitive(provider.getPackageSpecifications());
        } else {
          reporter.handle(Event.error(target.getLocation(),
              String.format("Label '%s' does not refer to a package group", groupLabel)));
        }
      }

      result.add(packageGroupsVisibility.getDirectPackages());
      return result.build();
    } else {
      throw new IllegalStateException("unknown visibility");
    }
  }

  private TransitiveInfoCollection findPrerequisite(
      OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> prerequisiteMap,
      Label label,
      BuildConfiguration config) {
    for (ConfiguredTargetAndData prerequisite :
        prerequisiteMap.get(DependencyResolver.VISIBILITY_DEPENDENCY)) {
      if (prerequisite.getTarget().getLabel().equals(label)
          && Objects.equals(prerequisite.getConfiguration(), config)) {
        return prerequisite.getConfiguredTarget();
      }
    }
    return null;
  }

  /**
   * Invokes the appropriate constructor to create a {@link ConfiguredTarget} instance.
   *
   * <p>For use in {@code ConfiguredTargetFunction}.
   *
   * <p>Returns null if Skyframe deps are missing or upon certain errors.
   */
  @Nullable
  public final ConfiguredTarget createConfiguredTarget(
      AnalysisEnvironment analysisEnvironment,
      ArtifactFactory artifactFactory,
      Target target,
      BuildConfiguration config,
      BuildConfiguration hostConfig,
      ConfiguredTargetKey configuredTargetKey,
      OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> prerequisiteMap,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      @Nullable ResolvedToolchainContext toolchainContext)
      throws InterruptedException, ActionConflictException {
    if (target instanceof Rule) {
      try {
        CurrentRuleTracker.beginConfiguredTarget(((Rule) target).getRuleClassObject());
        return createRule(
            analysisEnvironment,
            (Rule) target,
            config,
            hostConfig,
            configuredTargetKey,
            prerequisiteMap,
            configConditions,
            toolchainContext);
      } finally {
        CurrentRuleTracker.endConfiguredTarget();
      }
    }

    // Visibility, like all package groups, doesn't have a configuration
    NestedSet<PackageGroupContents> visibility =
        convertVisibility(prerequisiteMap, analysisEnvironment.getEventHandler(), target, null);
    if (target instanceof OutputFile) {
      OutputFile outputFile = (OutputFile) target;
      TargetContext targetContext =
          new TargetContext(
              analysisEnvironment,
              target,
              config,
              prerequisiteMap.get(DependencyResolver.OUTPUT_FILE_RULE_DEPENDENCY),
              visibility);
      if (analysisEnvironment.getSkyframeEnv().valuesMissing()) {
        return null;
      }
      RuleConfiguredTarget rule =
          (RuleConfiguredTarget)
              targetContext.findDirectPrerequisite(
                  outputFile.getGeneratingRule().getLabel(),
                  // Don't pass a specific configuration, as we don't care what configuration the
                  // generating rule is in. There can only be one actual dependency here, which is
                  // the target that generated the output file.
                  Optional.empty());
      Verify.verifyNotNull(rule);
      Artifact artifact = rule.getArtifactByOutputLabel(outputFile.getLabel());
      return new OutputFileConfiguredTarget(targetContext, outputFile, rule, artifact);
    } else if (target instanceof InputFile) {
      InputFile inputFile = (InputFile) target;
      TargetContext targetContext =
          new TargetContext(
              analysisEnvironment,
              target,
              config,
              prerequisiteMap.get(DependencyResolver.OUTPUT_FILE_RULE_DEPENDENCY),
              visibility);
      SourceArtifact artifact =
          artifactFactory.getSourceArtifact(
              inputFile.getExecPath(),
              inputFile.getPackage().getSourceRoot(),
              ConfiguredTargetKey.of(target.getLabel(), config));
      return new InputFileConfiguredTarget(targetContext, inputFile, artifact);
    } else if (target instanceof PackageGroup) {
      PackageGroup packageGroup = (PackageGroup) target;
      TargetContext targetContext =
          new TargetContext(
              analysisEnvironment,
              target,
              config,
              prerequisiteMap.get(DependencyResolver.VISIBILITY_DEPENDENCY),
              visibility);
      return new PackageGroupConfiguredTarget(targetContext, packageGroup);
    } else if (target instanceof EnvironmentGroup) {
      TargetContext targetContext =
          new TargetContext(analysisEnvironment, target, config, ImmutableSet.of(), visibility);
      return new EnvironmentGroupConfiguredTarget(targetContext);
    } else {
      throw new AssertionError("Unexpected target class: " + target.getClass().getName());
    }
  }

  /**
   * Returns, depending on the configuration, the {@link BuildConfiguration.Fragment}s and {@link
   * FragmentOptions} required by this rule and its transitive dependencies. This includes {@code
   * select}s and toolchain dependencies.
   *
   * <p>If {@code configuration} == {@link IncludeConfigFragmentsEnum#TRANSITIVE}, this includes
   * fragments required by the transitive dependencies. If {@code configuration} == {@link
   * IncludeConfigFragmentsEnum#DIRECT}, this only includes the rule's directly required fragments.
   * If {@code configuration} == {@link IncludeConfigFragmentsEnum#OFF}, this returns an empty list.
   *
   * <p>Returned values are user-friendly string representations of the classes.
   *
   * <p>This doesn't yet support the following fragment dependencies:
   *
   * <ul>
   *   <li>TODO: options read by Starlark or native transitions
   *   <li>TODO: Starlark options. Note that these don't belong to fragments like native options do.
   *       Figure out how to integrate them.
   *   <li>TODO: Android feature flags. These also don't belong to fragments.
   * </ul>
   *
   * @param configuration the configuration for this rule
   * @param universallyRequiredFragments fragments that are always required even if not explicitly
   *     specified for this rule
   * @param directlyRequiredFragments fragments directly required by this rule's definition
   * @param configConditions {@link FragmentOptions} required by {@code select}s on this rule. This
   *     is a different type than the others: options and fragments are different concepts. There's
   *     some subtlety to their relationship (e.g. a {@link FragmentOptions} can be associated with
   *     multiple {@link BuildConfiguration.Fragment}s). Rather than trying to merge all results
   *     into a pure set of {@link BuildConfiguration.Fragment}s we just allow the mix. In practice
   *     the conceptual dependencies remain clear enough without trying to resolve these subtleties.
   * @param prerequisites all prerequisties of this rule
   * @return an alphabetically ordered set of required fragments and options
   */
  private static ImmutableSet<String> getRequiredConfigFragments(
      BuildConfiguration configuration,
      Collection<Class<? extends BuildConfiguration.Fragment>> universallyRequiredFragments,
      Collection<Class<?>> directlyRequiredFragments,
      Collection<ConfigMatchingProvider> configConditions,
      Iterable<ConfiguredTargetAndData> prerequisites) {
    TreeSet<String> requiredFragments = new TreeSet<>();

    CoreOptions coreOptions = configuration.getOptions().get(CoreOptions.class);
    if (coreOptions.includeRequiredConfigFragmentsProvider == IncludeConfigFragmentsEnum.OFF) {
      return ImmutableSet.of();
    }

    universallyRequiredFragments.forEach(
        fragment -> requiredFragments.add(fragment.getSimpleName()));
    directlyRequiredFragments.forEach(fragment -> requiredFragments.add(fragment.getSimpleName()));
    configConditions.forEach(
        configCondition -> requiredFragments.addAll(configCondition.getRequiredFragmentOptions()));

    if (coreOptions.includeRequiredConfigFragmentsProvider
        == IncludeConfigFragmentsEnum.TRANSITIVE) {
      for (ConfiguredTargetAndData prereq : prerequisites) {
        RequiredConfigFragmentsProvider depProvider =
            prereq.getConfiguredTarget().getProvider(RequiredConfigFragmentsProvider.class);
        if (depProvider != null) {
          requiredFragments.addAll(depProvider.getRequiredConfigFragments());
        }
      }
    }

    return ImmutableSet.copyOf(requiredFragments);
  }
  /**
   * Factory method: constructs a RuleConfiguredTarget of the appropriate class, based on the rule
   * class. May return null if an error occurred.
   */
  @Nullable
  private ConfiguredTarget createRule(
      AnalysisEnvironment env,
      Rule rule,
      BuildConfiguration configuration,
      BuildConfiguration hostConfiguration,
      ConfiguredTargetKey configuredTargetKey,
      OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> prerequisiteMap,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      @Nullable ResolvedToolchainContext toolchainContext)
      throws InterruptedException, ActionConflictException {
    ConfigurationFragmentPolicy configurationFragmentPolicy =
        rule.getRuleClassObject().getConfigurationFragmentPolicy();
    // Visibility computation and checking is done for every rule.
    RuleContext ruleContext =
        new RuleContext.Builder(
                env,
                rule,
                ImmutableList.of(),
                configuration,
                hostConfiguration,
                ruleClassProvider.getPrerequisiteValidator(),
                configurationFragmentPolicy,
                configuredTargetKey)
            .setVisibility(convertVisibility(prerequisiteMap, env.getEventHandler(), rule, null))
            .setPrerequisites(transformPrerequisiteMap(prerequisiteMap, rule))
            .setConfigConditions(configConditions)
            .setUniversalFragments(ruleClassProvider.getUniversalFragments())
            .setToolchainContext(toolchainContext)
            .setConstraintSemantics(ruleClassProvider.getConstraintSemantics())
            .setRequiredConfigFragments(
                getRequiredConfigFragments(
                    configuration,
                    ruleClassProvider.getUniversalFragments(),
                    configurationFragmentPolicy.getRequiredConfigurationFragments(),
                    configConditions.values(),
                    prerequisiteMap.values()))
            .build();

    List<NestedSet<AnalysisFailure>> analysisFailures = depAnalysisFailures(ruleContext);
    if (!analysisFailures.isEmpty()) {
      return erroredConfiguredTargetWithFailures(ruleContext, analysisFailures);
    }
    if (ruleContext.hasErrors()) {
      return erroredConfiguredTarget(ruleContext);
    }

    MissingFragmentPolicy missingFragmentPolicy =
        configurationFragmentPolicy.getMissingFragmentPolicy();

    try {
      if (missingFragmentPolicy != MissingFragmentPolicy.IGNORE
          && !configuration.hasAllFragments(
              configurationFragmentPolicy.getRequiredConfigurationFragments())) {
        if (missingFragmentPolicy == MissingFragmentPolicy.FAIL_ANALYSIS) {
          ruleContext.ruleError(
              missingFragmentError(
                  ruleContext, configurationFragmentPolicy, configuration.checksum()));
          return null;
        }
        // Otherwise missingFragmentPolicy == MissingFragmentPolicy.CREATE_FAIL_ACTIONS:
        return createFailConfiguredTarget(ruleContext);
      }
      if (rule.getRuleClassObject().isSkylark()) {
        // TODO(bazel-team): maybe merge with RuleConfiguredTargetBuilder?
        ConfiguredTarget target =
            SkylarkRuleConfiguredTargetUtil.buildRule(
                ruleContext,
                rule.getRuleClassObject().getAdvertisedProviders(),
                rule.getRuleClassObject().getConfiguredTargetFunction(),
                rule.getLocation(),
                env.getSkylarkSemantics(),
                ruleClassProvider.getToolsRepository());

        return target != null ? target : erroredConfiguredTarget(ruleContext);
      } else {
        RuleClass.ConfiguredTargetFactory<ConfiguredTarget, RuleContext, ActionConflictException>
            factory =
                rule.getRuleClassObject()
                    .<ConfiguredTarget, RuleContext, ActionConflictException>
                        getConfiguredTargetFactory();
        Preconditions.checkNotNull(factory, rule.getRuleClassObject());
        return factory.create(ruleContext);
      }
    } catch (RuleErrorException ruleErrorException) {
      // Returning null in this method is an indication a rule error occurred. Exceptions are not
      // propagated, as this would show a nasty stack trace to users, and only provide info
      // on one specific failure with poor messaging. By returning null, the caller can
      // inspect ruleContext for multiple errors and output thorough messaging on each.
      return erroredConfiguredTarget(ruleContext);
    }
  }

  private List<NestedSet<AnalysisFailure>> depAnalysisFailures(RuleContext ruleContext) {
    if (ruleContext.getConfiguration().allowAnalysisFailures()) {
      ImmutableList.Builder<NestedSet<AnalysisFailure>> analysisFailures = ImmutableList.builder();
      Iterable<? extends TransitiveInfoCollection> infoCollections =
          ruleContext.getConfiguredTargetMap().values();
      for (TransitiveInfoCollection infoCollection : infoCollections) {
        AnalysisFailureInfo failureInfo =
            infoCollection.get(AnalysisFailureInfo.SKYLARK_CONSTRUCTOR);
        if (failureInfo != null) {
          analysisFailures.add(failureInfo.getCauses());
        }
      }
      return analysisFailures.build();
    }
    // Analysis failures are only created and propagated if --allow_analysis_failures is
    // enabled, otherwise these result in actual rule errors which are not caught.
    return ImmutableList.of();
  }

  private ConfiguredTarget erroredConfiguredTargetWithFailures(
      RuleContext ruleContext, List<NestedSet<AnalysisFailure>> analysisFailures)
      throws ActionConflictException {
    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext);
    builder.addNativeDeclaredProvider(AnalysisFailureInfo.forAnalysisFailureSets(analysisFailures));
    builder.addProvider(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY));
    return builder.build();
  }

  /**
   * Returns a {@link ConfiguredTarget} which indicates that an analysis error occurred in
   * processing the target. In most cases, this returns null, which signals to callers that
   * the target failed to build and thus the build should fail. However, if analysis failures
   * are allowed in this build, this returns a stub {@link ConfiguredTarget} which contains
   * information about the failure.
   */
  @Nullable
  private ConfiguredTarget erroredConfiguredTarget(RuleContext ruleContext)
      throws ActionConflictException {
    if (ruleContext.getConfiguration().allowAnalysisFailures()) {
      ImmutableList.Builder<AnalysisFailure> analysisFailures = ImmutableList.builder();

      for (String errorMessage : ruleContext.getSuppressedErrorMessages()) {
        analysisFailures.add(new AnalysisFailure(ruleContext.getLabel(), errorMessage));
      }
      RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext);
      builder.addNativeDeclaredProvider(
          AnalysisFailureInfo.forAnalysisFailures(analysisFailures.build()));
      builder.addProvider(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY));
      return builder.build();
    } else {
      // Returning a null ConfiguredTarget is an indication a rule error occurred. Exceptions are
      // not propagated, as this would show a nasty stack trace to users, and only provide info
      // on one specific failure with poor messaging. By returning null, the caller can
      // inspect ruleContext for multiple errors and output thorough messaging on each.
      return null;
    }
  }

  private String missingFragmentError(
      RuleContext ruleContext,
      ConfigurationFragmentPolicy configurationFragmentPolicy,
      String configurationId) {
    RuleClass ruleClass = ruleContext.getRule().getRuleClassObject();
    Set<Class<?>> missingFragments = new LinkedHashSet<>();
    for (Class<?> fragment : configurationFragmentPolicy.getRequiredConfigurationFragments()) {
      if (!ruleContext.getConfiguration().hasFragment(fragment.asSubclass(Fragment.class))) {
        missingFragments.add(fragment);
      }
    }
    Preconditions.checkState(!missingFragments.isEmpty());
    StringBuilder result = new StringBuilder();
    result.append("all rules of type " + ruleClass.getName() + " require the presence of ");
    result.append("all of [");
    result.append(
        missingFragments.stream().map(Class::getSimpleName).collect(Collectors.joining(",")));
    result.append("], but these were all disabled in configuration ").append(configurationId);
    return result.toString();
  }

  @VisibleForTesting
  public static OrderedSetMultimap<Attribute, ConfiguredTargetAndData> transformPrerequisiteMap(
      OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> map, Target target) {
    OrderedSetMultimap<Attribute, ConfiguredTargetAndData> result = OrderedSetMultimap.create();
    for (Map.Entry<DependencyKind, ConfiguredTargetAndData> entry : map.entries()) {
      if (entry.getKey() == DependencyResolver.TOOLCHAIN_DEPENDENCY) {
        continue;
      }
      Attribute attribute = entry.getKey().getAttribute();
      result.put(attribute, entry.getValue());
    }

    return result;
  }

  /**
   * Constructs an {@link ConfiguredAspect}. Returns null if an error occurs; in that case, {@code
   * aspectFactory} should call one of the error reporting methods of {@link RuleContext}.
   */
  public ConfiguredAspect createAspect(
      AnalysisEnvironment env,
      ConfiguredTargetAndData associatedTarget,
      ImmutableList<Aspect> aspectPath,
      ConfiguredAspectFactory aspectFactory,
      Aspect aspect,
      OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> prerequisiteMap,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      @Nullable ResolvedToolchainContext toolchainContext,
      BuildConfiguration aspectConfiguration,
      BuildConfiguration hostConfiguration,
      ActionLookupValue.ActionLookupKey aspectKey)
      throws AspectFunctionException, InterruptedException {

    RuleContext.Builder builder =
        new RuleContext.Builder(
            env,
            associatedTarget.getTarget(),
            aspectPath,
            aspectConfiguration,
            hostConfiguration,
            ruleClassProvider.getPrerequisiteValidator(),
            aspect.getDefinition().getConfigurationFragmentPolicy(),
            aspectKey);

    Map<String, Attribute> aspectAttributes = mergeAspectAttributes(aspectPath);

    RuleContext ruleContext =
        builder
            .setVisibility(
                convertVisibility(
                    prerequisiteMap, env.getEventHandler(), associatedTarget.getTarget(), null))
            .setPrerequisites(
                transformPrerequisiteMap(prerequisiteMap, associatedTarget.getTarget()))
            .setAspectAttributes(aspectAttributes)
            .setConfigConditions(configConditions)
            .setUniversalFragments(ruleClassProvider.getUniversalFragments())
            .setToolchainContext(toolchainContext)
            .setConstraintSemantics(ruleClassProvider.getConstraintSemantics())
            .build();

    // If allowing analysis failures, targets should be created as normal as possible, and errors
    // will be propagated via a hook elsewhere as AnalysisFailureInfo.
    boolean allowAnalysisFailures = ruleContext.getConfiguration().allowAnalysisFailures();

    if (ruleContext.hasErrors() && !allowAnalysisFailures) {
      return null;
    }

    ConfiguredAspect configuredAspect;
    try {
      configuredAspect =
          aspectFactory.create(
              associatedTarget,
              ruleContext,
              aspect.getParameters(),
              ruleClassProvider.getToolsRepository());
    } catch (ActionConflictException e) {
      throw new AspectFunctionException(e);
    }
    if (configuredAspect != null) {
      validateAdvertisedProviders(
          configuredAspect, aspect.getDefinition().getAdvertisedProviders(),
          associatedTarget.getTarget(),
          env.getEventHandler()
      );
    }
    return configuredAspect;
  }

  private ImmutableMap<String, Attribute> mergeAspectAttributes(ImmutableList<Aspect> aspectPath) {
    if (aspectPath.isEmpty()) {
      return ImmutableMap.of();
    } else if (aspectPath.size() == 1) {
      return aspectPath.get(0).getDefinition().getAttributes();
    } else {

      LinkedHashMap<String, Attribute> aspectAttributes = new LinkedHashMap<>();
      for (Aspect underlyingAspect : aspectPath) {
        ImmutableMap<String, Attribute> currentAttributes = underlyingAspect.getDefinition()
            .getAttributes();
        for (Map.Entry<String, Attribute> kv : currentAttributes.entrySet()) {
          if (!aspectAttributes.containsKey(kv.getKey())) {
            aspectAttributes.put(kv.getKey(), kv.getValue());
          }
        }
      }
      return ImmutableMap.copyOf(aspectAttributes);
    }
  }

  private void validateAdvertisedProviders(
      ConfiguredAspect configuredAspect,
      AdvertisedProviderSet advertisedProviders, Target target,
      EventHandler eventHandler) {
    if (advertisedProviders.canHaveAnyProvider()) {
      return;
    }
    for (Class<?> aClass : advertisedProviders.getNativeProviders()) {
      if (configuredAspect.getProvider(aClass.asSubclass(TransitiveInfoProvider.class)) == null) {
        eventHandler.handle(Event.error(
            target.getLocation(),
            String.format(
                "Aspect '%s', applied to '%s', does not provide advertised provider '%s'",
                configuredAspect.getName(),
                target.getLabel(),
                aClass.getSimpleName()
            )));
      }
    }

    for (SkylarkProviderIdentifier providerId : advertisedProviders.getSkylarkProviders()) {
      if (configuredAspect.getProvider(providerId) == null) {
        eventHandler.handle(Event.error(
            target.getLocation(),
            String.format(
                "Aspect '%s', applied to '%s', does not provide advertised provider '%s'",
                configuredAspect.getName(),
                target.getLabel(),
                providerId
            )));
      }
    }
  }

  /**
   * A pseudo-implementation for configured targets that creates fail actions for all declared
   * outputs, both implicit and explicit.
   */
  private static ConfiguredTarget createFailConfiguredTarget(RuleContext ruleContext)
      throws RuleErrorException, ActionConflictException {
    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext);
    if (!ruleContext.getOutputArtifacts().isEmpty()) {
      ruleContext.registerAction(new FailAction(ruleContext.getActionOwner(),
          ruleContext.getOutputArtifacts(), "Can't build this"));
    }
    builder.add(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY));
    return builder.build();
  }
}
