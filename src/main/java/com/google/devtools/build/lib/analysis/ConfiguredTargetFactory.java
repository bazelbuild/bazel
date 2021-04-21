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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.RuleContext.InvalidExecGroupException;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.ConfigConditions;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.RequiredFragmentsUtil;
import com.google.devtools.build.lib.analysis.configuredtargets.EnvironmentGroupConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.InputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.OutputFileConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.PackageGroupConfiguredTarget;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.constraints.RuleContextConstraintSemantics;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleConfiguredTargetUtil;
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
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.profiler.memory.CurrentRuleTracker;
import com.google.devtools.build.lib.server.FailureDetails.FailAction.Code;
import com.google.devtools.build.lib.skyframe.AspectValueKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import net.starlark.java.eval.Mutability;

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
        prerequisiteMap.get(DependencyKind.VISIBILITY_DEPENDENCY)) {
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
      ConfigConditions configConditions,
      @Nullable ToolchainCollection<ResolvedToolchainContext> toolchainContexts)
      throws InterruptedException, ActionConflictException, InvalidExecGroupException {
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
            toolchainContexts);
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
              prerequisiteMap.get(DependencyKind.OUTPUT_FILE_RULE_DEPENDENCY),
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

      DeniedImplicitOutputMarkerProvider deniedProvider =
          rule.get(DeniedImplicitOutputMarkerProvider.PROVIDER);
      if (deniedProvider != null) {
        analysisEnvironment.getEventHandler().handle(Event.error(deniedProvider.getErrorMessage()));
        return null;
      }

      return new OutputFileConfiguredTarget(targetContext, outputFile, rule, artifact);
    } else if (target instanceof InputFile) {
      InputFile inputFile = (InputFile) target;
      TargetContext targetContext =
          new TargetContext(
              analysisEnvironment,
              target,
              config,
              prerequisiteMap.get(DependencyKind.OUTPUT_FILE_RULE_DEPENDENCY),
              visibility);
      SourceArtifact artifact =
          artifactFactory.getSourceArtifact(
              inputFile.getExecPath(
                  analysisEnvironment
                      .getStarlarkSemantics()
                      .getBool(BuildLanguageOptions.EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT)),
              inputFile.getPackage().getSourceRoot().get(),
              ConfiguredTargetKey.builder()
                  .setLabel(target.getLabel())
                  .setConfiguration(config)
                  .build());
      return new InputFileConfiguredTarget(targetContext, inputFile, artifact);
    } else if (target instanceof PackageGroup) {
      PackageGroup packageGroup = (PackageGroup) target;
      TargetContext targetContext =
          new TargetContext(
              analysisEnvironment,
              target,
              config,
              prerequisiteMap.get(DependencyKind.VISIBILITY_DEPENDENCY),
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
      ConfigConditions configConditions,
      @Nullable ToolchainCollection<ResolvedToolchainContext> toolchainContexts)
      throws InterruptedException, ActionConflictException, InvalidExecGroupException {
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
            .setToolsRepository(ruleClassProvider.getToolsRepository())
            .setStarlarkSemantics(env.getStarlarkSemantics())
            .setMutability(Mutability.create("configured target"))
            .setVisibility(convertVisibility(prerequisiteMap, env.getEventHandler(), rule, null))
            .setPrerequisites(transformPrerequisiteMap(prerequisiteMap, rule))
            .setConfigConditions(configConditions)
            .setUniversalFragments(ruleClassProvider.getUniversalFragments())
            .setToolchainContexts(toolchainContexts)
            .setConstraintSemantics(ruleClassProvider.getConstraintSemantics())
            .setRequiredConfigFragments(
                RequiredFragmentsUtil.getRequiredFragments(
                    rule,
                    configuration,
                    ruleClassProvider.getUniversalFragments(),
                    configurationFragmentPolicy,
                    configConditions.asProviders(),
                    prerequisiteMap.values()))
            .build();

    List<NestedSet<AnalysisFailure>> analysisFailures =
        depAnalysisFailures(ruleContext, ImmutableList.of());
    if (!analysisFailures.isEmpty()) {
      return erroredConfiguredTargetWithFailures(ruleContext, analysisFailures);
    }
    if (ruleContext.hasErrors()) {
      return erroredConfiguredTarget(ruleContext);
    }

    ConfiguredTarget incompatibleTarget =
        RuleContextConstraintSemantics.incompatibleConfiguredTarget(ruleContext, prerequisiteMap);
    if (incompatibleTarget != null) {
      return incompatibleTarget;
    }

    try {
      Class<?> missingFragmentClass = null;
      for (Class<?> fragmentClass :
          configurationFragmentPolicy.getRequiredConfigurationFragments()) {
        if (!configuration.hasFragment(fragmentClass.asSubclass(Fragment.class))) {
          MissingFragmentPolicy missingFragmentPolicy =
              configurationFragmentPolicy.getMissingFragmentPolicy(fragmentClass);
          if (missingFragmentPolicy != MissingFragmentPolicy.IGNORE) {
            if (missingFragmentPolicy == MissingFragmentPolicy.FAIL_ANALYSIS) {
              ruleContext.ruleError(
                  missingFragmentError(
                      ruleContext, configurationFragmentPolicy, configuration.checksum()));
              return null;
            }
            // Otherwise missingFragmentPolicy == MissingFragmentPolicy.CREATE_FAIL_ACTIONS:
            missingFragmentClass = fragmentClass;
          }
        }
      }
      if (missingFragmentClass != null) {
        return createFailConfiguredTargetForMissingFragmentClass(ruleContext, missingFragmentClass);
      }

      try {
        if (rule.getRuleClassObject().isStarlark()) {
          // TODO(bazel-team): maybe merge with RuleConfiguredTargetBuilder?
          ConfiguredTarget target =
              StarlarkRuleConfiguredTargetUtil.buildRule(
                  ruleContext,
                  rule.getRuleClassObject().getAdvertisedProviders(),
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
      } finally {
        ruleContext.close();
      }
    } catch (RuleErrorException ruleErrorException) {
      return erroredConfiguredTarget(ruleContext);
    }
  }

  /**
   * If {@code --allow_analysis_failures} is true, returns a collection of propagated analysis
   * failures from the target's dependencies and {@code extraDeps} -- one NestedSet per dep with
   * failures to propagate. Otherwise if {@code --allow_analysis_failures} is false, returns the
   * empty set.
   */
  private ImmutableList<NestedSet<AnalysisFailure>> depAnalysisFailures(
      RuleContext ruleContext, Iterable<? extends TransitiveInfoCollection> extraDeps) {
    if (ruleContext.getConfiguration().allowAnalysisFailures()) {
      ImmutableList.Builder<NestedSet<AnalysisFailure>> analysisFailures = ImmutableList.builder();
      Iterable<? extends TransitiveInfoCollection> infoCollections =
          Iterables.concat(ruleContext.getConfiguredTargetMap().values(), extraDeps);
      for (TransitiveInfoCollection infoCollection : infoCollections) {
        AnalysisFailureInfo failureInfo =
            infoCollection.get(AnalysisFailureInfo.STARLARK_CONSTRUCTOR);
        if (failureInfo != null) {
          analysisFailures.add(failureInfo.getCausesNestedSet());
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
      throws ActionConflictException, InterruptedException {
    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext);
    builder.addNativeDeclaredProvider(AnalysisFailureInfo.forAnalysisFailureSets(analysisFailures));
    builder.addProvider(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY));
    return builder.build();
  }

  /**
   * Returns a {@link ConfiguredTarget} which indicates that an analysis error occurred in
   * processing the target. In most cases, this returns null, which signals to callers that the
   * target failed to build and thus the build should fail. However, if analysis failures are
   * allowed in this build, this returns a stub {@link ConfiguredTarget} which contains information
   * about the failure.
   */
  @Nullable
  private ConfiguredTarget erroredConfiguredTarget(RuleContext ruleContext)
      throws ActionConflictException, InterruptedException {
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
      if (DependencyKind.isToolchain(entry.getKey())) {
        continue;
      }
      Attribute attribute = entry.getKey().getAttribute();
      result.put(attribute, entry.getValue());
    }

    return result;
  }

  /**
   * Constructs a {@link ConfiguredAspect}. Returns null if an error occurs; in that case, {@code
   * aspectFactory} should call one of the error reporting methods of {@link RuleContext}.
   */
  public ConfiguredAspect createAspect(
      AnalysisEnvironment env,
      ConfiguredTargetAndData associatedTarget,
      ImmutableList<Aspect> aspectPath,
      ConfiguredAspectFactory aspectFactory,
      Aspect aspect,
      OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> prerequisiteMap,
      ConfigConditions configConditions,
      @Nullable ResolvedToolchainContext toolchainContext,
      BuildConfiguration aspectConfiguration,
      BuildConfiguration hostConfiguration,
      AspectValueKey.AspectKey aspectKey)
      throws InterruptedException, ActionConflictException, InvalidExecGroupException {

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
            .setToolsRepository(ruleClassProvider.getToolsRepository())
            .setStarlarkSemantics(env.getStarlarkSemantics())
            .setMutability(Mutability.create("aspect"))
            .setVisibility(
                convertVisibility(
                    prerequisiteMap, env.getEventHandler(), associatedTarget.getTarget(), null))
            .setPrerequisites(
                transformPrerequisiteMap(prerequisiteMap, associatedTarget.getTarget()))
            .setAspectAttributes(aspectAttributes)
            .setConfigConditions(configConditions)
            .setUniversalFragments(ruleClassProvider.getUniversalFragments())
            .setToolchainContext(toolchainContext)
            // TODO(b/161222568): Implement the exec_properties attr for aspects and read its value
            // here.
            .setExecProperties(ImmutableMap.of())
            .setConstraintSemantics(ruleClassProvider.getConstraintSemantics())
            .setRequiredConfigFragments(
                RequiredFragmentsUtil.getRequiredFragments(
                    aspect,
                    associatedTarget.getTarget().getAssociatedRule(),
                    aspectConfiguration,
                    ruleClassProvider.getUniversalFragments(),
                    aspect.getDefinition().getConfigurationFragmentPolicy(),
                    configConditions.asProviders(),
                    prerequisiteMap.values()))
            .build();

    // If allowing analysis failures, targets should be created as normal as possible, and errors
    // will be propagated via a hook elsewhere as AnalysisFailureInfo.
    boolean allowAnalysisFailures = ruleContext.getConfiguration().allowAnalysisFailures();

    List<NestedSet<AnalysisFailure>> analysisFailures =
        depAnalysisFailures(ruleContext, ImmutableList.of(associatedTarget.getConfiguredTarget()));
    if (!analysisFailures.isEmpty()) {
      return erroredConfiguredAspectWithFailures(ruleContext, analysisFailures);
    }
    if (ruleContext.hasErrors() && !allowAnalysisFailures) {
      return erroredConfiguredAspect(ruleContext);
    }

    ConfiguredAspect configuredAspect = null;
    try {
      configuredAspect =
          aspectFactory.create(
              associatedTarget,
              ruleContext,
              aspect.getParameters(),
              ruleClassProvider.getToolsRepository());
    } finally {
      ruleContext.close();
    }
    if (configuredAspect == null) {
      return erroredConfiguredAspect(ruleContext);
    }

    validateAdvertisedProviders(
        configuredAspect,
        aspectKey,
        aspect.getDefinition().getAdvertisedProviders(),
        associatedTarget.getTarget(),
        env.getEventHandler());
    return configuredAspect;
  }

  private ConfiguredAspect erroredConfiguredAspectWithFailures(
      RuleContext ruleContext, List<NestedSet<AnalysisFailure>> analysisFailures)
      throws ActionConflictException, InterruptedException {
    ConfiguredAspect.Builder builder = new ConfiguredAspect.Builder(ruleContext);
    builder.addNativeDeclaredProvider(AnalysisFailureInfo.forAnalysisFailureSets(analysisFailures));
    // Unlike erroredConfiguredTargetAspectWithFailures, we do not add a RunfilesProvider; that
    // would result in a RunfilesProvider being provided twice in the merged configured target.
    return builder.build();
  }

  /**
   * Returns a {@link ConfiguredAspect} which indicates that an analysis error occurred in
   * processing the aspect. In most cases, this returns null, which signals to callers that the
   * target failed to build and thus the build should fail. However, if analysis failures are
   * allowed in this build, this returns a stub {@link ConfiguredAspect} which contains information
   * about the failure.
   */
  @Nullable
  private ConfiguredAspect erroredConfiguredAspect(RuleContext ruleContext)
      throws ActionConflictException, InterruptedException {
    if (ruleContext.getConfiguration().allowAnalysisFailures()) {
      ImmutableList.Builder<AnalysisFailure> analysisFailures = ImmutableList.builder();

      for (String errorMessage : ruleContext.getSuppressedErrorMessages()) {
        analysisFailures.add(new AnalysisFailure(ruleContext.getLabel(), errorMessage));
      }
      ConfiguredAspect.Builder builder = new ConfiguredAspect.Builder(ruleContext);
      builder.addNativeDeclaredProvider(
          AnalysisFailureInfo.forAnalysisFailures(analysisFailures.build()));
      // Unlike erroredConfiguredTarget, we do not add a RunfilesProvider; that would result in a
      // RunfilesProvider being provided twice in the merged configured target.
      return builder.build();
    } else {
      // Returning a null ConfiguredAspect is an indication a rule error occurred. Exceptions are
      // not propagated, as this would show a nasty stack trace to users, and only provide info
      // on one specific failure with poor messaging. By returning null, the caller can
      // inspect ruleContext for multiple errors and output thorough messaging on each.
      return null;
    }
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
      AspectValueKey.AspectKey aspectKey,
      AdvertisedProviderSet advertisedProviders,
      Target target,
      EventHandler eventHandler) {
    if (advertisedProviders.canHaveAnyProvider()) {
      return;
    }
    for (Class<?> aClass : advertisedProviders.getBuiltinProviders()) {
      if (configuredAspect.getProvider(aClass.asSubclass(TransitiveInfoProvider.class)) == null) {
        eventHandler.handle(
            Event.error(
                target.getLocation(),
                String.format(
                    "Aspect '%s', applied to '%s', does not provide advertised provider '%s'",
                    aspectKey.getAspectClass().getName(),
                    target.getLabel(),
                    aClass.getSimpleName())));
      }
    }

    for (StarlarkProviderIdentifier providerId : advertisedProviders.getStarlarkProviders()) {
      if (configuredAspect.get(providerId) == null) {
        eventHandler.handle(
            Event.error(
                target.getLocation(),
                String.format(
                    "Aspect '%s', applied to '%s', does not provide advertised provider '%s'",
                    aspectKey.getAspectClass().getName(), target.getLabel(), providerId)));
      }
    }
  }

  /**
   * A pseudo-implementation for configured targets that creates fail actions for all declared
   * outputs, both implicit and explicit, due to a missing fragment class.
   */
  private static ConfiguredTarget createFailConfiguredTargetForMissingFragmentClass(
      RuleContext ruleContext, Class<?> missingFragmentClass) throws InterruptedException {
    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext);
    if (!ruleContext.getOutputArtifacts().isEmpty()) {
      ruleContext.registerAction(
          new FailAction(
              ruleContext.getActionOwner(),
              ruleContext.getOutputArtifacts(),
              "Missing fragment class: " + missingFragmentClass.getName(),
              Code.FRAGMENT_CLASS_MISSING));
    }
    builder.add(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY));
    try {
      return builder.build();
    } catch (ActionConflictException e) {
      throw new IllegalStateException(
          "Can't have an action conflict with one action: " + ruleContext.getLabel(), e);
    }
  }
}
