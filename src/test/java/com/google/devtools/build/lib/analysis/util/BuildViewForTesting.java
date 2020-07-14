// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.util;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.PackageRoots;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.AnalysisOptions;
import com.google.devtools.build.lib.analysis.AnalysisResult;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BuildView;
import com.google.devtools.build.lib.analysis.CachingAnalysisEnvironment;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.DependencyKey;
import com.google.devtools.build.lib.analysis.DependencyKind;
import com.google.devtools.build.lib.analysis.DependencyResolver;
import com.google.devtools.build.lib.analysis.InconsistentAspectOrderException;
import com.google.devtools.build.lib.analysis.ResolvedToolchainContext;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleContext.InvalidExecGroupException;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.ToolchainCollection;
import com.google.devtools.build.lib.analysis.ToolchainContext;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.ConfigurationResolver;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.TransitionResolver;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.starlark.StarlarkTransition;
import com.google.devtools.build.lib.analysis.test.CoverageReportActionFactory;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ExecGroup;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.PackageSpecification;
import com.google.devtools.build.lib.packages.PackageSpecification.PackageGroupContents;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.AspectValueKey.AspectKey;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.SkyFunctionEnvironmentForTesting;
import com.google.devtools.build.lib.skyframe.SkyframeBuildView;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.TargetPatternPhaseValue;
import com.google.devtools.build.lib.skyframe.ToolchainContextKey;
import com.google.devtools.build.lib.skyframe.ToolchainException;
import com.google.devtools.build.lib.skyframe.UnloadedToolchainContext;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.ValueOrException;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * A util class that contains all the helper stuff previously in BuildView that only exists to give
 * tests access to Skyframe internals. The code largely predates the introduction of Skyframe, and
 * mostly exists to avoid having to rewrite our tests to work with Skyframe natively.
 */
public class BuildViewForTesting {
  private final BuildView buildView;
  private final SkyframeExecutor skyframeExecutor;
  private final SkyframeBuildView skyframeBuildView;

  private final ConfiguredRuleClassProvider ruleClassProvider;

  public BuildViewForTesting(
      BlazeDirectories directories,
      ConfiguredRuleClassProvider ruleClassProvider,
      SkyframeExecutor skyframeExecutor,
      CoverageReportActionFactory coverageReportActionFactory) {
    this.buildView =
        new BuildView(
            directories,
            ruleClassProvider,
            skyframeExecutor,
            coverageReportActionFactory);
    this.ruleClassProvider = ruleClassProvider;
    this.skyframeExecutor = Preconditions.checkNotNull(skyframeExecutor);
    this.skyframeBuildView = skyframeExecutor.getSkyframeBuildView();
  }

  @VisibleForTesting
  public Set<SkyKey> getSkyframeEvaluatedTargetKeysForTesting() {
    return skyframeBuildView.getEvaluatedTargetKeys();
  }

  /**
   * Returns whether the given configured target has errors.
   */
  @VisibleForTesting
  public boolean hasErrors(ConfiguredTarget configuredTarget) {
    return configuredTarget == null;
  }

  @ThreadCompatible
  public AnalysisResult update(
      TargetPatternPhaseValue loadingResult,
      BuildOptions targetOptions,
      Set<String> multiCpu,
      List<String> aspects,
      AnalysisOptions viewOptions,
      boolean keepGoing,
      int loadingPhaseThreads,
      TopLevelArtifactContext topLevelOptions,
      ExtendedEventHandler eventHandler,
      EventBus eventBus)
      throws ViewCreationFailedException, InterruptedException, InvalidConfigurationException {
    return buildView.update(
        loadingResult,
        targetOptions,
        multiCpu,
        aspects,
        viewOptions,
        keepGoing,
        loadingPhaseThreads,
        topLevelOptions,
        eventHandler,
        eventBus);
  }

  /** Sets the configurations. Not thread-safe. DO NOT CALL except from tests! */
  @VisibleForTesting
  public void setConfigurationsForTesting(
      EventHandler eventHandler, BuildConfigurationCollection configurations) {
    skyframeBuildView.setConfigurations(
        eventHandler, configurations, /* maxDifferencesToShow */ -1);
  }

  public ArtifactFactory getArtifactFactory() {
    return skyframeBuildView.getArtifactFactory();
  }

  /**
   * Gets a configuration for the given target.
   *
   * <p>If {@link BuildConfiguration#trimConfigurations()} is true, the configuration only includes
   * the fragments needed by the fragment and its transitive closure. Else unconditionally includes
   * all fragments.
   */
  @VisibleForTesting
  public BuildConfiguration getConfigurationForTesting(
      Target target, BuildConfiguration config, ExtendedEventHandler eventHandler)
      throws InvalidConfigurationException {
    List<TargetAndConfiguration> node =
        ImmutableList.of(new TargetAndConfiguration(target, config));
    Collection<TargetAndConfiguration> configs =
        ConfigurationResolver.getConfigurationsFromExecutor(
                node,
                AnalysisUtils.targetsToDeps(new LinkedHashSet<>(node), ruleClassProvider),
                eventHandler,
                skyframeExecutor)
            .getTargetsAndConfigs();
    return configs.iterator().next().getConfiguration();
  }

  /**
   * Sets the possible artifact roots in the artifact factory. This allows the factory to resolve
   * paths with unknown roots to artifacts.
   */
  @VisibleForTesting // for BuildViewTestCase
  public void setArtifactRoots(PackageRoots packageRoots) {
    getArtifactFactory().setPackageRoots(packageRoots.getPackageRootLookup());
  }

  @VisibleForTesting
  public Collection<ConfiguredTarget> getDirectPrerequisitesForTesting(
      ExtendedEventHandler eventHandler,
      ConfiguredTarget ct,
      BuildConfigurationCollection configurations)
      throws EvalException, InvalidConfigurationException, InterruptedException,
          InconsistentAspectOrderException, StarlarkTransition.TransitionException {
    return Collections2.transform(
        getConfiguredTargetAndDataDirectPrerequisitesForTesting(eventHandler, ct, configurations),
        ConfiguredTargetAndData::getConfiguredTarget);
  }

  // TODO(janakr): pass the configuration in as a parameter here and above.
  private Collection<ConfiguredTargetAndData>
      getConfiguredTargetAndDataDirectPrerequisitesForTesting(
          ExtendedEventHandler eventHandler,
          ConfiguredTarget ct,
          BuildConfigurationCollection configurations)
          throws EvalException, InvalidConfigurationException, InterruptedException,
              InconsistentAspectOrderException, StarlarkTransition.TransitionException {
    return getConfiguredTargetAndDataDirectPrerequisitesForTesting(
        eventHandler, ct, ct.getConfigurationKey(), configurations);
  }

  @VisibleForTesting
  public Collection<ConfiguredTargetAndData>
      getConfiguredTargetAndDataDirectPrerequisitesForTesting(
          ExtendedEventHandler eventHandler,
          ConfiguredTargetAndData ct,
          BuildConfigurationCollection configurations)
          throws EvalException, InvalidConfigurationException, InterruptedException,
              InconsistentAspectOrderException, StarlarkTransition.TransitionException {
    return getConfiguredTargetAndDataDirectPrerequisitesForTesting(
        eventHandler,
        ct.getConfiguredTarget(),
        ct.getConfiguredTarget().getConfigurationKey(),
        configurations);
  }

  private Collection<ConfiguredTargetAndData>
      getConfiguredTargetAndDataDirectPrerequisitesForTesting(
          ExtendedEventHandler eventHandler,
          ConfiguredTarget ct,
          BuildConfigurationValue.Key configuration,
          BuildConfigurationCollection configurations)
          throws EvalException, InvalidConfigurationException, InterruptedException,
              InconsistentAspectOrderException, StarlarkTransition.TransitionException {
    return skyframeExecutor.getConfiguredTargetsForTesting(
        eventHandler,
        configuration,
        ImmutableSet.copyOf(
            getDirectPrerequisiteDependenciesForTesting(
                    eventHandler, ct, configurations, /* toolchainContext= */ null)
                .values()));
  }

  @VisibleForTesting
  public OrderedSetMultimap<DependencyKind, DependencyKey>
      getDirectPrerequisiteDependenciesForTesting(
          final ExtendedEventHandler eventHandler,
          final ConfiguredTarget ct,
          BuildConfigurationCollection configurations,
          @Nullable ToolchainCollection<ToolchainContext> toolchainContexts)
          throws EvalException, InterruptedException, InconsistentAspectOrderException,
              StarlarkTransition.TransitionException, InvalidConfigurationException {

    Target target = null;
    try {
      target = skyframeExecutor.getPackageManager().getTarget(eventHandler, ct.getLabel());
    } catch (NoSuchPackageException | NoSuchTargetException | InterruptedException e) {
      eventHandler.handle(
          Event.error("Failed to get target from package during prerequisite analysis." + e));
      return OrderedSetMultimap.create();
    }

    if (!(target instanceof Rule)) {
      return OrderedSetMultimap.create();
    }

    class SilentDependencyResolver extends DependencyResolver {
      private SilentDependencyResolver() {
      }

      @Override
      protected Map<Label, Target> getTargets(
          OrderedSetMultimap<DependencyKind, Label> labelMap,
          TargetAndConfiguration fromNode,
          NestedSetBuilder<Cause> rootCauses) {
        return labelMap.values().stream()
            .distinct()
            .collect(
                Collectors.toMap(
                    Function.identity(),
                    label -> {
                      try {
                        return skyframeExecutor.getPackageManager().getTarget(eventHandler, label);
                      } catch (NoSuchPackageException
                          | NoSuchTargetException
                          | InterruptedException e) {
                        throw new IllegalStateException(e);
                      }
                    }));
      }
    }

    DependencyResolver dependencyResolver = new SilentDependencyResolver();
    BuildConfiguration configuration =
        skyframeExecutor.getConfiguration(eventHandler, ct.getConfigurationKey());
    TargetAndConfiguration ctgNode = new TargetAndConfiguration(target, configuration);
    return dependencyResolver.dependentNodeMap(
        ctgNode,
        configurations.getHostConfiguration(),
        /*aspect=*/ null,
        getConfigurableAttributeKeysForTesting(eventHandler, ctgNode),
        toolchainContexts,
        DependencyResolver.shouldUseToolchainTransition(configuration, target),
        ruleClassProvider.getTrimmingTransitionFactory());
  }

  /**
   * Returns ConfigMatchingProvider instances corresponding to the configurable attribute keys
   * present in this rule's attributes.
   */
  private ImmutableMap<Label, ConfigMatchingProvider> getConfigurableAttributeKeysForTesting(
      ExtendedEventHandler eventHandler, TargetAndConfiguration ctg)
      throws StarlarkTransition.TransitionException, InvalidConfigurationException,
          InterruptedException {
    if (!(ctg.getTarget() instanceof Rule)) {
      return ImmutableMap.of();
    }
    Rule rule = (Rule) ctg.getTarget();
    Map<Label, ConfigMatchingProvider> keys = new LinkedHashMap<>();
    RawAttributeMapper mapper = RawAttributeMapper.of(rule);
    for (Attribute attribute : rule.getAttributes()) {
      for (Label label : mapper.getConfigurabilityKeys(attribute.getName(), attribute.getType())) {
        if (BuildType.Selector.isReservedLabel(label)) {
          continue;
        }
        ConfiguredTarget ct = getConfiguredTargetForTesting(
            eventHandler, label, ctg.getConfiguration());
        keys.put(label, Preconditions.checkNotNull(ct.getProvider(ConfigMatchingProvider.class)));
      }
    }
    return ImmutableMap.copyOf(keys);
  }

  private OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> getPrerequisiteMapForTesting(
      final ExtendedEventHandler eventHandler,
      ConfiguredTarget target,
      BuildConfigurationCollection configurations,
      @Nullable ToolchainCollection<ToolchainContext> toolchainContexts)
      throws EvalException, InvalidConfigurationException, InterruptedException,
          InconsistentAspectOrderException, StarlarkTransition.TransitionException {
    OrderedSetMultimap<DependencyKind, DependencyKey> depNodeNames =
        getDirectPrerequisiteDependenciesForTesting(
            eventHandler, target, configurations, toolchainContexts);

    ImmutableMultimap<DependencyKey, ConfiguredTargetAndData> cts =
        skyframeExecutor.getConfiguredTargetMapForTesting(
            eventHandler, target.getConfigurationKey(), ImmutableSet.copyOf(depNodeNames.values()));

    OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> result =
        OrderedSetMultimap.create();
    for (Map.Entry<DependencyKind, DependencyKey> entry : depNodeNames.entries()) {
      result.putAll(entry.getKey(), cts.get(entry.getValue()));
    }
    return result;
  }

  private ConfigurationTransition getTopLevelTransitionForTarget(
      Label label, BuildConfiguration config, ExtendedEventHandler handler) {
    Target target;
    try {
      target = skyframeExecutor.getPackageManager().getTarget(handler, label);
    } catch (NoSuchPackageException | NoSuchTargetException e) {
      // TODO(bazel-team): refactor this method so we actually throw an exception here (likely
      // {@link TransitionException}. Every version of getConfiguredTarget runs through this
      // method and many test cases rely on not erroring out here so be able to reach an error
      // later on.
      return NoTransition.INSTANCE;
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new AssertionError("Configuration of " + label + " interrupted");
    }
    // Return early if a rule target is in error. We don't want whatever caused the rule error to
    // also cause problems in computing the transition (e.g. an unchecked exception).
    if (target instanceof Rule && ((Rule) target).containsErrors()) {
      return null;
    }
    return TransitionResolver.evaluateTransition(
        config, NoTransition.INSTANCE, target, ruleClassProvider.getTrimmingTransitionFactory());
  }

  /**
   * Returns a configured target for the specified target and configuration. If the target in
   * question has a top-level rule class transition, that transition is applied in the returned
   * ConfiguredTarget.
   *
   * <p>Returns {@code null} if something goes wrong.
   */
  @VisibleForTesting
  public ConfiguredTarget getConfiguredTargetForTesting(
      ExtendedEventHandler eventHandler, Label label, BuildConfiguration config)
      throws StarlarkTransition.TransitionException, InvalidConfigurationException,
          InterruptedException {
    ConfigurationTransition transition =
        getTopLevelTransitionForTarget(label, config, eventHandler);
    if (transition == null) {
      return null;
    }
    return skyframeExecutor.getConfiguredTargetForTesting(eventHandler, label, config, transition);
  }

  @VisibleForTesting
  ConfiguredTargetAndData getConfiguredTargetAndDataForTesting(
      ExtendedEventHandler eventHandler, Label label, BuildConfiguration config)
      throws StarlarkTransition.TransitionException, InvalidConfigurationException,
          InterruptedException {
    ConfigurationTransition transition =
        getTopLevelTransitionForTarget(label, config, eventHandler);
    if (transition == null) {
      return null;
    }
    return skyframeExecutor.getConfiguredTargetAndDataForTesting(
        eventHandler, label, config, transition);
  }

  /**
   * Returns a RuleContext which is the same as the original RuleContext of the target parameter.
   */
  @VisibleForTesting
  public RuleContext getRuleContextForTesting(
      ConfiguredTarget target,
      StoredEventHandler eventHandler,
      BuildConfigurationCollection configurations)
      throws EvalException, InvalidConfigurationException, InterruptedException,
          InconsistentAspectOrderException, ToolchainException,
          StarlarkTransition.TransitionException, InvalidExecGroupException {
    BuildConfiguration targetConfig =
        skyframeExecutor.getConfiguration(eventHandler, target.getConfigurationKey());
    CachingAnalysisEnvironment env =
        new CachingAnalysisEnvironment(
            getArtifactFactory(),
            skyframeExecutor.getActionKeyContext(),
            ConfiguredTargetKey.builder()
                .setLabel(target.getLabel())
                .setConfiguration(targetConfig)
                .build(),
            /*isSystemEnv=*/ false,
            targetConfig.extendedSanityChecks(),
            targetConfig.allowAnalysisFailures(),
            eventHandler,
            skyframeExecutor.getSkyFunctionEnvironmentForTesting(eventHandler));
    return getRuleContextForTesting(eventHandler, target, env, configurations);
  }

  /**
   * Creates and returns a rule context that is equivalent to the one that was used to create the
   * given configured target.
   */
  @VisibleForTesting
  public RuleContext getRuleContextForTesting(
      ExtendedEventHandler eventHandler,
      ConfiguredTarget configuredTarget,
      AnalysisEnvironment env,
      BuildConfigurationCollection configurations)
      throws EvalException, InvalidConfigurationException, InterruptedException,
          InconsistentAspectOrderException, ToolchainException,
          StarlarkTransition.TransitionException, InvalidExecGroupException {
    BuildConfiguration targetConfig =
        skyframeExecutor.getConfiguration(eventHandler, configuredTarget.getConfigurationKey());
    Target target = null;
    try {
      target =
          skyframeExecutor.getPackageManager().getTarget(eventHandler, configuredTarget.getLabel());
    } catch (NoSuchPackageException | NoSuchTargetException e) {
      eventHandler.handle(
          Event.error("Failed to get target when trying to get rule context for testing"));
      throw new IllegalStateException(e);
    }
    ImmutableSet<Label> requiredToolchains =
        target.getAssociatedRule().getRuleClassObject().getRequiredToolchains();
    ImmutableMap<String, ExecGroup> execGroups =
        target.getAssociatedRule().getRuleClassObject().getExecGroups();
    SkyFunctionEnvironmentForTesting skyfunctionEnvironment =
        skyframeExecutor.getSkyFunctionEnvironmentForTesting(eventHandler);

    Map<String, ToolchainContextKey> toolchainContextKeys = new HashMap<>();
    for (Map.Entry<String, ExecGroup> execGroup : execGroups.entrySet()) {
      toolchainContextKeys.put(
          execGroup.getKey(),
          ToolchainContextKey.key()
              .configurationKey(BuildConfigurationValue.key(targetConfig))
              .requiredToolchainTypeLabels(execGroup.getValue().requiredToolchains())
              .build());
    }
    String targetUnloadedToolchainContextKey = "target-unloaded-toolchain-context";
    toolchainContextKeys.put(
        targetUnloadedToolchainContextKey,
        ToolchainContextKey.key()
            .configurationKey(BuildConfigurationValue.key(targetConfig))
            .requiredToolchainTypeLabels(requiredToolchains)
            .build());

    Map<SkyKey, ValueOrException<ToolchainException>> values =
        skyfunctionEnvironment.getValuesOrThrow(
            toolchainContextKeys.values(), ToolchainException.class);

    ToolchainCollection.Builder<UnloadedToolchainContext> unloadedToolchainContexts =
        ToolchainCollection.builder();
    for (Map.Entry<String, ToolchainContextKey> unloadedToolchainContextKey :
        toolchainContextKeys.entrySet()) {
      UnloadedToolchainContext unloadedToolchainContext =
          (UnloadedToolchainContext) values.get(unloadedToolchainContextKey.getValue()).get();
      String execGroup = unloadedToolchainContextKey.getKey();
      if (execGroup.equals(targetUnloadedToolchainContextKey)) {
        unloadedToolchainContexts.addDefaultContext(unloadedToolchainContext);
      } else {
        unloadedToolchainContexts.addContext(execGroup, unloadedToolchainContext);
      }
    }

    ToolchainCollection<UnloadedToolchainContext> unloadedToolchainCollection =
        unloadedToolchainContexts.build();

    OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> prerequisiteMap =
        getPrerequisiteMapForTesting(
            eventHandler,
            configuredTarget,
            configurations,
            unloadedToolchainCollection.asToolchainContexts());
    String targetDescription = target.toString();

    ToolchainCollection.Builder<ResolvedToolchainContext> resolvedToolchainContext =
        ToolchainCollection.builder();
    for (Map.Entry<String, UnloadedToolchainContext> unloadedToolchainContext :
        unloadedToolchainCollection.getContextMap().entrySet()) {
      ResolvedToolchainContext toolchainContext =
          ResolvedToolchainContext.load(
              target.getPackage().getRepositoryMapping(),
              unloadedToolchainContext.getValue(),
              targetDescription,
              prerequisiteMap.get(DependencyKind.TOOLCHAIN_DEPENDENCY));
      resolvedToolchainContext.addContext(unloadedToolchainContext.getKey(), toolchainContext);
    }

    return new RuleContext.Builder(
            env,
            target,
            ImmutableList.of(),
            targetConfig,
            configurations.getHostConfiguration(),
            ruleClassProvider.getPrerequisiteValidator(),
            target.getAssociatedRule().getRuleClassObject().getConfigurationFragmentPolicy(),
            ConfiguredTargetKey.builder()
                .setConfiguredTarget(configuredTarget)
                .setConfigurationKey(configuredTarget.getConfigurationKey())
                .build())
        .setVisibility(
            NestedSetBuilder.create(
                Order.STABLE_ORDER,
                PackageGroupContents.create(ImmutableList.of(PackageSpecification.everything()))))
        .setPrerequisites(
            ConfiguredTargetFactory.transformPrerequisiteMap(
                prerequisiteMap, target.getAssociatedRule()))
        .setConfigConditions(ImmutableMap.<Label, ConfigMatchingProvider>of())
        .setUniversalFragments(ruleClassProvider.getUniversalFragments())
        .setToolchainContexts(resolvedToolchainContext.build())
        .setConstraintSemantics(ruleClassProvider.getConstraintSemantics())
        .build();
  }

  /**
   * For a configured target dependentTarget, returns the desired configured target that is depended
   * upon. Useful for obtaining the a target with aspects required by the dependent.
   */
  @VisibleForTesting
  public ConfiguredTarget getPrerequisiteConfiguredTargetForTesting(
      ExtendedEventHandler eventHandler,
      ConfiguredTarget dependentTarget,
      Label desiredTarget,
      BuildConfigurationCollection configurations)
      throws EvalException, InvalidConfigurationException, InterruptedException,
          InconsistentAspectOrderException, StarlarkTransition.TransitionException {
    Collection<ConfiguredTargetAndData> configuredTargets =
        getPrerequisiteMapForTesting(
                eventHandler, dependentTarget, configurations, /*toolchainContext=*/ null)
            .values();
    for (ConfiguredTargetAndData ct : configuredTargets) {
      if (ct.getTarget().getLabel().equals(desiredTarget)) {
        return ct.getConfiguredTarget();
      }
    }
    return null;
  }

  /** Clears the analysis cache as in --discard_analysis_cache. */
  @VisibleForTesting
  void clearAnalysisCache(
      Collection<ConfiguredTarget> topLevelTargets, ImmutableSet<AspectKey> topLevelAspects) {
    skyframeBuildView.clearAnalysisCache(topLevelTargets, topLevelAspects);
  }
}
