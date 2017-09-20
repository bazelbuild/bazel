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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Actions.GeneratingActions;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.AspectResolver;
import com.google.devtools.build.lib.analysis.CachingAnalysisEnvironment;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.Dependency;
import com.google.devtools.build.lib.analysis.DependencyResolver.InconsistentAspectOrderException;
import com.google.devtools.build.lib.analysis.LabelAndConfiguration;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.ToolchainContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.ConfigurationResolver;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.configuredtargets.MergedConfiguredTarget.DuplicateException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.skyframe.AspectFunction.AspectCreationException;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor.BuildViewProvider;
import com.google.devtools.build.lib.skyframe.ToolchainUtil.ToolchainContextException;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Semaphore;
import javax.annotation.Nullable;

/**
 * SkyFunction for {@link ConfiguredTargetValue}s.
 *
 * <p>This class, together with {@link AspectFunction} drives the analysis phase. For more
 * information, see {@link com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory}.
 *
 * @see com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory
 */
public final class ConfiguredTargetFunction implements SkyFunction {
  // This construction is a bit funky, but guarantees that the Object reference here is globally
  // unique.
  static final ImmutableMap<Label, ConfigMatchingProvider> NO_CONFIG_CONDITIONS =
      ImmutableMap.<Label, ConfigMatchingProvider>of();

  /**
   * Exception class that signals an error during the evaluation of a dependency.
   */
  public static class DependencyEvaluationException extends Exception {
    public DependencyEvaluationException(InvalidConfigurationException cause) {
      super(cause);
    }

    public DependencyEvaluationException(ConfiguredValueCreationException cause) {
      super(cause);
    }

    public DependencyEvaluationException(InconsistentAspectOrderException cause) {
      super(cause);
    }

    @Override
    public synchronized Exception getCause() {
      return (Exception) super.getCause();
    }
  }

  private final BuildViewProvider buildViewProvider;
  private final RuleClassProvider ruleClassProvider;
  private final Semaphore cpuBoundSemaphore;
  private final Supplier<Boolean> removeActionsAfterEvaluation;

  ConfiguredTargetFunction(
      BuildViewProvider buildViewProvider,
      RuleClassProvider ruleClassProvider,
      Semaphore cpuBoundSemaphore,
      Supplier<Boolean> removeActionsAfterEvaluation) {
    this.buildViewProvider = buildViewProvider;
    this.ruleClassProvider = ruleClassProvider;
    this.cpuBoundSemaphore = cpuBoundSemaphore;
    this.removeActionsAfterEvaluation = Preconditions.checkNotNull(removeActionsAfterEvaluation);
  }

  @Override
  public SkyValue compute(SkyKey key, Environment env) throws ConfiguredTargetFunctionException,
      InterruptedException {
    SkyframeBuildView view = buildViewProvider.getSkyframeBuildView();
    NestedSetBuilder<Package> transitivePackages = NestedSetBuilder.stableOrder();
    NestedSetBuilder<Label> transitiveLoadingRootCauses = NestedSetBuilder.stableOrder();
    ConfiguredTargetKey configuredTargetKey = (ConfiguredTargetKey) key.argument();
    LabelAndConfiguration lc = LabelAndConfiguration.of(
        configuredTargetKey.getLabel(), configuredTargetKey.getConfiguration());

    BuildConfiguration configuration = lc.getConfiguration();

    PackageValue packageValue =
        (PackageValue) env.getValue(PackageValue.key(lc.getLabel().getPackageIdentifier()));
    if (packageValue == null) {
      return null;
    }

    // TODO(ulfjack): This tries to match the logic in TransitiveTargetFunction /
    // TargetMarkerFunction. Maybe we can merge the two?
    Package pkg = packageValue.getPackage();
    Target target;
    try {
      target = pkg.getTarget(lc.getLabel().getName());
    } catch (NoSuchTargetException e) {
      throw new ConfiguredTargetFunctionException(
          new ConfiguredValueCreationException(e.getMessage(), lc.getLabel()));
    }
    if (pkg.containsErrors()) {
      transitiveLoadingRootCauses.add(lc.getLabel());
    }
    transitivePackages.add(pkg);
    // TODO(bazel-team): This is problematic - we create the right key, but then end up with a value
    // that doesn't match; we can even have the same value multiple times. However, I think it's
    // only triggered in tests (i.e., in normal operation, the configuration passed in is already
    // null).
    if (!target.isConfigurable()) {
      configuration = null;
    }

    // This line is only needed for accurate error messaging. Say this target has a circular
    // dependency with one of its deps. With this line, loading this target fails so Bazel
    // associates the corresponding error with this target, as expected. Without this line,
    // the first TransitiveTargetValue call happens on its dep (in trimConfigurations), so Bazel
    // associates the error with the dep, which is misleading.
    if (configuration != null && configuration.trimConfigurations()
        && env.getValue(TransitiveTargetValue.key(lc.getLabel())) == null) {
      return null;
    }

    TargetAndConfiguration ctgValue = new TargetAndConfiguration(target, configuration);

    SkyframeDependencyResolver resolver = view.createDependencyResolver(env);

    ToolchainContext toolchainContext = null;

    // TODO(janakr): this acquire() call may tie up this thread indefinitely, reducing the
    // parallelism of Skyframe. This is a strict improvement over the prior state of the code, in
    // which we ran with #processors threads, but ideally we would call #tryAcquire here, and if we
    // failed, would exit this SkyFunction and restart it when permits were available.
    cpuBoundSemaphore.acquire();
    try {
      // Get the configuration targets that trigger this rule's configurable attributes.
      ImmutableMap<Label, ConfigMatchingProvider> configConditions = getConfigConditions(
          ctgValue.getTarget(), env, resolver, ctgValue, transitivePackages,
          transitiveLoadingRootCauses);
      if (env.valuesMissing()) {
        return null;
      }
      // TODO(ulfjack): ConfiguredAttributeMapper (indirectly used from computeDependencies) isn't
      // safe to use if there are missing config conditions, so we stop here, but only if there are
      // config conditions - though note that we can't check if configConditions is non-empty - it
      // may be empty for other reasons. It would be better to continue here so that we can collect
      // more root causes during computeDependencies.
      // Note that this doesn't apply to AspectFunction, because aspects can't have configurable
      // attributes.
      if (!transitiveLoadingRootCauses.isEmpty() && configConditions != NO_CONFIG_CONDITIONS) {
        throw new ConfiguredTargetFunctionException(
            new ConfiguredValueCreationException(transitiveLoadingRootCauses.build()));
      }

      // Determine what toolchains are needed by this target.
      if (target instanceof Rule) {
        Rule rule = ((Rule) target);
        ImmutableSet<Label> requiredToolchains = rule.getRuleClassObject().getRequiredToolchains();
        toolchainContext =
            ToolchainUtil.createToolchainContext(
                env, rule.toString(), requiredToolchains, configuration);
        if (env.valuesMissing()) {
          return null;
        }
      }

      // Calculate the dependencies of this target.
      OrderedSetMultimap<Attribute, ConfiguredTarget> depValueMap =
          computeDependencies(
              env,
              resolver,
              ctgValue,
              ImmutableList.<Aspect>of(),
              configConditions,
              toolchainContext,
              ruleClassProvider,
              view.getHostConfiguration(configuration),
              transitivePackages,
              transitiveLoadingRootCauses);
      if (env.valuesMissing()) {
        return null;
      }
      if (!transitiveLoadingRootCauses.isEmpty()) {
        throw new ConfiguredTargetFunctionException(
            new ConfiguredValueCreationException(transitiveLoadingRootCauses.build()));
      }
      Preconditions.checkNotNull(depValueMap);
      ConfiguredTargetValue ans =
          createConfiguredTarget(
              view,
              env,
              target,
              configuration,
              depValueMap,
              configConditions,
              toolchainContext,
              transitivePackages);
      return ans;
    } catch (DependencyEvaluationException e) {
      if (e.getCause() instanceof ConfiguredValueCreationException) {
        ConfiguredValueCreationException cvce = (ConfiguredValueCreationException) e.getCause();

        // Check if this is caused by an unresolved toolchain, and report it as such.
        if (toolchainContext != null) {
          ImmutableSet.Builder<Label> causes = new ImmutableSet.Builder<Label>();
          if (cvce.getAnalysisRootCause() != null) {
            causes.add(cvce.getAnalysisRootCause());
          }
          if (!cvce.getRootCauses().isEmpty()) {
            causes.addAll(cvce.getRootCauses());
          }
          Set<Label> toolchainDependencyErrors =
              toolchainContext.filterToolchainLabels(causes.build());
          if (!toolchainDependencyErrors.isEmpty()) {
            env.getListener()
                .handle(
                    Event.error(
                        String.format(
                            "While resolving toolchains for target %s: %s",
                            target.getLabel(), e.getCause().getMessage())));
          }
        }

        throw new ConfiguredTargetFunctionException(cvce);
      } else if (e.getCause() instanceof InconsistentAspectOrderException) {
        InconsistentAspectOrderException cause = (InconsistentAspectOrderException) e.getCause();
        throw new ConfiguredTargetFunctionException(
            new ConfiguredValueCreationException(cause.getMessage(), target.getLabel()));
      } else if (e.getCause() instanceof InvalidConfigurationException) {
        InvalidConfigurationException cause = (InvalidConfigurationException) e.getCause();
        env.getListener().handle(Event.error(cause.getMessage()));
        throw new ConfiguredTargetFunctionException(
            new ConfiguredValueCreationException(cause.getMessage(), target.getLabel()));
      } else {
        // Unknown exception type.
        throw new ConfiguredTargetFunctionException(
            new ConfiguredValueCreationException(e.getMessage(), target.getLabel()));
      }
    } catch (AspectCreationException e) {
      // getAnalysisRootCause may be null if the analysis of the aspect itself failed.
      Label analysisRootCause = target.getLabel();
      if (e.getAnalysisRootCause() != null) {
        analysisRootCause = e.getAnalysisRootCause();
      }
      throw new ConfiguredTargetFunctionException(
          new ConfiguredValueCreationException(e.getMessage(), analysisRootCause));
    } catch (ToolchainContextException e) {
      // We need to throw a ConfiguredValueCreationException, so either find one or make one.
      ConfiguredValueCreationException cvce;
      if (e.getCause() instanceof ConfiguredValueCreationException) {
        cvce = (ConfiguredValueCreationException) e.getCause();
      } else {
        cvce = new ConfiguredValueCreationException(e.getCause().getMessage(), target.getLabel());
      }

      env.getListener()
          .handle(
              Event.error(
                  String.format(
                      "While resolving toolchains for target %s: %s",
                      target.getLabel(), e.getCause().getMessage())));
      throw new ConfiguredTargetFunctionException(cvce);
    } finally {
      cpuBoundSemaphore.release();
    }
  }

  /**
   * Computes the direct dependencies of a node in the configured target graph (a configured target
   * or an aspects).
   *
   * <p>Returns null if Skyframe hasn't evaluated the required dependencies yet. In this case, the
   * caller should also return null to Skyframe.
   *
   * @param env the Skyframe environment
   * @param resolver the dependency resolver
   * @param ctgValue the label and the configuration of the node
   * @param aspects
   * @param configConditions the configuration conditions for evaluating the attributes of the node
   * @param toolchainContext context information for required toolchains
   * @param ruleClassProvider rule class provider for determining the right configuration fragments
   *     to apply to deps
   * @param hostConfiguration the host configuration. There's a noticeable performance hit from
   *     instantiating this on demand for every dependency that wants it, so it's best to compute
   *     the host configuration as early as possible and pass this reference to all consumers
   */
  @Nullable
  static OrderedSetMultimap<Attribute, ConfiguredTarget> computeDependencies(
      Environment env,
      SkyframeDependencyResolver resolver,
      TargetAndConfiguration ctgValue,
      Iterable<Aspect> aspects,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      @Nullable ToolchainContext toolchainContext,
      RuleClassProvider ruleClassProvider,
      BuildConfiguration hostConfiguration,
      NestedSetBuilder<Package> transitivePackages,
      NestedSetBuilder<Label> transitiveLoadingRootCauses)
      throws DependencyEvaluationException, ConfiguredTargetFunctionException,
          AspectCreationException, InterruptedException {
    // Create the map from attributes to set of (target, configuration) pairs.
    OrderedSetMultimap<Attribute, Dependency> depValueNames;
    try {
      depValueNames =
          resolver.dependentNodeMap(
              ctgValue,
              hostConfiguration,
              aspects,
              configConditions,
              toolchainContext,
              transitiveLoadingRootCauses);
    } catch (EvalException e) {
      // EvalException can only be thrown by computed Skylark attributes in the current rule.
      env.getListener().handle(Event.error(e.getLocation(), e.getMessage()));
      throw new DependencyEvaluationException(
          new ConfiguredValueCreationException(e.print(), ctgValue.getLabel()));
    } catch (InvalidConfigurationException e) {
      throw new DependencyEvaluationException(e);
    } catch (InconsistentAspectOrderException e) {
      env.getListener().handle(Event.error(e.getLocation(), e.getMessage()));
      throw new DependencyEvaluationException(e);
    }

    // Trim each dep's configuration so it only includes the fragments needed by its transitive
    // closure.
    if (ctgValue.getConfiguration() != null) {
      depValueNames = ConfigurationResolver.resolveConfigurations(env, ctgValue, depValueNames,
          hostConfiguration, ruleClassProvider);
      // It's important that we don't use "if (env.missingValues()) { return null }" here (or
      // in the following lines). See the comments in getDynamicConfigurations' Skyframe call
      // for explanation.
      if (depValueNames == null) {
        return null;
      }
    }

    // Resolve configured target dependencies and handle errors.
    Map<SkyKey, ConfiguredTarget> depValues = resolveConfiguredTargetDependencies(env,
        depValueNames.values(), transitivePackages, transitiveLoadingRootCauses);
    if (depValues == null) {
      return null;
    }

    // Resolve required aspects.
    OrderedSetMultimap<Dependency, ConfiguredAspect> depAspects =
        AspectResolver.resolveAspectDependencies(env, depValues, depValueNames.values(),
            transitivePackages);
    if (depAspects == null) {
      return null;
    }

    // Merge the dependent configured targets and aspects into a single map.
    try {
      return AspectResolver.mergeAspects(depValueNames, depValues, depAspects);
    } catch (DuplicateException e) {
      env.getListener().handle(
          Event.error(ctgValue.getTarget().getLocation(), e.getMessage()));

      throw new ConfiguredTargetFunctionException(
          new ConfiguredValueCreationException(e.getMessage(), ctgValue.getLabel()));
    }
  }

  /**
   * Returns the set of {@link ConfigMatchingProvider}s that key the configurable attributes used by
   * this rule.
   *
   * <p>>If the configured targets supplying those providers aren't yet resolved by the dependency
   * resolver, returns null.
   */
  @Nullable
  static ImmutableMap<Label, ConfigMatchingProvider> getConfigConditions(
      Target target,
      Environment env,
      SkyframeDependencyResolver resolver,
      TargetAndConfiguration ctgValue,
      NestedSetBuilder<Package> transitivePackages,
      NestedSetBuilder<Label> transitiveLoadingRootCauses)
      throws DependencyEvaluationException, InterruptedException {
    if (!(target instanceof Rule)) {
      return NO_CONFIG_CONDITIONS;
    }

    Map<Label, ConfigMatchingProvider> configConditions = new LinkedHashMap<>();

    // Collect the labels of the configured targets we need to resolve.
    OrderedSetMultimap<Attribute, Label> configLabelMap = OrderedSetMultimap.create();
    RawAttributeMapper attributeMap = RawAttributeMapper.of(((Rule) target));
    for (Attribute a : ((Rule) target).getAttributes()) {
      for (Label configLabel : attributeMap.getConfigurabilityKeys(a.getName(), a.getType())) {
        if (!BuildType.Selector.isReservedLabel(configLabel)) {
          configLabelMap.put(a, target.getLabel().resolveRepositoryRelative(configLabel));
        }
      }
    }
    if (configLabelMap.isEmpty()) {
      return NO_CONFIG_CONDITIONS;
    }

    // Collect the corresponding Skyframe configured target values. Abort early if they haven't
    // been computed yet.
    Collection<Dependency> configValueNames;
    try {
      configValueNames = resolver.resolveRuleLabels(
          ctgValue, configLabelMap, transitiveLoadingRootCauses);
    } catch (InconsistentAspectOrderException e) {
      throw new DependencyEvaluationException(e);
    }
    if (env.valuesMissing()) {
      return null;
    }

    // No need to get new configs from Skyframe - config_setting rules always use the current
    // target's config.
    // TODO(bazel-team): remove the need for this special transformation. We can probably do this by
    // simply passing this through trimConfigurations.
    ImmutableList.Builder<Dependency> staticConfigs = ImmutableList.builder();
    for (Dependency dep : configValueNames) {
      staticConfigs.add(Dependency.withConfigurationAndAspects(dep.getLabel(),
          ctgValue.getConfiguration(), dep.getAspects()));
    }
    configValueNames = staticConfigs.build();

    Map<SkyKey, ConfiguredTarget> configValues = resolveConfiguredTargetDependencies(
        env, configValueNames, transitivePackages, transitiveLoadingRootCauses);
    if (configValues == null) {
      return null;
    }

    // Get the configured targets as ConfigMatchingProvider interfaces.
    for (Dependency entry : configValueNames) {
      SkyKey baseKey = ConfiguredTargetValue.key(entry.getLabel(), entry.getConfiguration());
      ConfiguredTarget value = configValues.get(baseKey);
      // The code above guarantees that value is non-null here.
      ConfigMatchingProvider provider = value.getProvider(ConfigMatchingProvider.class);
      if (provider != null) {
        configConditions.put(entry.getLabel(), provider);
      } else {
        // Not a valid provider for configuration conditions.
        String message =
            entry.getLabel() + " is not a valid configuration key for " + target.getLabel();
        env.getListener().handle(Event.error(TargetUtils.getLocationMaybe(target), message));
        throw new DependencyEvaluationException(new ConfiguredValueCreationException(
            message, target.getLabel()));
      }
    }

    return ImmutableMap.copyOf(configConditions);
  }

  /**
   * * Resolves the targets referenced in depValueNames and returns their ConfiguredTarget
   * instances.
   *
   * <p>Returns null if not all instances are available yet.
   */
  @Nullable
  private static Map<SkyKey, ConfiguredTarget> resolveConfiguredTargetDependencies(
      Environment env,
      Collection<Dependency> deps,
      NestedSetBuilder<Package> transitivePackages,
      NestedSetBuilder<Label> transitiveLoadingRootCauses)
      throws DependencyEvaluationException, InterruptedException {
    boolean missedValues = env.valuesMissing();
    boolean failed = false;
    Iterable<SkyKey> depKeys = Iterables.transform(deps,
        input -> ConfiguredTargetValue.key(input.getLabel(), input.getConfiguration()));
    Map<SkyKey, ValueOrException<ConfiguredValueCreationException>> depValuesOrExceptions =
            env.getValuesOrThrow(depKeys, ConfiguredValueCreationException.class);
    Map<SkyKey, ConfiguredTarget> result =
        Maps.newHashMapWithExpectedSize(depValuesOrExceptions.size());
    for (Map.Entry<SkyKey, ValueOrException<ConfiguredValueCreationException>> entry
        : depValuesOrExceptions.entrySet()) {
      try {
        ConfiguredTargetValue depValue = (ConfiguredTargetValue) entry.getValue().get();
        if (depValue == null) {
          missedValues = true;
        } else {
          result.put(entry.getKey(), depValue.getConfiguredTarget());
          transitivePackages.addTransitive(depValue.getTransitivePackages());
        }
      } catch (ConfiguredValueCreationException e) {
        // TODO(ulfjack): If there is an analysis root cause, we drop all loading root causes.
        if (e.getAnalysisRootCause() != null) {
          throw new DependencyEvaluationException(e);
        }
        transitiveLoadingRootCauses.addTransitive(e.loadingRootCauses);
        failed = true;
      }
    }
    if (missedValues) {
      return null;
    } else if (failed) {
      throw new DependencyEvaluationException(
          new ConfiguredValueCreationException(transitiveLoadingRootCauses.build()));
    } else {
      return result;
    }
  }


  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print(((ConfiguredTargetKey) skyKey.argument()).getLabel());
  }

  @Nullable
  private ConfiguredTargetValue createConfiguredTarget(
      SkyframeBuildView view,
      Environment env,
      Target target,
      BuildConfiguration configuration,
      OrderedSetMultimap<Attribute, ConfiguredTarget> depValueMap,
      ImmutableMap<Label, ConfigMatchingProvider> configConditions,
      @Nullable ToolchainContext toolchainContext,
      NestedSetBuilder<Package> transitivePackages)
      throws ConfiguredTargetFunctionException, InterruptedException {
    StoredEventHandler events = new StoredEventHandler();
    BuildConfiguration ownerConfig =
        ConfiguredTargetFactory.getArtifactOwnerConfiguration(env, configuration);
    if (env.valuesMissing()) {
      return null;
    }
    CachingAnalysisEnvironment analysisEnvironment = view.createAnalysisEnvironment(
        new ConfiguredTargetKey(target.getLabel(), ownerConfig), false,
        events, env, configuration);
    if (env.valuesMissing()) {
      return null;
    }

    Preconditions.checkNotNull(depValueMap);
    ConfiguredTarget configuredTarget =
        view.createConfiguredTarget(
            target,
            configuration,
            analysisEnvironment,
            depValueMap,
            configConditions,
            toolchainContext);

    events.replayOn(env.getListener());
    if (events.hasErrors()) {
      analysisEnvironment.disable(target);
      throw new ConfiguredTargetFunctionException(new ConfiguredValueCreationException(
          "Analysis of target '" + target.getLabel() + "' failed; build aborted",
          target.getLabel()));
    }
    Preconditions.checkState(!analysisEnvironment.hasErrors(),
        "Analysis environment hasError() but no errors reported");
    if (env.valuesMissing()) {
      return null;
    }

    analysisEnvironment.disable(target);
    Preconditions.checkNotNull(configuredTarget, target);

    GeneratingActions generatingActions;
    // Check for conflicting actions within this configured target (that indicates a bug in the
    // rule implementation).
    try {
      generatingActions = Actions.filterSharedActionsAndThrowActionConflict(
          analysisEnvironment.getRegisteredActions());
    } catch (ActionConflictException e) {
      throw new ConfiguredTargetFunctionException(e);
    }
    return new ConfiguredTargetValue(
        configuredTarget,
        generatingActions,
        transitivePackages.build(),
        removeActionsAfterEvaluation.get());
  }

  /**
   * An exception indicating that there was a problem during the construction of
   * a ConfiguredTargetValue.
   */
  public static final class ConfiguredValueCreationException extends Exception {
    private final NestedSet<Label> loadingRootCauses;
    // TODO(ulfjack): Collect all analysis root causes, not just the first one.
    @Nullable private final Label analysisRootCause;

    public ConfiguredValueCreationException(String message, Label currentTarget) {
      super(message);
      this.loadingRootCauses = NestedSetBuilder.<Label>emptySet(Order.STABLE_ORDER);
      this.analysisRootCause = Preconditions.checkNotNull(currentTarget);
    }

    public ConfiguredValueCreationException(String message, NestedSet<Label> rootCauses) {
      super(message);
      this.loadingRootCauses = rootCauses;
      this.analysisRootCause = null;
    }

    public ConfiguredValueCreationException(NestedSet<Label> rootCauses) {
      this("Loading failed", rootCauses);
    }

    public ConfiguredValueCreationException(String message) {
      this(message, NestedSetBuilder.<Label>emptySet(Order.STABLE_ORDER));
    }

    public NestedSet<Label> getRootCauses() {
      return loadingRootCauses;
    }

    public Label getAnalysisRootCause() {
      return analysisRootCause;
    }
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link ConfiguredTargetFunction#compute}.
   */
  public static final class ConfiguredTargetFunctionException extends SkyFunctionException {
    public ConfiguredTargetFunctionException(NoSuchThingException e) {
      super(e, Transience.PERSISTENT);
    }

    private ConfiguredTargetFunctionException(ConfiguredValueCreationException e) {
      super(e, Transience.PERSISTENT);
    }

    private ConfiguredTargetFunctionException(ActionConflictException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
