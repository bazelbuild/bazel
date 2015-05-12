// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.Aspect;
import com.google.devtools.build.lib.analysis.CachingAnalysisEnvironment;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.DependencyResolver.Dependency;
import com.google.devtools.build.lib.analysis.LabelAndConfiguration;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectFactory;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.skyframe.AspectFunction.AspectCreationException;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor.BuildViewProvider;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException2;
import com.google.devtools.build.skyframe.ValueOrException3;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * SkyFunction for {@link ConfiguredTargetValue}s.
 */
final class ConfiguredTargetFunction implements SkyFunction {

  /**
   * Exception class that signals an error during the evaluation of a dependency.
   */
  public static class DependencyEvaluationException extends Exception {
    private final SkyKey rootCauseSkyKey;

    public DependencyEvaluationException(Exception cause) {
      super(cause);
      this.rootCauseSkyKey = null;
    }

    public DependencyEvaluationException(SkyKey rootCauseSkyKey, Exception cause) {
      super(cause);
      this.rootCauseSkyKey = rootCauseSkyKey;
    }

    /**
     * Returns the key of the root cause or null if the problem was with this target.
     */
    public SkyKey getRootCauseSkyKey() {
      return rootCauseSkyKey;
    }

    @Override
    public Exception getCause() {
      return (Exception) super.getCause();
    }
  }

  private static final Function<Dependency, SkyKey> TO_KEYS =
      new Function<Dependency, SkyKey>() {
    @Override
    public SkyKey apply(Dependency input) {
      return ConfiguredTargetValue.key(input.getLabel(), input.getConfiguration());
    }
  };

  private final BuildViewProvider buildViewProvider;

  ConfiguredTargetFunction(BuildViewProvider buildViewProvider) {
    this.buildViewProvider = buildViewProvider;
  }

  @Override
  public SkyValue compute(SkyKey key, Environment env) throws ConfiguredTargetFunctionException,
      InterruptedException {
    SkyframeBuildView view = buildViewProvider.getSkyframeBuildView();

    ConfiguredTargetKey configuredTargetKey = (ConfiguredTargetKey) key.argument();
    LabelAndConfiguration lc = LabelAndConfiguration.of(
        configuredTargetKey.getLabel(), configuredTargetKey.getConfiguration());

    BuildConfiguration configuration = lc.getConfiguration();

    PackageValue packageValue =
        (PackageValue) env.getValue(PackageValue.key(lc.getLabel().getPackageIdentifier()));
    if (packageValue == null) {
      return null;
    }

    Target target;
    try {
      target = packageValue.getPackage().getTarget(lc.getLabel().getName());
    } catch (NoSuchTargetException e1) {
      throw new ConfiguredTargetFunctionException(new NoSuchTargetException(lc.getLabel(),
          "No such target"));
    }
    // TODO(bazel-team): This is problematic - we create the right key, but then end up with a value
    // that doesn't match; we can even have the same value multiple times. However, I think it's
    // only triggered in tests (i.e., in normal operation, the configuration passed in is already
    // null).
    if (target instanceof InputFile) {
      // InputFileConfiguredTarget expects its configuration to be null since it's not used.
      configuration = null;
    } else if (target instanceof PackageGroup) {
      // Same for PackageGroupConfiguredTarget.
      configuration = null;
    }
    TargetAndConfiguration ctgValue =
        new TargetAndConfiguration(target, configuration);

    SkyframeDependencyResolver resolver = view.createDependencyResolver(env);
    if (resolver == null) {
      return null;
    }

    try {
      // Get the configuration targets that trigger this rule's configurable attributes.
      Set<ConfigMatchingProvider> configConditions =
          getConfigConditions(ctgValue.getTarget(), env, resolver, ctgValue);
      if (configConditions == null) {
        // Those targets haven't yet been resolved.
        return null;
      }

      ListMultimap<Attribute, ConfiguredTarget> depValueMap =
          computeDependencies(env, resolver, ctgValue, null, configConditions);
      return createConfiguredTarget(
          view, env, target, configuration, depValueMap, configConditions);
    } catch (DependencyEvaluationException e) {
      throw new ConfiguredTargetFunctionException(e.getRootCauseSkyKey(), e.getCause());
    }
  }

  /**
   * Computes the direct dependencies of a node in the configured target graph (a configured
   * target or an aspect).
   *
   * <p>Returns null if Skyframe hasn't evaluated the required dependencies yet. In this case, the
   * caller should also return null to Skyframe.
   *
   * @param env the Skyframe environment
   * @param resolver The dependency resolver
   * @param ctgValue The label and the configuration of the node
   * @param aspectDefinition the aspect of the node (if null, the node is a configured target,
   *     otherwise it's an asect)
   * @param configConditions the configuration conditions for evaluating the attributes of the node
   * @return an attribute -&gt; direct dependency multimap
   */
  @Nullable
  static ListMultimap<Attribute, ConfiguredTarget> computeDependencies(
      Environment env, SkyframeDependencyResolver resolver, TargetAndConfiguration ctgValue,
      AspectDefinition aspectDefinition, Set<ConfigMatchingProvider> configConditions)
      throws DependencyEvaluationException {

    // 1. Create the map from attributes to list of (target, configuration) pairs.
    ListMultimap<Attribute, Dependency> depValueNames;
    try {
      depValueNames = resolver.dependentNodeMap(ctgValue, aspectDefinition, configConditions);
    } catch (EvalException e) {
      env.getListener().handle(Event.error(e.getLocation(), e.getMessage()));
      throw new DependencyEvaluationException(new ConfiguredValueCreationException(e.print()));
    }

    // 2. Resolve configured target dependencies and handle errors.
    Map<SkyKey, ConfiguredTarget> depValues =
        resolveConfiguredTargetDependencies(env, depValueNames.values(), ctgValue.getTarget());
    if (depValues == null) {
      return null;
    }

    // 3. Resolve required aspects.
    ListMultimap<SkyKey, Aspect> depAspects = resolveAspectDependencies(
        env, depValues, depValueNames.values());
    if (depAspects == null) {
      return null;
    }

    // 3. Merge the dependent configured targets and aspects into a single map.
    return mergeAspects(depValueNames, depValues, depAspects);
  }

  /**
   * Merges the each direct dependency configured target with the aspects associated with it.
   *
   * <p>Note that the combination of a configured target and its associated aspects are not
   * represented by a Skyframe node. This is because there can possibly be many different
   * combinations of aspects for a particular configured target, so it would result in a
   * combinatiorial explosion of Skyframe nodes.
   */
  private static ListMultimap<Attribute, ConfiguredTarget> mergeAspects(
      ListMultimap<Attribute, Dependency> depValueNames,
      Map<SkyKey, ConfiguredTarget> depConfiguredTargetMap,
      ListMultimap<SkyKey, Aspect> depAspectMap) {
    ListMultimap<Attribute, ConfiguredTarget> result = ArrayListMultimap.create();

    for (Map.Entry<Attribute, Dependency> entry : depValueNames.entries()) {
      Dependency dep = entry.getValue();
      SkyKey depKey = TO_KEYS.apply(dep);
      ConfiguredTarget depConfiguredTarget = depConfiguredTargetMap.get(depKey);
      result.put(entry.getKey(),
          RuleConfiguredTarget.mergeAspects(depConfiguredTarget, depAspectMap.get(depKey)));
    }

    return result;
  }

  /**
   * Given a list of {@link Dependency} objects, returns a multimap from the {@link SkyKey} of the
   * dependency to the {@link Aspect} instances that should be merged into it.
   *
   * <p>Returns null if the required aspects are not computed yet.
   */
  @Nullable
  private static ListMultimap<SkyKey, Aspect> resolveAspectDependencies(Environment env,
      Map<SkyKey, ConfiguredTarget> configuredTargetMap, Iterable<Dependency> deps)
      throws DependencyEvaluationException {
    ListMultimap<SkyKey, Aspect> result = ArrayListMultimap.create();
    Set<SkyKey> aspectKeys = new HashSet<>();
    for (Dependency dep : deps) {
      for (Class<? extends ConfiguredAspectFactory> depAspect : dep.getAspects()) {
        aspectKeys.add(AspectValue.key(dep.getLabel(), dep.getConfiguration(), depAspect));
      }
    }

    Map<SkyKey, ValueOrException3<
        AspectCreationException, NoSuchThingException, ConfiguredValueCreationException>>
        depAspects = env.getValuesOrThrow(aspectKeys, AspectCreationException.class,
            NoSuchThingException.class, ConfiguredValueCreationException.class);

    for (Dependency dep : deps) {
      SkyKey depKey = TO_KEYS.apply(dep);
      // If the same target was declared in different attributes of rule, we should not process it
      // twice.
      if (result.containsKey(depKey)) {
        continue;
      }
      ConfiguredTarget depConfiguredTarget = configuredTargetMap.get(depKey);
      for (Class<? extends ConfiguredAspectFactory> depAspect : dep.getAspects()) {
        if (!aspectMatchesConfiguredTarget(depConfiguredTarget, depAspect)) {
          continue;
        }

        SkyKey aspectKey = AspectValue.key(dep.getLabel(), dep.getConfiguration(), depAspect);
        AspectValue aspectValue = null;
        try {
          aspectValue = (AspectValue) depAspects.get(aspectKey).get();
        } catch (ConfiguredValueCreationException e) {
          // The configured target should have been created in resolveConfiguredTargetDependencies()
          throw new IllegalStateException(e);
        } catch (NoSuchThingException | AspectCreationException e) {
          AspectFactory<?, ?, ?> depAspectFactory = AspectFactory.Util.create(depAspect);
          throw new DependencyEvaluationException(new ConfiguredValueCreationException(
              String.format("Evaluation of aspect %s on %s failed: %s",
                  depAspectFactory.getDefinition().getName(), dep.getLabel(), e.toString())));
        }

        if (aspectValue == null) {
          // Dependent aspect has either not been computed yet or is in error.
          return null;
        }
        result.put(depKey, aspectValue.get());
      }
    }

    return result;
  }

  private static boolean aspectMatchesConfiguredTarget(ConfiguredTarget dep,
      Class<? extends ConfiguredAspectFactory> aspectFactory) {
    AspectDefinition aspectDefinition = AspectFactory.Util.create(aspectFactory).getDefinition();
    for (Class<?> provider : aspectDefinition.getRequiredProviders()) {
      if (dep.getProvider((Class<? extends TransitiveInfoProvider>) provider) == null) {
        return false;
      }
    }

    return true;
  }

  /**
   * Returns the set of {@link ConfigMatchingProvider}s that key the configurable attributes
   * used by this rule.
   *
   * <p>>If the configured targets supplying those providers aren't yet resolved by the
   * dependency resolver, returns null.
   */
  @Nullable
  static Set<ConfigMatchingProvider> getConfigConditions(Target target, Environment env,
      SkyframeDependencyResolver resolver, TargetAndConfiguration ctgValue)
      throws DependencyEvaluationException {
    if (!(target instanceof Rule)) {
      return ImmutableSet.of();
    }

    ImmutableSet.Builder<ConfigMatchingProvider> configConditions = ImmutableSet.builder();

    // Collect the labels of the configured targets we need to resolve.
    ListMultimap<Attribute, LabelAndConfiguration> configLabelMap = ArrayListMultimap.create();
    RawAttributeMapper attributeMap = RawAttributeMapper.of(((Rule) target));
    for (Attribute a : ((Rule) target).getAttributes()) {
      for (Label configLabel : attributeMap.getConfigurabilityKeys(a.getName(), a.getType())) {
        if (!Type.Selector.isReservedLabel(configLabel)) {
          configLabelMap.put(a, LabelAndConfiguration.of(
              configLabel, ctgValue.getConfiguration()));
        }
      }
    }
    if (configLabelMap.isEmpty()) {
      return ImmutableSet.of();
    }

    // Collect the corresponding Skyframe configured target values. Abort early if they haven't
    // been computed yet.
    Collection<Dependency> configValueNames =
        resolver.resolveRuleLabels(ctgValue, null, configLabelMap);
    Map<SkyKey, ConfiguredTarget> configValues =
        resolveConfiguredTargetDependencies(env, configValueNames, target);
    if (configValues == null) {
      return null;
    }

    // Get the configured targets as ConfigMatchingProvider interfaces.
    for (Dependency entry : configValueNames) {
      ConfiguredTarget value = configValues.get(TO_KEYS.apply(entry));
      // The code above guarantees that value is non-null here.
      ConfigMatchingProvider provider = value.getProvider(ConfigMatchingProvider.class);
      if (provider != null) {
        configConditions.add(provider);
      } else {
        // Not a valid provider for configuration conditions.
        String message =
            entry.getLabel() + " is not a valid configuration key for " + target.getLabel();
        env.getListener().handle(Event.error(TargetUtils.getLocationMaybe(target), message));
        throw new DependencyEvaluationException(new ConfiguredValueCreationException(message));
      }
    }

    return configConditions.build();
  }

  /***
   * Resolves the targets referenced in depValueNames and returns their ConfiguredTarget
   * instances.
   *
   * <p>Returns null if not all instances are available yet.
   *
   */
  @Nullable
  private static Map<SkyKey, ConfiguredTarget> resolveConfiguredTargetDependencies(
      Environment env, Collection<Dependency> deps, Target target)
      throws DependencyEvaluationException {
    boolean ok = !env.valuesMissing();
    String message = null;
    Iterable<SkyKey> depKeys = Iterables.transform(deps, TO_KEYS);
    Map<SkyKey, ValueOrException2<NoSuchTargetException,
        NoSuchPackageException>> depValuesOrExceptions = env.getValuesOrThrow(depKeys,
            NoSuchTargetException.class, NoSuchPackageException.class);
    Map<SkyKey, ConfiguredTarget> depValues = new HashMap<>(depValuesOrExceptions.size());
    SkyKey childKey = null;
    NoSuchThingException transitiveChildException = null;
    for (Map.Entry<SkyKey, ValueOrException2<NoSuchTargetException, NoSuchPackageException>> entry
        : depValuesOrExceptions.entrySet()) {
      ConfiguredTargetKey depKey = (ConfiguredTargetKey) entry.getKey().argument();
      LabelAndConfiguration depLabelAndConfiguration = LabelAndConfiguration.of(
          depKey.getLabel(), depKey.getConfiguration());
      Label depLabel = depLabelAndConfiguration.getLabel();
      ConfiguredTargetValue depValue = null;
      NoSuchThingException directChildException = null;
      try {
        depValue = (ConfiguredTargetValue) entry.getValue().get();
      } catch (NoSuchTargetException e) {
        if (depLabel.equals(e.getLabel())) {
          directChildException = e;
        } else {
          childKey = entry.getKey();
          transitiveChildException = e;
        }
      } catch (NoSuchPackageException e) {
        if (depLabel.getPackageName().equals(e.getPackageName())) {
          directChildException = e;
        } else {
          childKey = entry.getKey();
          transitiveChildException = e;
        }
      }
      // If an exception wasn't caused by a direct child target value, we'll treat it the same
      // as any other missing dep by setting ok = false below, and returning null at the end.
      if (directChildException != null) {
        // Only update messages for missing targets we depend on directly.
        message = TargetUtils.formatMissingEdge(target, depLabel, directChildException);
        env.getListener().handle(Event.error(TargetUtils.getLocationMaybe(target), message));
      }

      if (depValue == null) {
        ok = false;
      } else {
        depValues.put(entry.getKey(), depValue.getConfiguredTarget());
      }
    }
    if (message != null) {
      throw new DependencyEvaluationException(new NoSuchTargetException(message));
    }
    if (childKey != null) {
      throw new DependencyEvaluationException(childKey, transitiveChildException);
    }
    if (!ok) {
      return null;
    } else {
      return depValues;
    }
  }


  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print(((ConfiguredTargetKey) skyKey.argument()).getLabel());
  }

  @Nullable
  private ConfiguredTargetValue createConfiguredTarget(SkyframeBuildView view,
      Environment env, Target target, BuildConfiguration configuration,
      ListMultimap<Attribute, ConfiguredTarget> depValueMap,
      Set<ConfigMatchingProvider> configConditions)
      throws ConfiguredTargetFunctionException,
      InterruptedException {
    StoredEventHandler events = new StoredEventHandler();
    BuildConfiguration ownerConfig = (configuration == null)
        ? null : configuration.getArtifactOwnerConfiguration();
    CachingAnalysisEnvironment analysisEnvironment = view.createAnalysisEnvironment(
        new ConfiguredTargetKey(target.getLabel(), ownerConfig), false,
        events, env, configuration);
    if (env.valuesMissing()) {
      return null;
    }

    ConfiguredTarget configuredTarget = view.createConfiguredTarget(target, configuration,
        analysisEnvironment, depValueMap, configConditions);

    events.replayOn(env.getListener());
    if (events.hasErrors()) {
      analysisEnvironment.disable(target);
      throw new ConfiguredTargetFunctionException(new ConfiguredValueCreationException(
              "Analysis of target '" + target.getLabel() + "' failed; build aborted"));
    }
    Preconditions.checkState(!analysisEnvironment.hasErrors(),
        "Analysis environment hasError() but no errors reported");
    if (env.valuesMissing()) {
      return null;
    }

    analysisEnvironment.disable(target);
    Preconditions.checkNotNull(configuredTarget, target);

    try {
      return new ConfiguredTargetValue(configuredTarget,
          filterSharedActionsAndThrowIfConflict(analysisEnvironment.getRegisteredActions()));
    } catch (ActionConflictException e) {
      throw new ConfiguredTargetFunctionException(e);
    }
  }

  static Map<Artifact, Action> filterSharedActionsAndThrowIfConflict(Iterable<Action> actions)
      throws ActionConflictException {
    Map<Artifact, Action> generatingActions = new HashMap<>();
    for (Action action : actions) {
      for (Artifact artifact : action.getOutputs()) {
        Action previousAction = generatingActions.put(artifact, action);
        if (previousAction != null && previousAction != action
            && !Actions.canBeShared(previousAction, action)) {
          throw new ActionConflictException(artifact, previousAction, action);
        }
      }
    }
    return generatingActions;
  }

  /**
   * An exception indicating that there was a problem during the construction of
   * a ConfiguredTargetValue.
   */
  public static final class ConfiguredValueCreationException extends Exception {

    public ConfiguredValueCreationException(String message) {
      super(message);
    }
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link ConfiguredTargetFunction#compute}.
   */
  public static final class ConfiguredTargetFunctionException extends SkyFunctionException {
    public ConfiguredTargetFunctionException(NoSuchTargetException e) {
      super(e, Transience.PERSISTENT);
    }

    private ConfiguredTargetFunctionException(ConfiguredValueCreationException error) {
      super(error, Transience.PERSISTENT);
    };

    private ConfiguredTargetFunctionException(ActionConflictException e) {
      super(e, Transience.PERSISTENT);
    }

    private ConfiguredTargetFunctionException(
        @Nullable SkyKey childKey, Exception transitiveError) {
      super(transitiveError, childKey);
    }
  }
}
