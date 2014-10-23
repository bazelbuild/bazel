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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.StoredEventHandler;
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
import com.google.devtools.build.lib.skyframe.SkyframeExecutor.BuildViewProvider;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.view.CachingAnalysisEnvironment;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.TargetAndConfiguration;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.lib.view.config.ConfigMatchingProvider;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * SkyFunction for {@link ConfiguredTargetValue}s.
 */
final class ConfiguredTargetFunction implements SkyFunction {
  private static final Function<TargetAndConfiguration, SkyKey> TO_KEYS =
      new Function<TargetAndConfiguration, SkyKey>() {
    @Override
    public SkyKey apply(TargetAndConfiguration input) {
      Label depLabel = input.getLabel();
      return ConfiguredTargetValue.key(depLabel, input.getConfiguration());
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

    LabelAndConfiguration lc = (LabelAndConfiguration) key.argument();

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
      throw new ConfiguredTargetFunctionException(key,
          new NoSuchTargetException(lc.getLabel(), "No such target"));
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

    // 1. Get the configuration targets that trigger this rule's configurable attributes.
    Set<ConfigMatchingProvider> configConditions =
        getConfigConditions(target, env, resolver, ctgValue, key);
    if (configConditions == null) {
      // Those targets haven't yet been resolved.
      return null;
    }

    // 2. Create the map from attributes to list of (target, configuration) pairs.
    ListMultimap<Attribute, TargetAndConfiguration> depValueNames;
    try {
      depValueNames = resolver.dependentNodeMap(ctgValue, configConditions);
    } catch (EvalException e) {
      env.getListener().handle(Event.error(e.getLocation(), e.getMessage()));
      throw new ConfiguredTargetFunctionException(key,
          new ConfiguredValueCreationException(e.print()));
    }

    // 3. Resolve dependencies and handle errors.
    Map<SkyKey, ConfiguredTargetValue> depValues =
        resolveDependencies(env, depValueNames, key, target);
    if (depValues == null) {
      return null;
    }

    // 4. Convert each (target, configuration) pair to a ConfiguredTarget instance.
    ListMultimap<Attribute, ConfiguredTarget> depValueMap = ArrayListMultimap.create();
    for (Map.Entry<Attribute, TargetAndConfiguration> entry : depValueNames.entries()) {
      ConfiguredTargetValue value = depValues.get(TO_KEYS.apply(entry.getValue()));
      // The code above guarantees that value is non-null here.
      depValueMap.put(entry.getKey(), value.getConfiguredTarget());
    }

    // 5. Create the ConfiguredTarget for the present value.
    return createConfiguredTarget(view, env, target, configuration, depValueMap, configConditions);
  }

  /**
   * Returns the set of {@link ConfigMatchingProvider}s that key the configurable attributes
   * used by this rule.
   *
   * <p>>If the configured targets supplying those providers aren't yet resolved by the
   * dependency resolver, returns null.
   */
  private Set<ConfigMatchingProvider> getConfigConditions(Target target, Environment env,
      SkyframeDependencyResolver resolver, TargetAndConfiguration ctgValue, SkyKey skyKey)
      throws ConfiguredTargetFunctionException {
    if (!(target instanceof Rule)) {
      return ImmutableSet.of();
    }

    ImmutableSet.Builder<ConfigMatchingProvider> configConditions = ImmutableSet.builder();

    // Collect the labels of the configured targets we need to resolve.
    ListMultimap<Attribute, Label> configLabelMap = ArrayListMultimap.create();
    RawAttributeMapper attributeMap = RawAttributeMapper.of(((Rule) target));
    for (Attribute a : ((Rule) target).getAttributes()) {
      for (Label configLabel : attributeMap.getConfigurabilityKeys(a.getName(), a.getType())) {
        if (!Type.Selector.isReservedLabel(configLabel)) {
          configLabelMap.put(a, configLabel);
        }
      }
    }
    if (configLabelMap.isEmpty()) {
      return ImmutableSet.of();
    }

    // Collect the corresponding Skyframe configured target values. Abort early if they haven't
    // been computed yet.
    ListMultimap<Attribute, TargetAndConfiguration> configValueNames =
        resolver.resolveRuleLabels(ctgValue, configLabelMap);
    Map<SkyKey, ConfiguredTargetValue> configValues =
        resolveDependencies(env, configValueNames, skyKey, target);
    if (configValues == null) {
      return null;
    }

    // Get the configured targets as ConfigMatchingProvider interfaces.
    for (Map.Entry<Attribute, TargetAndConfiguration> entry : configValueNames.entries()) {
      ConfiguredTargetValue value = configValues.get(TO_KEYS.apply(entry.getValue()));
      // The code above guarantees that value is non-null here.
      ConfigMatchingProvider provider =
          value.getConfiguredTarget().getProvider(ConfigMatchingProvider.class);
      if (provider != null) {
        configConditions.add(provider);
      } else {
        // Not a valid provider for configuration conditions.
        Target badTarget = entry.getValue().getTarget();
        String message = badTarget + " is not a valid configuration key for "
            + target.getLabel().toString();
        env.getListener().handle(Event.error(TargetUtils.getLocationMaybe(badTarget), message));
        throw new ConfiguredTargetFunctionException(skyKey,
            new ConfiguredValueCreationException(message));
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
  private Map<SkyKey, ConfiguredTargetValue> resolveDependencies(Environment env,
      ListMultimap<Attribute, TargetAndConfiguration> depValueNames, SkyKey skyKey,
      Target target) throws ConfiguredTargetFunctionException {
    boolean ok = !env.valuesMissing();
    String message = null;
    Iterable<SkyKey> depKeys = Iterables.transform(depValueNames.values(), TO_KEYS);
    // TODO(bazel-team): maybe having a two-exception argument is better than typing a generic
    // Exception here.
    Map<SkyKey, ValueOrException<Exception>> depValuesOrExceptions =
        env.getValuesOrThrow(depKeys, Exception.class);
    Map<SkyKey, ConfiguredTargetValue> depValues = new HashMap<>(depValuesOrExceptions.size());
    for (Map.Entry<SkyKey, ValueOrException<Exception>> entry : depValuesOrExceptions.entrySet()) {
      LabelAndConfiguration depLabelAndConfiguration =
          (LabelAndConfiguration) entry.getKey().argument();
      Label depLabel = depLabelAndConfiguration.getLabel();
      ConfiguredTargetValue depValue = null;
      NoSuchThingException directChildException = null;
      try {
        depValue = (ConfiguredTargetValue) entry.getValue().get();
      } catch (NoSuchTargetException e) {
        if (depLabel.equals(e.getLabel())) {
          directChildException = e;
        }
      } catch (NoSuchPackageException e) {
        if (depLabel.getPackageName().equals(e.getPackageName())) {
          directChildException = e;
        }
      } catch (ConfiguredValueCreationException e) {
        // Do nothing.
      } catch (Exception e) {
        throw new IllegalStateException("Not NoSuchTargetException or NoSuchPackageException"
            + " or ViewCreationFailedException: " + e.getMessage(), e);
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
        depValues.put(entry.getKey(), depValue);
      }
    }
    if (message != null) {
      throw new ConfiguredTargetFunctionException(skyKey, new NoSuchTargetException(message));
    }
    if (!ok) {
      return null;
    } else {
      return depValues;
    }
  }


  @Override
  public String extractTag(SkyKey skyKey) {
    return Label.print(((LabelAndConfiguration) skyKey.argument()).getLabel());
  }

  @Nullable
  private ConfiguredTargetValue createConfiguredTarget(SkyframeBuildView view,
      Environment env, Target target, BuildConfiguration configuration,
      ListMultimap<Attribute, ConfiguredTarget> depValueMap,
      Set<ConfigMatchingProvider> configConditions)
      throws ConfiguredTargetFunctionException,
      InterruptedException {
    boolean extendedSanityChecks = configuration != null && configuration.extendedSanityChecks();

    StoredEventHandler events = new StoredEventHandler();
    BuildConfiguration ownerConfig = (configuration == null)
        ? null : configuration.getArtifactOwnerConfiguration();
    boolean allowRegisteringActions = configuration == null || configuration.isActionsEnabled();
    CachingAnalysisEnvironment analysisEnvironment = view.createAnalysisEnvironment(
        new LabelAndConfiguration(target.getLabel(), ownerConfig), false,
        extendedSanityChecks, events, env, allowRegisteringActions);
    if (env.valuesMissing()) {
      return null;
    }

    ConfiguredTarget configuredTarget = view.createAndInitialize(
        target, configuration, analysisEnvironment, depValueMap, configConditions);

    events.replayOn(env.getListener());
    if (events.hasErrors()) {
      analysisEnvironment.disable(target);
      throw new ConfiguredTargetFunctionException(ConfiguredTargetValue.key(target.getLabel(),
          configuration), new ConfiguredValueCreationException(
              "Analysis of target '" + target.getLabel() + "' failed; build aborted"));
    }
    Preconditions.checkState(!analysisEnvironment.hasErrors(),
        "Analysis environment hasError() but no errors reported");
    if (env.valuesMissing()) {
      return null;
    }

    analysisEnvironment.disable(target);
    Preconditions.checkNotNull(configuredTarget, target);

    return new ConfiguredTargetValue(configuredTarget,
        ImmutableList.copyOf(analysisEnvironment.getRegisteredActions()));
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
  private static final class ConfiguredTargetFunctionException extends SkyFunctionException {
    public ConfiguredTargetFunctionException(SkyKey key, NoSuchTargetException e) {
      super(key, e);
    }

    public ConfiguredTargetFunctionException(SkyKey key, ConfiguredValueCreationException e) {
      super(key, e);
    }
  }
}
