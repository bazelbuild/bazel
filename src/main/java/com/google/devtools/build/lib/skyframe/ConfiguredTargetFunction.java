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
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.SkyframeDependencyResolver.ConfiguredValueCreationException;
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
  public SkyValue compute(SkyKey key, Environment env) throws SkyFunctionException,
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
        resolver.getConfigConditions(target, env, resolver, ctgValue, key);
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
        resolver.resolveDependencies(env, depValueNames, key, target);
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
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link ConfiguredTargetFunction#compute}.
   */
  private static final class ConfiguredTargetFunctionException extends SkyFunctionException {
    public ConfiguredTargetFunctionException(SkyKey key, NoSuchTargetException e) {
      super(key, e, Transience.PERSISTENT);
    }

    public ConfiguredTargetFunctionException(SkyKey key, ConfiguredValueCreationException e) {
      super(key, e, Transience.PERSISTENT);
    }
  }
}
