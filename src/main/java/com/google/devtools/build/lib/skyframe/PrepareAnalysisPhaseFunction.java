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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.DependencyKey;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.ConfigurationResolver;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.FragmentClassSet;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NullTransition;
import com.google.devtools.build.lib.analysis.starlark.StarlarkTransition;
import com.google.devtools.build.lib.analysis.starlark.StarlarkTransition.TransitionException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.ResolvedTargets;
import com.google.devtools.build.lib.events.ErrorSensingEventHandler;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.server.FailureDetails.BuildConfiguration.Code;
import com.google.devtools.build.lib.skyframe.PrepareAnalysisPhaseValue.PrepareAnalysisPhaseKey;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Prepares for analysis - creates the top-level configurations and evaluates the transitions needed
 * for the top-level targets (including trimming).
 */
final class PrepareAnalysisPhaseFunction implements SkyFunction {
  private final ConfiguredRuleClassProvider ruleClassProvider;

  PrepareAnalysisPhaseFunction(ConfiguredRuleClassProvider ruleClassProvider) {
    this.ruleClassProvider = ruleClassProvider;
  }

  @Override
  public PrepareAnalysisPhaseValue compute(SkyKey key, Environment env)
      throws InterruptedException, PrepareAnalysisPhaseFunctionException {
    PrepareAnalysisPhaseKey options = (PrepareAnalysisPhaseKey) key.argument();

    BuildOptions targetOptions = options.getOptions();
    BuildOptionsView hostTransitionOptionsView =
        new BuildOptionsView(targetOptions, HostTransition.INSTANCE.requiresOptionFragments());
    BuildOptions hostOptions =
        targetOptions.get(CoreOptions.class).useDistinctHostConfiguration
            ? HostTransition.INSTANCE.patch(hostTransitionOptionsView, env.getListener())
            : targetOptions;

    FragmentClassSet allFragments = options.getFragments();

    PathFragment platformMappingPath = targetOptions.get(PlatformOptions.class).platformMappings;
    PlatformMappingValue platformMappingValue =
        (PlatformMappingValue) env.getValue(PlatformMappingValue.Key.create(platformMappingPath));
    if (platformMappingValue == null) {
      return null;
    }

    List<BuildOptions> topLevelBuildOptions =
        getTopLevelBuildOptions(targetOptions, options.getMultiCpu());

    ImmutableList.Builder<BuildConfigurationValue.Key> targetConfigurationKeysBuilder =
        ImmutableList.builderWithExpectedSize(topLevelBuildOptions.size());
    BuildConfigurationValue.Key hostConfigurationKey;
    try {
      hostConfigurationKey =
          BuildConfigurationValue.keyWithPlatformMapping(
              platformMappingValue, allFragments, hostOptions);
      for (BuildOptions buildOptions :
          getTopLevelBuildOptions(targetOptions, options.getMultiCpu())) {
        targetConfigurationKeysBuilder.add(
            BuildConfigurationValue.keyWithPlatformMapping(
                platformMappingValue, allFragments, buildOptions));
      }
    } catch (OptionsParsingException e) {
      throw new PrepareAnalysisPhaseFunctionException(new InvalidConfigurationException(e));
    }

    // We don't need the host configuration below, but we call this to get the error, if any.
    try {
      env.getValueOrThrow(hostConfigurationKey, InvalidConfigurationException.class);
    } catch (InvalidConfigurationException e) {
      throw new PrepareAnalysisPhaseFunctionException(e);
    }

    ImmutableList<BuildConfigurationValue.Key> targetConfigurationKeys =
        targetConfigurationKeysBuilder.build();
    Map<SkyKey, SkyValue> configs = env.getValues(targetConfigurationKeys);

    // We only report invalid options for the target configurations, and abort if there's an error.
    ErrorSensingEventHandler<Void> nosyEventHandler =
        ErrorSensingEventHandler.withoutPropertyValueTracking(env.getListener());
    targetConfigurationKeys.stream()
        .map(configs::get)
        .filter(Objects::nonNull)
        .map(v -> ((BuildConfigurationValue) v).getConfiguration())
        .forEach(config -> config.reportInvalidOptions(nosyEventHandler));
    if (nosyEventHandler.hasErrors()) {
      throw new PrepareAnalysisPhaseFunctionException(
          new InvalidConfigurationException(
              "Build options are invalid", Code.INVALID_BUILD_OPTIONS));
    }

    // We get the list of labels from the TargetPatternPhaseValue, so we are reasonably certain that
    // there will not be an error loading these again.
    ResolvedTargets<Target> resolvedTargets =
        TestsForTargetPatternFunction.labelsToTargets(env, options.getLabels(), false);
    if (resolvedTargets == null) {
      return null;
    }
    ImmutableSet<Target> targets = resolvedTargets.getTargets();

    // We use a hash set here to remove duplicate nodes; this can happen for input files and package
    // groups.
    LinkedHashSet<TargetAndConfiguration> nodes = new LinkedHashSet<>(targets.size());
    for (Target target : targets) {
      for (BuildConfigurationValue.Key configKey : targetConfigurationKeys) {
        BuildConfiguration config =
            ((BuildConfigurationValue) configs.get(configKey)).getConfiguration();
        nodes.add(new TargetAndConfiguration(target, config));
      }
    }

    // We'll get the configs from #resolveConfigurations below, which started out as a copy of the
    // same code in SkyframeExecutor, which gets configurations for deps including transitions. So,
    // for now, to satisfy its API we resolve transitions and repackage each target as a Dependency
    // (with a NONE transition if necessary).
    // Keep this in sync with AnalysisUtils#getTargetsWithConfigs.
    Multimap<BuildConfiguration, DependencyKey> asDeps =
        AnalysisUtils.targetsToDeps(nodes, ruleClassProvider);
    LinkedHashSet<TargetAndConfiguration> topLevelTargetsWithConfigs;
    try {
      topLevelTargetsWithConfigs = resolveConfigurations(env, nodes, asDeps);
    } catch (TransitionException | OptionsParsingException e) {
      throw new PrepareAnalysisPhaseFunctionException(new InvalidConfigurationException(e));
    }
    if (env.valuesMissing()) {
      return null;
    }
    ImmutableList<ConfiguredTargetKey> topLevelCtKeys =
        topLevelTargetsWithConfigs.stream()
            .map(TargetAndConfiguration::getConfiguredTargetKey)
            .collect(ImmutableList.toImmutableList());
    return new PrepareAnalysisPhaseValue(
        hostConfigurationKey, targetConfigurationKeys, topLevelCtKeys);
  }

  /**
   * Returns the {@link BuildOptions} to apply to the top-level build configurations. This can be
   * plural because of {@code multiCpu}.
   */
  // Visible for SkyframeExecutor, which uses it for tests.
  static List<BuildOptions> getTopLevelBuildOptions(
      BuildOptions buildOptions, Set<String> multiCpu) {
    if (multiCpu.isEmpty()) {
      return ImmutableList.of(buildOptions);
    }
    ImmutableList.Builder<BuildOptions> multiCpuOptions = ImmutableList.builder();
    for (String cpu : multiCpu) {
      BuildOptions clonedOptions = buildOptions.clone();
      clonedOptions.get(CoreOptions.class).cpu = cpu;
      multiCpuOptions.add(clonedOptions);
    }
    return multiCpuOptions.build();
  }

  // TODO(bazel-team): error out early for targets that fail - untrimmed configurations should
  // never make it through analysis (and especially not seed ConfiguredTargetValues)
  // Keep this in sync with {@link ConfigurationResolver#getConfigurationsFromExecutor}.
  private LinkedHashSet<TargetAndConfiguration> resolveConfigurations(
      SkyFunction.Environment env,
      Iterable<TargetAndConfiguration> nodes,
      Multimap<BuildConfiguration, DependencyKey> asDeps)
      throws InterruptedException, TransitionException, OptionsParsingException {
    Map<Label, Target> labelsToTargets = new LinkedHashMap<>();
    for (TargetAndConfiguration node : nodes) {
      labelsToTargets.put(node.getTarget().getLabel(), node.getTarget());
    }

    // Maps <target, originalConfig> pairs to <target, finalConfig> pairs for targets that
    // could be successfully Skyframe-evaluated.
    Map<TargetAndConfiguration, TargetAndConfiguration> successfullyEvaluatedTargets =
        new LinkedHashMap<>();
    for (BuildConfiguration fromConfig : asDeps.keySet()) {
      Multimap<DependencyKey, BuildConfiguration> trimmedTargets =
          getConfigurations(env, fromConfig.getOptions(), asDeps.get(fromConfig));
      if (trimmedTargets == null) {
        continue;
      }
      for (Map.Entry<DependencyKey, BuildConfiguration> trimmedTarget : trimmedTargets.entries()) {
        Target target = labelsToTargets.get(trimmedTarget.getKey().getLabel());
        successfullyEvaluatedTargets.put(
            new TargetAndConfiguration(target, fromConfig),
            new TargetAndConfiguration(target, trimmedTarget.getValue()));
      }
    }

    if (env.valuesMissing()) {
      return null;
    }

    LinkedHashSet<TargetAndConfiguration> result = new LinkedHashSet<>();
    for (TargetAndConfiguration originalNode : nodes) {
      // If the configuration couldn't be determined (e.g. loading phase error), use the original.
      result.add(successfullyEvaluatedTargets.getOrDefault(originalNode, originalNode));
    }
    return result;
  }

  // Keep in sync with {@link SkyframeExecutor#getConfigurations}.
  // Note: this implementation runs inside Skyframe, so it has access to SkyFunction.Environment.
  private Multimap<DependencyKey, BuildConfiguration> getConfigurations(
      SkyFunction.Environment env, BuildOptions fromOptions, Iterable<DependencyKey> keys)
      throws InterruptedException, TransitionException, OptionsParsingException {
    Multimap<DependencyKey, BuildConfiguration> builder = ArrayListMultimap.create();

    FragmentClassSet allFragments = ruleClassProvider.getAllFragments();

    // Now get the configurations.
    PathFragment platformMappingPath = fromOptions.get(PlatformOptions.class).platformMappings;
    PlatformMappingValue platformMappingValue =
        (PlatformMappingValue) env.getValue(PlatformMappingValue.Key.create(platformMappingPath));
    if (platformMappingValue == null) {
      return null;
    }

    List<SkyKey> configSkyKeys = new ArrayList<>();
    for (DependencyKey key : keys) {
      if (key.getTransition() == NullTransition.INSTANCE) {
        continue;
      }
      ConfigurationTransition transition = key.getTransition();

      HashMap<PackageValue.Key, PackageValue> buildSettingPackages =
          StarlarkTransition.getBuildSettingPackages(env, transition);
      if (buildSettingPackages == null) {
        return null;
      }
      Collection<BuildOptions> toOptions =
          ConfigurationResolver.applyTransitionWithSkyframe(
                  fromOptions, transition, env, env.getListener())
              .values();
      for (BuildOptions toOption : toOptions) {
        configSkyKeys.add(
            BuildConfigurationValue.keyWithPlatformMapping(
                platformMappingValue, allFragments, toOption));
      }
    }

    Map<SkyKey, SkyValue> configsResult = env.getValues(configSkyKeys);
    if (env.valuesMissing()) {
      return null;
    }

    for (DependencyKey key : keys) {
      if (key.getTransition() == NullTransition.INSTANCE) {
        continue;
      }
      ConfigurationTransition transition = key.getTransition();
      HashMap<PackageValue.Key, PackageValue> buildSettingPackages =
          StarlarkTransition.getBuildSettingPackages(env, transition);
      if (buildSettingPackages == null) {
        return null;
      }
      Collection<BuildOptions> toOptions =
          ConfigurationResolver.applyTransitionWithSkyframe(
                  fromOptions, transition, env, env.getListener())
              .values();
      for (BuildOptions toOption : toOptions) {
        SkyKey configKey =
            BuildConfigurationValue.keyWithPlatformMapping(
                platformMappingValue, allFragments, toOption);
        BuildConfigurationValue configValue =
            ((BuildConfigurationValue) configsResult.get(configKey));
        // configValue will be null here if there was an exception thrown during configuration
        // creation. This will be reported elsewhere.
        if (configValue != null) {
          builder.put(key, configValue.getConfiguration());
        }
      }
    }
    return builder;
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link PrepareAnalysisPhaseFunction#compute}.
   */
  private static final class PrepareAnalysisPhaseFunctionException extends SkyFunctionException {
    PrepareAnalysisPhaseFunctionException(InvalidConfigurationException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
