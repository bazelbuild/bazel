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

package com.google.devtools.build.lib.bazel.rules;

import com.google.common.cache.Cache;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Table;
import com.google.devtools.build.lib.analysis.ConfigurationCollectionFactory;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection.ConfigurationHolder;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection.Transitions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFactory;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.PackageProviderForConfigurations;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.Attribute.SplitTransition;
import com.google.devtools.build.lib.packages.Attribute.Transition;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses.LipoTransition;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Configuration collection used by the rules Bazel knows.
 */
public class BazelConfigurationCollection implements ConfigurationCollectionFactory {
  @Override
  @Nullable
  public BuildConfiguration createConfigurations(
      ConfigurationFactory configurationFactory,
      Cache<String, BuildConfiguration> cache,
      PackageProviderForConfigurations packageProvider,
      BuildOptions buildOptions,
      EventHandler eventHandler)
      throws InvalidConfigurationException, InterruptedException {
    // Target configuration
    BuildConfiguration targetConfiguration = configurationFactory.getConfiguration(
        packageProvider, buildOptions, false, cache);
    if (targetConfiguration == null) {
      return null;
    }

    BuildConfiguration dataConfiguration = targetConfiguration;

    // Host configuration
    // Note that this passes in the dataConfiguration, not the target
    // configuration. This is intentional.
    BuildConfiguration hostConfiguration = getHostConfigurationFromRequest(configurationFactory,
        packageProvider, dataConfiguration, buildOptions, cache);
    if (hostConfiguration == null) {
      return null;
    }

    ListMultimap<SplitTransition<?>, BuildConfiguration> splitTransitionsTable =
        ArrayListMultimap.create();
    for (SplitTransition<BuildOptions> transition : buildOptions.getPotentialSplitTransitions()) {
      for (BuildOptions splitOptions : transition.split(buildOptions)) {
        BuildConfiguration splitConfig = configurationFactory.getConfiguration(
            packageProvider, splitOptions, false, cache);
        splitTransitionsTable.put(transition, splitConfig);
      }
    }
    if (packageProvider.valuesMissing()) {
      return null;
    }
    BuildConfiguration result = setupTransitions(
        targetConfiguration, dataConfiguration, hostConfiguration, splitTransitionsTable);
    result.reportInvalidOptions(eventHandler);
    return result;
  }

  private static class BazelTransitions extends BuildConfigurationCollection.Transitions {
    public BazelTransitions(BuildConfiguration configuration,
        Map<? extends Transition, ConfigurationHolder> transitionTable,
        ListMultimap<? extends SplitTransition<?>, BuildConfiguration> splitTransitionTable) {
      super(configuration, transitionTable, splitTransitionTable);
    }

    @Override
    protected Transition getDynamicTransition(Transition configurationTransition) {
      if (configurationTransition == ConfigurationTransition.DATA) {
        return ConfigurationTransition.NONE;
      } else {
        return super.getDynamicTransition(configurationTransition);
      }
    }
  }

  @Override
  public Transitions getDynamicTransitionLogic(BuildConfiguration config)  {
    return new BazelTransitions(config, ImmutableMap.<Transition, ConfigurationHolder>of(),
          ImmutableListMultimap.<SplitTransition<?>, BuildConfiguration>of());
  }

  /**
   * Gets the correct host configuration for this build. The behavior depends on the value of the
   * --distinct_host_configuration flag.
   *
   * <p>With --distinct_host_configuration=false, we use identical configurations for the host and
   * target, and you can ignore everything below. But please note: if you're cross-compiling for k8
   * on a piii machine, your build will fail. This is a stopgap measure.
   *
   * <p>Currently, every build is (in effect) a cross-compile, in the strict sense that host and
   * target configurations are unequal, thus we do not issue a "cross-compiling" warning. (Perhaps
   * we should?) *
   *
   * @param requestConfig the requested target (not host!) configuration for this build.
   * @param buildOptions the configuration options used for the target configuration
   */
  @Nullable
  private static BuildConfiguration getHostConfigurationFromRequest(
      ConfigurationFactory configurationFactory,
      PackageProviderForConfigurations loadedPackageProvider,
      BuildConfiguration requestConfig,
      BuildOptions buildOptions,
      Cache<String, BuildConfiguration> cache)
      throws InvalidConfigurationException, InterruptedException {
    BuildConfiguration.Options commonOptions = buildOptions.get(BuildConfiguration.Options.class);
    if (!commonOptions.useDistinctHostConfiguration) {
      return requestConfig;
    } else {
      BuildConfiguration hostConfig = configurationFactory.getConfiguration(
          loadedPackageProvider, buildOptions.createHostOptions(false), false, cache);
      if (hostConfig == null) {
        return null;
      }
      return hostConfig;
    }
  }

  static BuildConfiguration setupTransitions(BuildConfiguration targetConfiguration,
      BuildConfiguration dataConfiguration, BuildConfiguration hostConfiguration,
      ListMultimap<SplitTransition<?>, BuildConfiguration> splitTransitionsTable) {
    Set<BuildConfiguration> allConfigurations = new LinkedHashSet<>();
    allConfigurations.add(targetConfiguration);
    allConfigurations.add(dataConfiguration);
    allConfigurations.add(hostConfiguration);
    allConfigurations.addAll(splitTransitionsTable.values());

    Table<BuildConfiguration, Transition, ConfigurationHolder> transitionBuilder =
        HashBasedTable.create();
    for (BuildConfiguration from : allConfigurations) {
      for (ConfigurationTransition transition : ConfigurationTransition.values()) {
        BuildConfiguration to;
        if (transition == ConfigurationTransition.HOST) {
          to = hostConfiguration;
        } else if (transition == ConfigurationTransition.DATA && from == targetConfiguration) {
          to = dataConfiguration;
        } else {
          to = from;
        }
        transitionBuilder.put(from, transition, new ConfigurationHolder(to));
      }
    }

    // TODO(bazel-team): This makes LIPO totally not work. Just a band-aid until we get around to
    // implementing a way for the C++ rules to contribute this transition to the configuration
    // collection.
    for (BuildConfiguration config : allConfigurations) {
      transitionBuilder.put(config, LipoTransition.LIPO_COLLECTOR,
          new ConfigurationHolder(config));
      transitionBuilder.put(config, LipoTransition.TARGET_CONFIG_FOR_LIPO,
          new ConfigurationHolder(config.isHostConfiguration() ? null : config));
    }

    for (BuildConfiguration config : allConfigurations) {
      // We allow host configurations to be shared between target configurations. In that case, the
      // transitions may already be set.
      // TODO(bazel-team): Check that the transitions are identical, or even better, change the
      // code to set the host configuration transitions before we even create the target
      // configuration.
      if (config.isHostConfiguration() && config.getTransitions() != null) {
        continue;
      }
      boolean isSplitConfig = splitTransitionsTable.values().contains(config);
      // When --experimental_multi_cpu is set, we create multiple target configurations that only
      // differ by --cpu. We may therefore end up with multiple identical split configurations, if
      // the split transition overwrites the cpu, which it usually does.
      if (isSplitConfig && config.getTransitions() != null) {
        continue;
      }
      Transitions outgoingTransitions =
          new BazelTransitions(
              config,
              transitionBuilder.row(config),
              // Split transitions must not have their own split transitions because then they
              // would be applied twice due to a quirk in DependencyResolver. See the comment in
              // DependencyResolver.resolveLateBoundAttributes().
              isSplitConfig
                  ? ImmutableListMultimap.<SplitTransition<?>, BuildConfiguration>of()
                  : splitTransitionsTable);
      config.setConfigurationTransitions(outgoingTransitions);
    }

    return targetConfiguration;
  }
}
