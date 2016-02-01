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
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFactory;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.PackageProviderForConfigurations;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.skyframe.ConfigurationCollectionValue.ConfigurationCollectionKey;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.ArrayList;
import java.util.List;

import javax.annotation.Nullable;

/**
 * A builder for {@link ConfigurationCollectionValue} instances.
 */
public class ConfigurationCollectionFunction implements SkyFunction {

  private final Supplier<ConfigurationFactory> configurationFactory;
  private final RuleClassProvider ruleClassProvider;

  public ConfigurationCollectionFunction(Supplier<ConfigurationFactory> configurationFactory,
      RuleClassProvider ruleClassProvider) {
    this.configurationFactory = configurationFactory;
    this.ruleClassProvider = ruleClassProvider;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException,
      ConfigurationCollectionFunctionException {
    ConfigurationCollectionKey collectionKey = (ConfigurationCollectionKey) skyKey.argument();
    try {
      BuildConfigurationCollection result = getConfigurations(env,
          new SkyframePackageLoaderWithValueEnvironment(env, ruleClassProvider),
          collectionKey.getBuildOptions(), collectionKey.getMultiCpu());

      // BuildConfigurationCollection can be created, but dependencies to some files might be
      // missing. In that case we need to build configurationCollection again.
      if (env.valuesMissing()) {
        return null;
      }

      for (BuildConfiguration config : result.getTargetConfigurations()) {
        config.declareSkyframeDependencies(env);
      }
      if (env.valuesMissing()) {
        return null;
      }
      return new ConfigurationCollectionValue(result);
    } catch (InvalidConfigurationException e) {
      throw new ConfigurationCollectionFunctionException(e);
    }
  }

  /** Create the build configurations with the given options. */
  private BuildConfigurationCollection getConfigurations(Environment env,
      PackageProviderForConfigurations loadedPackageProvider, BuildOptions buildOptions,
      ImmutableSet<String> multiCpu)
          throws InvalidConfigurationException {
    // We cache all the related configurations for this target configuration in a cache that is
    // dropped at the end of this method call. We instead rely on the cache for entire collections
    // for caching the target and related configurations, and on a dedicated host configuration
    // cache for the host configuration.
    Cache<String, BuildConfiguration> cache =
        CacheBuilder.newBuilder().<String, BuildConfiguration>build();
    List<BuildConfiguration> targetConfigurations = new ArrayList<>();

    if (!multiCpu.isEmpty()) {
      for (String cpu : multiCpu) {
        BuildConfiguration targetConfiguration = createConfiguration(
         cache, env.getListener(), loadedPackageProvider, buildOptions, cpu);
        if (targetConfiguration == null || targetConfigurations.contains(targetConfiguration)) {
          continue;
        }
        targetConfigurations.add(targetConfiguration);
      }
      if (loadedPackageProvider.valuesMissing()) {
        return null;
      }
    } else {
      BuildConfiguration targetConfiguration = createConfiguration(
         cache, env.getListener(), loadedPackageProvider, buildOptions, null);
      if (targetConfiguration == null) {
        return null;
      }
      targetConfigurations.add(targetConfiguration);
    }
    BuildConfiguration hostConfiguration = getHostConfiguration(env, targetConfigurations.get(0));
    if (hostConfiguration == null) {
      return null;
    }

    return new BuildConfigurationCollection(targetConfigurations, hostConfiguration);
  }

  /**
   * Returns the host configuration, or null on missing Skyframe deps.
   */
  private BuildConfiguration getHostConfiguration(Environment env,
      BuildConfiguration targetConfiguration) throws InvalidConfigurationException {
    if (targetConfiguration.useDynamicConfigurations()) {
      BuildOptions hostOptions = HostTransition.INSTANCE.apply(targetConfiguration.getOptions());
      SkyKey hostConfigKey =
          BuildConfigurationValue.key(targetConfiguration.fragmentClasses(), hostOptions);
      BuildConfigurationValue skyValHost = (BuildConfigurationValue)
          env.getValueOrThrow(hostConfigKey, InvalidConfigurationException.class);

      // Also preload the target configuration so the configured target functions for
      // top-level targets don't have to waste cycles from a missing Skyframe dep.
      BuildOptions targetOptions = targetConfiguration.getOptions();
      SkyKey targetConfigKey =
          BuildConfigurationValue.key(targetConfiguration.fragmentClasses(), targetOptions);
      BuildConfigurationValue skyValTarget = (BuildConfigurationValue)
          env.getValueOrThrow(targetConfigKey, InvalidConfigurationException.class);

      if (skyValHost == null || skyValTarget == null) {
        return null;
      }
      return skyValHost.getConfiguration();
    } else {
      return targetConfiguration.getConfiguration(Attribute.ConfigurationTransition.HOST);
    }
  }

  @Nullable
  private BuildConfiguration createConfiguration(
      Cache<String, BuildConfiguration> cache,
      EventHandler originalEventListener,
      PackageProviderForConfigurations loadedPackageProvider,
      BuildOptions buildOptions, String cpuOverride) throws InvalidConfigurationException {
    StoredEventHandler errorEventListener = new StoredEventHandler();
    if (cpuOverride != null) {
      // TODO(bazel-team): Options classes should be immutable. This is a bit of a hack.
      buildOptions = buildOptions.clone();
      buildOptions.get(BuildConfiguration.Options.class).cpu = cpuOverride;
    }

    BuildConfiguration targetConfig = configurationFactory.get().createConfigurations(
        cache, loadedPackageProvider, buildOptions, errorEventListener);
    if (targetConfig == null) {
      return null;
    }
    errorEventListener.replayOn(originalEventListener);
    if (errorEventListener.hasErrors()) {
      throw new InvalidConfigurationException("Build options are invalid");
    }
    return targetConfig;
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link ConfigurationCollectionFunction#compute}.
   */
  private static final class ConfigurationCollectionFunctionException extends
      SkyFunctionException {
    public ConfigurationCollectionFunctionException(InvalidConfigurationException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
