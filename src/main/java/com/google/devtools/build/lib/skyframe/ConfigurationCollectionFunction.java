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

import com.google.common.base.Supplier;
import com.google.common.base.Verify;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFactory;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.PackageProviderForConfigurations;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.skyframe.ConfigurationCollectionValue.ConfigurationCollectionKey;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * A builder for {@link ConfigurationCollectionValue} instances.
 */
public class ConfigurationCollectionFunction implements SkyFunction {

  private final Supplier<ConfigurationFactory> configurationFactory;
  private final Supplier<Set<Package>> configurationPackages;

  public ConfigurationCollectionFunction(
      Supplier<ConfigurationFactory> configurationFactory,
      Supplier<Set<Package>> configurationPackages) {
    this.configurationFactory = configurationFactory;
    this.configurationPackages = configurationPackages;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException,
      ConfigurationCollectionFunctionException {
    ConfigurationCollectionKey collectionKey = (ConfigurationCollectionKey) skyKey.argument();
    try {
      BuildConfigurationCollection result =
          getConfigurations(env.getListener(),
              new SkyframePackageLoaderWithValueEnvironment(env, configurationPackages.get()),
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
      return new ConfigurationCollectionValue(result, configurationPackages.get());
    } catch (InvalidConfigurationException e) {
      throw new ConfigurationCollectionFunctionException(e);
    }
  }

  /** Create the build configurations with the given options. */
  private BuildConfigurationCollection getConfigurations(EventHandler eventHandler,
      PackageProviderForConfigurations loadedPackageProvider, BuildOptions buildOptions,
      ImmutableSet<String> multiCpu)
          throws InvalidConfigurationException {
    List<BuildConfiguration> targetConfigurations = new ArrayList<>();
    if (!multiCpu.isEmpty()) {
      for (String cpu : multiCpu) {
        BuildConfiguration targetConfiguration = createConfiguration(
            eventHandler, loadedPackageProvider, buildOptions, cpu);
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
          eventHandler, loadedPackageProvider, buildOptions, null);
      if (targetConfiguration == null) {
        return null;
      }
      targetConfigurations.add(targetConfiguration);
    }

    // Sanity check that all host configs are the same. This may not be true once we have
    // better support for multi-host builds.
    BuildConfiguration hostConfiguration =
        targetConfigurations.get(0).getConfiguration(ConfigurationTransition.HOST);
    for (BuildConfiguration targetConfig :
        targetConfigurations.subList(1, targetConfigurations.size())) {
      Verify.verify(
          targetConfig.getConfiguration(ConfigurationTransition.HOST).equals(hostConfiguration));
    }

    return new BuildConfigurationCollection(targetConfigurations, hostConfiguration);
  }

  @Nullable
  public BuildConfiguration createConfiguration(
      EventHandler originalEventListener,
      PackageProviderForConfigurations loadedPackageProvider,
      BuildOptions buildOptions, String cpuOverride) throws InvalidConfigurationException {
    StoredEventHandler errorEventListener = new StoredEventHandler();
    if (cpuOverride != null) {
      // TODO(bazel-team): Options classes should be immutable. This is a bit of a hack.
      buildOptions = buildOptions.clone();
      buildOptions.get(BuildConfiguration.Options.class).cpu = cpuOverride;
    }

    BuildConfiguration targetConfig = configurationFactory.get().createConfiguration(
        loadedPackageProvider, buildOptions, errorEventListener);
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
