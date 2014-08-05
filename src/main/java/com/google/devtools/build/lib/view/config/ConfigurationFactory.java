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

package com.google.devtools.build.lib.view.config;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.blaze.BlazeDirectories;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.events.StoredErrorEventListener;
import com.google.devtools.build.lib.view.ConfigurationCollectionFactory;
import com.google.devtools.build.lib.view.config.BuildConfiguration.Fragment;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.annotation.Nullable;

/**
 * A factory class for {@link BuildConfiguration} instances. This is
 * unfortunately more complex, and should be simplified in the future, if
 * possible. Right now, creating a {@link BuildConfiguration} instance involves
 * creating the instance itself and the related configurations; the main method
 * is {@link #getConfigurations}.
 *
 * <p>This class also defines which target configuration options are written
 * through to the host configuration; it then checks whether the host
 * configuration can be run on the current machine - if that is not the case,
 * then it falls back to the default options.
 *
 * <p>Blaze currently relies on the fact that all {@link BuildConfiguration}
 * instances used in a build can be constructed ahead of time by this class.
 */
@ThreadCompatible // safe as long as separate instances are used
public final class ConfigurationFactory {

  /**
   * The machine configuration for the host, which is used to validate host
   * configurations.
   */
  private final MachineSpecification hostMachineSpecification;
  private final List<ConfigurationFragmentFactory> configurationFragmentFactories;
  private final ConfigurationCollectionFactory configurationCollectionFactory;

  // A cache of key to configuration instances.
  private final Cache<String, BuildConfiguration> hostConfigCache =
      CacheBuilder.newBuilder().softValues().build();

  private boolean performSanityCheck = true;

  public ConfigurationFactory(
      MachineSpecification hostMachineSpecification,
      ConfigurationCollectionFactory configurationCollectionFactory,
      List<ConfigurationFragmentFactory> fragmentFactories) {
    this.hostMachineSpecification = hostMachineSpecification;
    this.configurationCollectionFactory =
        Preconditions.checkNotNull(configurationCollectionFactory);
    this.configurationFragmentFactories = fragmentFactories;
  }

  public MachineSpecification getHostMachineSpecification() {
    return hostMachineSpecification;
  }

  @VisibleForTesting
  public void forbidSanityCheck() {
    performSanityCheck = false;
  }

  /**
   * Returns a plain BuildConfiguration with no additional configuration
   * information. This method should only be used during tests when no extra
   * configuration components are required.
   */
  @VisibleForTesting
  public BuildConfiguration getTestConfiguration(
      PackageProviderForConfigurations loadedPackageProvider, BuildOptions buildOptions,
      Map<String, String> clientEnv) throws InvalidConfigurationException {    
    return getConfiguration(loadedPackageProvider, buildOptions, clientEnv, false,
        CacheBuilder.newBuilder().<String, BuildConfiguration>build());
  }

  /** Create the build configurations with the given options. */
  @Nullable
  public BuildConfigurationCollection getConfigurations(ErrorEventListener listener,
      PackageProviderForConfigurations loadedPackageProvider, BuildConfigurationKey key)
          throws InvalidConfigurationException {
    List<BuildConfiguration> targetConfigurations = new ArrayList<>();
    if (!key.getMultiCpu().isEmpty()) {
      for (String cpu : key.getMultiCpu()) {
        BuildConfiguration targetConfiguration = createConfiguration(
            listener, loadedPackageProvider, key, cpu);
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
          listener, loadedPackageProvider, key, null);
      if (targetConfiguration == null) {
        return null;
      }
      targetConfigurations.add(targetConfiguration);
    }
    return new BuildConfigurationCollection(targetConfigurations);
  }

  @Nullable
  private BuildConfiguration createConfiguration(
      ErrorEventListener originalEventListener,
      PackageProviderForConfigurations loadedPackageProvider,
      BuildConfigurationKey key, String cpuOverride) throws InvalidConfigurationException {
    StoredErrorEventListener errorEventListener = new StoredErrorEventListener();
    BuildOptions buildOptions = key.getBuildOptions();
    if (cpuOverride != null) {
      // TODO(bazel-team): Options classes should be immutable. This is a bit of a hack.
      buildOptions = buildOptions.clone();
      buildOptions.get(BuildConfiguration.Options.class).cpu = cpuOverride;
    }

    BuildConfiguration targetConfig = configurationCollectionFactory.createConfigurations(this,
        hostMachineSpecification, loadedPackageProvider, buildOptions, key.getClientEnv(),
        errorEventListener, performSanityCheck);
    if (targetConfig == null) {
      return null;
    }
    errorEventListener.replayOn(originalEventListener);
    if (errorEventListener.hasErrors()) {
      throw new InvalidConfigurationException("Build options are invalid");
    }
    return targetConfig;
  }

  /**
   * Returns a (possibly new) canonical host BuildConfiguration instance based
   * upon a given request configuration
   */
  @Nullable
  public BuildConfiguration getHostConfiguration(
      PackageProviderForConfigurations loadedPackageProvider, Map<String, String> clientEnv,
      BuildOptions buildOptions, boolean fallback) throws InvalidConfigurationException {
    return getConfiguration(loadedPackageProvider, buildOptions.createHostOptions(fallback),
        clientEnv, false, hostConfigCache);
  }

  /**
   * The core of BuildConfiguration creation. All host and target instances are
   * constructed and cached here.
   */
  @Nullable
  public BuildConfiguration getConfiguration(PackageProviderForConfigurations loadedPackageProvider,
      BuildOptions buildOptions, Map<String, String> clientEnv,
      boolean actionsDisabled, Cache<String, BuildConfiguration> cache)
          throws InvalidConfigurationException {
    
    Map<Class<? extends Fragment>, Fragment> fragments = new HashMap<>();
    // Create configuration fragments
    for (ConfigurationFragmentFactory factory : configurationFragmentFactories) {
      Class<? extends Fragment> fragmentType = factory.creates();
      Fragment fragment = loadedPackageProvider.getFragment(buildOptions, fragmentType);
      if (fragment != null && fragments.get(fragment) == null) {
        fragments.put(fragment.getClass(), fragment);
      }
    }
    BlazeDirectories directories = loadedPackageProvider.getDirectories();
    if (loadedPackageProvider.valuesMissing()) {
      return null;
    }

    // Sort the fragments by class name to make sure that the order is stable. Afterwards, copy to
    // an ImmutableMap, which keeps the order stable, but uses hashing, and drops the reference to
    // the Comparator object.
    fragments = ImmutableSortedMap.copyOf(fragments, new Comparator<Class<? extends Fragment>>() {
      @Override
      public int compare(Class<? extends Fragment> o1, Class<? extends Fragment> o2) {
        return o1.getName().compareTo(o2.getName());
      }
    });
    fragments = ImmutableMap.copyOf(fragments);

    String key = BuildConfiguration.computeCacheKey(
        directories, fragments, buildOptions, clientEnv);
    BuildConfiguration configuration = cache.getIfPresent(key);
    if (configuration == null) {
      configuration = new BuildConfiguration(directories, fragments, buildOptions,
          clientEnv, actionsDisabled);
      cache.put(key, configuration);
    }
    return configuration;
  }

  public List<ConfigurationFragmentFactory> getFactories() {
    return configurationFragmentFactories;
  }
}
