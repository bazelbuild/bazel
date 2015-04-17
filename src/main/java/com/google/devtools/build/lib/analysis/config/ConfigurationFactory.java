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

package com.google.devtools.build.lib.analysis.config;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfigurationCollectionFactory;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.EventHandler;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.annotation.Nullable;

/**
 * A factory class for {@link BuildConfiguration} instances. This is unfortunately more complex,
 * and should be simplified in the future, if
 * possible. Right now, creating a {@link BuildConfiguration} instance involves
 * creating the instance itself and the related configurations; the main method
 * is {@link #createConfiguration}.
 *
 * <p>Avoid calling into this class, and instead use the skyframe infrastructure to obtain
 * configuration instances.
 *
 * <p>Blaze currently relies on the fact that all {@link BuildConfiguration}
 * instances used in a build can be constructed ahead of time by this class.
 */
@ThreadCompatible // safe as long as separate instances are used
public final class ConfigurationFactory {
  private final List<ConfigurationFragmentFactory> configurationFragmentFactories;
  private final ConfigurationCollectionFactory configurationCollectionFactory;

  // A cache of key to configuration instances.
  private final Cache<String, BuildConfiguration> hostConfigCache =
      CacheBuilder.newBuilder().softValues().build();

  private boolean performSanityCheck = true;

  public ConfigurationFactory(
      ConfigurationCollectionFactory configurationCollectionFactory,
      ConfigurationFragmentFactory... fragmentFactories) {
    this(configurationCollectionFactory, ImmutableList.copyOf(fragmentFactories));
  }

  public ConfigurationFactory(
      ConfigurationCollectionFactory configurationCollectionFactory,
      List<ConfigurationFragmentFactory> fragmentFactories) {
    this.configurationCollectionFactory =
        Preconditions.checkNotNull(configurationCollectionFactory);
    this.configurationFragmentFactories = ImmutableList.copyOf(fragmentFactories);
  }

  @VisibleForTesting
  public void forbidSanityCheck() {
    performSanityCheck = false;
  }

  /** Create the build configurations with the given options. */
  @Nullable
  public BuildConfiguration createConfiguration(
      PackageProviderForConfigurations loadedPackageProvider, BuildOptions buildOptions,
      BuildConfigurationKey key, EventHandler errorEventListener)
          throws InvalidConfigurationException {
    return configurationCollectionFactory.createConfigurations(this,
        loadedPackageProvider, buildOptions, key.getTestEnv(),
        errorEventListener, performSanityCheck);
  }

  /**
   * Returns a (possibly new) canonical host BuildConfiguration instance based
   * upon a given request configuration
   */
  @Nullable
  public BuildConfiguration getHostConfiguration(
      PackageProviderForConfigurations loadedPackageProvider, Map<String, String> testEnv,
      BuildOptions buildOptions, boolean fallback) throws InvalidConfigurationException {
    return getConfiguration(loadedPackageProvider, buildOptions.createHostOptions(fallback),
        testEnv, false, hostConfigCache);
  }

  /**
   * The core of BuildConfiguration creation. All host and target instances are
   * constructed and cached here.
   */
  @Nullable
  public BuildConfiguration getConfiguration(PackageProviderForConfigurations loadedPackageProvider,
      BuildOptions buildOptions, Map<String, String> testEnv,
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
        directories, fragments, buildOptions, testEnv);
    BuildConfiguration configuration = cache.getIfPresent(key);
    if (configuration == null) {
      configuration = new BuildConfiguration(directories, fragments, buildOptions,
          testEnv, actionsDisabled);
      cache.put(key, configuration);
    }
    return configuration;
  }

  public List<ConfigurationFragmentFactory> getFactories() {
    return configurationFragmentFactories;
  }
}
