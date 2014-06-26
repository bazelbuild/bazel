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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.blaze.BlazeDirectories;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.events.StoredErrorEventListener;
import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.graph.Node;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.LoadedPackageProvider;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.Label.SyntaxException;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.view.ConfigurationCollectionFactory;
import com.google.devtools.build.lib.view.config.BuildConfiguration.Fragment;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

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
  private final FragmentFactories fragmentFactories;
  private final ConfigurationCollectionFactory configurationCollectionFactory;

  // A cache of key to configuration instances.
  private final Cache<String, BuildConfiguration> hostConfigCache =
      CacheBuilder.newBuilder().softValues().build();

  private final Cache<String, TargetConfigurationCacheEntry> configurationCollectionCache =
      CacheBuilder.newBuilder().softValues().build();

  private boolean performSanityCheck = true;

  public ConfigurationFactory(
      MachineSpecification hostMachineSpecification,
      ConfigurationCollectionFactory configurationCollectionFactory,
      List<ConfigurationFragmentFactory> fragmentFactories) {
    this.hostMachineSpecification = hostMachineSpecification;
    this.configurationCollectionFactory =
        Preconditions.checkNotNull(configurationCollectionFactory);
    this.fragmentFactories = new FragmentFactories(fragmentFactories);
  }

  public MachineSpecification getHostMachineSpecification() {
    return hostMachineSpecification;
  }

  @VisibleForTesting
  public List<ConfigurationFragmentFactory> getOrderedFragmentFactories() {
    return fragmentFactories.creationOrder;
  }

  @VisibleForTesting
  public void forbidSanityCheck() {
    performSanityCheck = false;
  }

  public Map<Class<? extends Fragment>, ConfigurationFragmentFactory> getFragmentFactoryMap() {
    return fragmentFactories.factoryMap;
  }

  /** Create the build configurations with the given options. */
  public BuildConfigurationCollection getConfigurations(ErrorEventListener listener,
      LoadedPackageProvider loadedPackageProvider, BuildConfigurationKey key)
          throws InvalidConfigurationException {
    return getConfigurations(loadedPackageProvider, key.getBuildOptions(), key.getDirectories(),
        key.getClientEnv(), listener);
  }


  /**
   * Returns a plain BuildConfiguration with no additional configuration
   * information. This method should only be used during tests when no extra
   * configuration components are required.
   */
  @VisibleForTesting
  public BuildConfiguration getTestConfiguration(BlazeDirectories directories,
      LoadedPackageProvider loadedPackageProvider, BuildOptions buildOptions,
      Map<String, String> clientEnv) throws InvalidConfigurationException {
    ConfigurationEnvironment env =
        new ConfigurationEnvironment.TargetProviderEnvironment(loadedPackageProvider);
    return getConfiguration(env, directories, buildOptions, clientEnv, false,
        CacheBuilder.newBuilder().<String, BuildConfiguration>build());
  }

  /**
   * Constructs and returns a set of build configurations for the given options. The reporter is
   * only used to warn about unsupported configurations.
   *
   * <p>Use {@link #getConfigurations(ErrorEventListener, LoadedPackageProvider,
   * BuildConfigurationKey)} instead.
   */
  @VisibleForTesting
  public BuildConfigurationCollection getConfigurations(LoadedPackageProvider loadedPackageProvider,
      BuildOptions buildOptions, BlazeDirectories directories,
      Map<String, String> clientEnv, ErrorEventListener errorEventListener)
      throws InvalidConfigurationException {
    return new BuildConfigurationCollection(getConfiguration(loadedPackageProvider, buildOptions,
        directories, clientEnv, errorEventListener));
  }

  /**
   * Constructs and returns a set of build configurations for the given options. The reporter is
   * only used to warn about unsupported configurations.
   *
   * <p>Use {@link #getConfigurations(ErrorEventListener, LoadedPackageProvider,
   * BuildConfigurationKey)} instead.
   */
  @VisibleForTesting
  public BuildConfiguration getConfiguration(LoadedPackageProvider loadedPackageProvider,
      BuildOptions buildOptions, BlazeDirectories directories,
      Map<String, String> clientEnv, ErrorEventListener errorEventListener)
      throws InvalidConfigurationException {
    BuildConfiguration.Options commonOptions = buildOptions.get(BuildConfiguration.Options.class);
    // Try to get a cache hit on the entire configuration collection.
    String cacheKey = StringUtilities.combineKeys(
        // NOTE: identityHashCode isn't sound; may cause tests to fail.
        String.valueOf(System.identityHashCode(directories.getOutputBase().getFileSystem())),
        directories.getOutputBase().toString(),
        directories.getWorkspace().toString(),
        buildOptions.computeCacheKey(),
        BuildConfiguration.getTestEnv(commonOptions.testEnvironment, clientEnv).toString());

    TargetConfigurationCacheEntry cacheEntry =
        configurationCollectionCache.getIfPresent(cacheKey);
    if ((cacheEntry == null) || !cacheEntry.isUpToDate(loadedPackageProvider)) {
      cacheEntry = makeCacheEntry(loadedPackageProvider, buildOptions, directories, clientEnv);
      configurationCollectionCache.put(cacheKey, cacheEntry);
    }
    cacheEntry.storedErrorEventListener.replayOn(errorEventListener);
    if (cacheEntry.storedErrorEventListener.hasErrors()) {
      throw new InvalidConfigurationException("Build options are invalid");
    }
    return cacheEntry.targetConfiguration;
  }

  /** Create the build configurations with the given options. */
  public BuildConfigurationCollection getConfigurationsInSkyframe(ErrorEventListener listener,
      LoadedPackageProvider loadedPackageProvider, BuildConfigurationKey key)
          throws InvalidConfigurationException {
    BuildOptions buildOptions = key.getBuildOptions();
    BlazeDirectories directories = key.getDirectories();
    Map<String, String> clientEnv = key.getClientEnv();

    TargetConfigurationCacheEntry cacheEntry =
        makeCacheEntry(loadedPackageProvider, buildOptions, directories, clientEnv);
    cacheEntry.storedErrorEventListener.replayOn(listener);
    if (cacheEntry.storedErrorEventListener.hasErrors()) {
      throw new InvalidConfigurationException("Build options are invalid");
    }
    return new BuildConfigurationCollection(cacheEntry.targetConfiguration);
  }

  private TargetConfigurationCacheEntry makeCacheEntry(
      LoadedPackageProvider loadedPackageProvider,
      BuildOptions buildOptions,
      BlazeDirectories directories,
      Map<String, String> clientEnv) throws InvalidConfigurationException {
    StoredErrorEventListener errorEventListener = new StoredErrorEventListener();

    CachingConfigurationEnvironment env =
        new CachingConfigurationEnvironment(loadedPackageProvider);

    BuildConfiguration targetConfig = configurationCollectionFactory.createConfigurations(this,
        hostMachineSpecification, loadedPackageProvider, buildOptions, directories, clientEnv,
        errorEventListener, env, performSanityCheck);
    return new TargetConfigurationCacheEntry(targetConfig, env.targets, env.nonExistentLabels,
        env.paths, errorEventListener);
  }

  /**
   * Returns a (possibly new) canonical host BuildConfiguration instance based
   * upon a given request configuration
   */
  public BuildConfiguration getHostConfiguration(ConfigurationEnvironment env,
      BlazeDirectories directories, Map<String, String> clientEnv, BuildOptions buildOptions,
      boolean fallback) throws InvalidConfigurationException {
    return getConfiguration(env, directories, buildOptions.createHostOptions(fallback), clientEnv,
        false, hostConfigCache);
  }

  /**
   * The core of BuildConfiguration creation. All host and target instances are
   * constructed and cached here.
   */
  public BuildConfiguration getConfiguration(ConfigurationEnvironment env,
      BlazeDirectories directories, BuildOptions buildOptions,
      Map<String, String> clientEnv, boolean actionsDisabled,
      Cache<String, BuildConfiguration> cache)
          throws InvalidConfigurationException {
    // Create configuration fragments
    Map<Class<? extends Fragment>, Fragment> fragments = new HashMap<>();
    for (ConfigurationFragmentFactory factory : fragmentFactories.creationOrder) {
      // Assemble a strict subset of required fragments (see FragmentLoader.requires)
      Map<Class<? extends Fragment>, Fragment> fragmentsSubset = new HashMap<>();
      for (Class<? extends Fragment> dep : fragmentFactories.dependencies.get(factory)) {
        fragmentsSubset.put(dep, fragments.get(dep));
      }

      Fragment fragment = factory.create(env, directories, buildOptions, fragmentsSubset);
      if (fragment != null) {
        if (fragments.put(fragment.getClass(), fragment) != null) {
          throw new InvalidConfigurationException(
              fragment.getClass() + " is created more than once.");
        }
      }
    }

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

  /**
   * A {@link ConfigurationEnvironment} implementation that keeps a list of all packages seen in the
   * process.
   */
  private final class CachingConfigurationEnvironment implements ConfigurationEnvironment {
    private final LoadedPackageProvider loadedPackageProvider;
    /** The set of targets seen. */
    private final Set<Target> targets = new HashSet<>();
    /** The set of labels that were not found. */
    private final Set<Label> nonExistentLabels = new HashSet<>();
    /** The set of paths seen. */
    private final Map<Path, String> paths = new HashMap<>();

    CachingConfigurationEnvironment(LoadedPackageProvider loadedPackageProvider) {
      this.loadedPackageProvider = loadedPackageProvider;
    }

    @Override
    public Target getTarget(Label label) throws NoSuchPackageException, NoSuchTargetException {
      Target target = null;
      try {
        target = loadedPackageProvider.getLoadedTarget(label);
        targets.add(target);
      } finally {
        if (target == null) {
          // That means an exception was thrown before we got here.
          nonExistentLabels.add(label);
        }
      }
      return target;
    }

    @Override
    public Path getPath(Package pkg, String fileName) {
      Path result = pkg.getPackageDirectory().getRelative(fileName);
      try {
        // TODO(bazel-team): remove this line and field, after removing old-style configuration
        // creation from BlazeInfoCommand
        paths.put(result,
            BaseEncoding.base16().lowerCase().encode(result.getMD5Digest()));
        loadedPackageProvider.addDependency(pkg, fileName);
      } catch (IOException | SyntaxException e) {
        return null;
      }
      return result;
    }
  }

  /**
   * A cache entry of target {@link BuildConfiguration} instances.
   *
   * <p>This also keeps the list of packages that were accessed during the creation of the entry.
   * This allows quickly checking if the cache entry is safe to be re-used, as long as the creation
   * is hermetic (i.e. only the {@link ConfigurationEnvironment} and the configuration options are
   * used, but not, for example, files directly read from the file system).
   */
  private static final class TargetConfigurationCacheEntry {
    private final BuildConfiguration targetConfiguration;
    private final ImmutableList<Target> targets;
    private final ImmutableList<Label> nonExistentLabels;
    private final ImmutableMap<Path, String> paths;
    private final StoredErrorEventListener storedErrorEventListener;

    private TargetConfigurationCacheEntry(BuildConfiguration targetConfiguration,
        Set<Target> targets, Set<Label> nonExistentLabels, Map<Path, String> paths,
        StoredErrorEventListener storedErrorEventListener) {
      this.targetConfiguration = targetConfiguration;
      this.targets = ImmutableList.copyOf(targets);
      this.nonExistentLabels = ImmutableList.copyOf(nonExistentLabels);
      this.paths = ImmutableMap.copyOf(paths);
      this.storedErrorEventListener = storedErrorEventListener;
    }

    public boolean isUpToDate(LoadedPackageProvider loadedPackageProvider) {
      if (!targetConfiguration.supportsIncrementalBuild()) {
        return false;
      }
      try {
        for (Target target : targets) {
          if (!loadedPackageProvider.isTargetCurrent(target)) {
            return false;
          }
        }
        for (Label label : nonExistentLabels) {
          // If the target exists now, then the cache entry is no longer up-to-date.
          if (exists(loadedPackageProvider, label)) {
            return false;
          }
        }
        for (Map.Entry<Path, String> entry : paths.entrySet()) {
          String currentMd5 = BaseEncoding.base16().lowerCase().encode(
              entry.getKey().getMD5Digest());
          if (!currentMd5.equals(entry.getValue())) {
            return false;
          }
        }
        return true;
      } catch (IOException e) {
        return false;
      }
    }

    private boolean exists(LoadedPackageProvider loadedPackageProvider, Label label) {
      try {
        loadedPackageProvider.getLoadedTarget(label);
        return true;
      } catch (NoSuchPackageException | NoSuchTargetException e) {
        return false;
      }
    }
  }

  /**
   * This class creates a topological order for configuration fragments factories.
   * Takes dependency information from {@code ConfigurationFragmentFactory.requires()}.
   */
  private class FragmentFactories {
    // The topological order of fragment factories
    final List<ConfigurationFragmentFactory> creationOrder;

    // Mapping from fragment to their factories
    final Map<Class<? extends Fragment>, ConfigurationFragmentFactory> factoryMap;

    // Mapping from fragments to their dependencies
    final Map<ConfigurationFragmentFactory, List<Class<? extends Fragment>>> dependencies;

    FragmentFactories(List<ConfigurationFragmentFactory> factories) {
      ImmutableMap.Builder<Class<? extends Fragment>, ConfigurationFragmentFactory> factoryBuilder =
          ImmutableMap.builder();
      ImmutableMap.Builder<ConfigurationFragmentFactory,
          List<Class<? extends Fragment>>> depsBuilder = ImmutableMap.builder();
      // Adding fragments to a directed graph, with each edge representing a dependency.
      // The topological ordering of the digraph nodes represent a correct build order.
      Digraph<Class<? extends Fragment>> dependencyGraph = new Digraph<>();
      for (ConfigurationFragmentFactory factory : factories) {
        dependencyGraph.createNode(factory.creates());
        factoryBuilder.put(factory.creates(), factory);
        depsBuilder.put(factory, factory.requires());
        for (Class<? extends Fragment> dependency : factory.requires()) {
          dependencyGraph.addEdge(dependency, factory.creates());
        }
      }
      factoryMap = factoryBuilder.build();
      dependencies = depsBuilder.build();

      ImmutableList.Builder<ConfigurationFragmentFactory> builder = ImmutableList.builder();
      for (Node<Class<? extends Fragment>> fragment : dependencyGraph.getTopologicalOrder()) {
        ConfigurationFragmentFactory factory = factoryMap.get(fragment.getLabel());
        if (factory == null) {
          throw new RuntimeException("There is no configuration loader for " + fragment.getLabel());
        }
        builder.add(factory);
      }
      creationOrder = builder.build();
      return;
    }
  }
}
