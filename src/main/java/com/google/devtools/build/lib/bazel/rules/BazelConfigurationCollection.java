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

package com.google.devtools.build.lib.bazel.rules;

import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Table;
import com.google.devtools.build.lib.blaze.BlazeDirectories;
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.pkgcache.LoadedPackageProvider;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.view.ConfigurationCollectionFactory;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.lib.view.config.BuildConfigurationCollection.ConfigurationHolder;
import com.google.devtools.build.lib.view.config.BuildConfigurationCollection.Transitions;
import com.google.devtools.build.lib.view.config.BuildConfigurationKey;
import com.google.devtools.build.lib.view.config.BuildOptions;
import com.google.devtools.build.lib.view.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.view.config.ConfigurationFactory;
import com.google.devtools.build.lib.view.config.InvalidConfigurationException;
import com.google.devtools.build.lib.view.config.MachineSpecification;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Configuration collection used by the rules Bazel knows.
 */
public class BazelConfigurationCollection implements ConfigurationCollectionFactory {
  @Override
  public BuildConfiguration createConfigurations(
      ConfigurationFactory configurationFactory,
      MachineSpecification hostMachineSpecification,
      LoadedPackageProvider loadedPackageProvider,
      BuildOptions buildOptions,
      BlazeDirectories directories,
      Map<String, String> clientEnv,
      ErrorEventListener errorEventListener,
      ConfigurationEnvironment env,
      boolean performSanityCheck) throws InvalidConfigurationException {

    // We cache all the related configurations for this target configuration in a cache that is
    // dropped at the end of this method call. We instead rely on the cache for entire collections
    // for caching the target and related configurations, and on a dedicated host configuration
    // cache for the host configuration.
    Cache<String, BuildConfiguration> cache =
        CacheBuilder.newBuilder().<String, BuildConfiguration>build();

    // Target configuration
    BuildConfiguration targetConfiguration = configurationFactory.getConfiguration(
        env, directories, buildOptions, clientEnv, false, cache);

    BuildConfiguration dataConfiguration = targetConfiguration;

    // Host configuration
    // Note that this passes in the dataConfiguration, not the target
    // configuration. This is intentional.
    BuildConfiguration hostConfiguration = getHostConfigurationFromRequest(configurationFactory,
        env, directories, clientEnv, dataConfiguration, buildOptions, errorEventListener,
        hostMachineSpecification);

    // Sanity check that the implicit labels are all in the transitive closure of explicit ones.
    // This also registers all targets in the cache entry and validates them on subsequent requests.
    Set<Label> reachableLabels = new HashSet<>();
    // TODO(bazel-team): remove first part of condition when legacy code is gone.
    if (loadedPackageProvider != null && performSanityCheck) {
      // We allow the package provider to be null for testing.
      for (Label label : buildOptions.getAllLabels().values()) {
        try {
          collectTransitiveClosure(env, reachableLabels, label);
        } catch (NoSuchThingException e) {
          // We've loaded the transitive closure of the labels-to-load above, and made sure that
          // there are no errors loading it, so this can't happen.
          throw new IllegalStateException(e);
        }
      }
      sanityCheckImplicitLabels(reachableLabels, targetConfiguration);
      sanityCheckImplicitLabels(reachableLabels, hostConfiguration);
    }

    BuildConfiguration result = setupTransitions(
        targetConfiguration, dataConfiguration, hostConfiguration);
    result.reportInvalidOptions(errorEventListener, hostConfiguration);
    return result;
  }

  /**
   * Gets the correct host configuration for this build. The behavior
   * depends on the value of the --distinct_host_configuration flag.
   *
   * <p>With --distinct_host_configuration=false, we use identical configurations
   * for the host and target, and you can ignore everything below.  But please
   * note: if you're cross-compiling for k8 on a piii machine, your build will
   * fail.  This is a stopgap measure.
   *
   * <p>Currently, every build is (in effect) a cross-compile, in the strict
   * sense that host and target configurations are unequal, thus we do not
   * issue a "cross-compiling" warning.  (Perhaps we should?)
   *   *
   * @param requestConfig the requested target (not host!) configuration for
   *   this build.
   * @param buildOptions the configuration options used for the target configuration
   */
  private BuildConfiguration getHostConfigurationFromRequest(
      ConfigurationFactory configurationFactory, ConfigurationEnvironment env,
      BlazeDirectories directories, Map<String, String> clientEnv, BuildConfiguration requestConfig,
      BuildOptions buildOptions, ErrorEventListener errorEventListener,
      MachineSpecification hostMachineSpecification)
      throws InvalidConfigurationException {
    BuildConfiguration.Options commonOptions = buildOptions.get(BuildConfiguration.Options.class);
    if (!commonOptions.useDistinctHostConfiguration) {
      if (!requestConfig.canRunOn(hostMachineSpecification)) {
        throw new InvalidConfigurationException("Your machine cannot execute tools built for the "
            + requestConfig + " configuration; specify --distinct_host_configuration to build the "
            + "tools in a machine-compatible configuration, or use --cpu=piii");
      }
      return requestConfig;
    } else {
      BuildConfiguration hostConfig = configurationFactory.getHostConfiguration(env, directories,
          clientEnv, buildOptions, /*fallback=*/false);
      // Check that the user's inputs are sensible. If they are not, retry with a default value.
      if (!hostConfig.canRunOn(hostMachineSpecification)) {
        errorEventListener.warn(null, "The host configuration appears to contain settings that "
            + "are incompatible with the machine the build is run on.");
        // TODO(bazel-team): Unfortunately, we don't have distinct options for the host
        // configuration, so for now we fall back to a known configuration. We need to add a full
        // set of options to control the host configuration and then remove the fallback and fail
        // with a meaningful error message.
        hostConfig = configurationFactory.getHostConfiguration(env, directories, clientEnv,
            buildOptions, /*fallback=*/true);
      }
      return hostConfig;
    }
  }

  static BuildConfiguration setupTransitions(BuildConfiguration targetConfiguration,
      BuildConfiguration dataConfiguration, BuildConfiguration hostConfiguration) {
    Set<BuildConfiguration> allConfigurations = ImmutableSet.of(targetConfiguration,
        dataConfiguration, hostConfiguration);

    Table<BuildConfiguration, ConfigurationTransition, ConfigurationHolder> transitionBuilder =
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

    for (BuildConfiguration config : allConfigurations) {
      Transitions outgoingTransitions =
          new BazelConfigurationTransitions(config, transitionBuilder.row(config));
      // We allow host configurations to be shared between target configurations. In that case, the
      // transitions may already be set.
      // TODO(bazel-team): Check that the transitions are identical, or even better, change the
      // code to set the host configuration transitions before we even create the target
      // configuration.
      if (config.isHostConfiguration() && config.getTransitions() != null) {
        continue;
      }
      config.setConfigurationTransitions(outgoingTransitions);
    }

    // Invariant: the cacheKey() values must be distinct if the configurations are not equal.
    // Otherwise Blaze gets very confused.
    HashMap<String, BuildConfiguration> cacheKeyConflictDetector = new HashMap<>();
    for (BuildConfiguration config : allConfigurations) {
      if (cacheKeyConflictDetector.containsKey(config.cacheKey())) {
        throw new IllegalStateException(config + " "
            + cacheKeyConflictDetector.get(config.cacheKey()));
      }
      cacheKeyConflictDetector.put(config.cacheKey(), config);
    }

    return targetConfiguration;
  }

  /**
   * Checks that the implicit labels are reachable from the loaded labels. The loaded labels are
   * those returned from {@link BuildConfigurationKey#getLabelsToLoadUnconditionally()}, and the
   * implicit ones are those that need to be available for late-bound attributes.
   */
  private void sanityCheckImplicitLabels(Collection<Label> reachableLabels,
      BuildConfiguration config) throws InvalidConfigurationException {
    for (Map.Entry<String, Label> entry : config.getImplicitLabels().entries()) {
      if (!reachableLabels.contains(entry.getValue())) {
        throw new InvalidConfigurationException("The required " + entry.getKey()
            + " target is not transitively reachable from a command-line option: '"
            + entry.getValue() + "'");
      }
    }
  }

  private void collectTransitiveClosure(ConfigurationEnvironment env,
      Set<Label> reachableLabels, Label from) throws NoSuchThingException {
    if (!reachableLabels.add(from)) {
      return;
    }
    Target fromTarget = env.getTarget(from);
    if (fromTarget instanceof Rule) {
      Rule rule = (Rule) fromTarget;
      if (rule.getRuleClassObject().hasAttr("srcs", Type.LABEL_LIST)) {
        for (Label label : rule.get("srcs", Type.LABEL_LIST)) {
          collectTransitiveClosure(env, reachableLabels, label);
        }
      }
    }
  }
}
