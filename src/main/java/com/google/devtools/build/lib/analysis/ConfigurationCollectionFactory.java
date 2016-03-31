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
package com.google.devtools.build.lib.analysis;

import com.google.common.cache.Cache;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFactory;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.PackageProviderForConfigurations;
import com.google.devtools.build.lib.events.EventHandler;

import javax.annotation.Nullable;

/**
 * A factory for configuration collection creation.
 */
public interface ConfigurationCollectionFactory {
  /**
   * Creates the top-level configuration for a build.
   *
   * <p>Also it may create a set of BuildConfigurations and define a transition table over them.
   * All configurations during a build should be accessible from this top-level configuration
   * via configuration transitions.
   * @param configurationFactory the configuration factory
   * @param cache a cache for BuildConfigurations
   * @param loadedPackageProvider the package provider
   * @param buildOptions top-level build options representing the command-line
   * @param errorEventListener the event listener for errors
   * @return the top-level configuration
   * @throws InvalidConfigurationException
   */
  @Nullable
  BuildConfiguration createConfigurations(
      ConfigurationFactory configurationFactory,
      Cache<String, BuildConfiguration> cache,
      PackageProviderForConfigurations loadedPackageProvider,
      BuildOptions buildOptions,
      EventHandler errorEventListener) throws InvalidConfigurationException;

  /**
   * Returns the module the given configuration should use for choosing dynamic transitions.
   *
   * <p>We can presumably factor away this method once static global configurations are properly
   * deprecated. But for now we retain the
   * {@link com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection.Transitions}
   * interface since that's the same place where static transition logic is determined and
   * {@link BuildConfigurationCollection.Transitions#configurationHook}
   * is still used.
   */
  BuildConfigurationCollection.Transitions getDynamicTransitionLogic(BuildConfiguration config);
}
