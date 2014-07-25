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
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.lib.view.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.view.config.BuildConfigurationKey;
import com.google.devtools.build.lib.view.config.ConfigurationFactory;
import com.google.devtools.build.lib.view.config.InvalidConfigurationException;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * A builder for {@link ConfigurationCollectionValue}s.
 */
public class ConfigurationCollectionFunction implements SkyFunction {

  private final Supplier<ConfigurationFactory> configurationFactory;
  private final Supplier<BuildConfigurationKey> configurationKey;

  public ConfigurationCollectionFunction(
      Supplier<ConfigurationFactory> configurationFactory,
      Supplier<BuildConfigurationKey> key) {
    this.configurationFactory = configurationFactory;
    this.configurationKey = key;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException,
      ConfigurationCollectionFunctionException {
    try {
      // We are not using these values, because we have copies inside BuildConfigurationKey.
      // Unfortunately, we can't use BuildConfigurationKey as BuildVariableValue, because it
      // contains clientEnvironment and we would have to invalidate ConfigurationCollectionValue
      // each time when any variable in client environment changes.
      BuildVariableValue.BUILD_OPTIONS.get(env);
      BuildVariableValue.TEST_ENVIRONMENT_VARIABLES.get(env);
      BuildVariableValue.BLAZE_DIRECTORIES.get(env);
      if (env.valuesMissing()) {
        return null;
      }

      BuildConfigurationCollection result =
          configurationFactory.get().getConfigurations(env.getListener(),
          new SkyframePackageLoaderWithValueEnvironment(env), configurationKey.get());

      // BuildConfigurationCollection can be created, but dependencies to some files might be
      // missing. In that case we need to build configurationCollection second time.
      if (env.valuesMissing()) {
        return null;
      }
      // For non-incremental builds the configuration collection is not going to be cached.
      for (BuildConfiguration config : result.getTargetConfigurations()) {
        if (!config.supportsIncrementalBuild()) {
          BuildVariableValue.BUILD_ID.get(env);
        }
      }
      return new ConfigurationCollectionValue(result);
    } catch (InvalidConfigurationException e) {
      throw new ConfigurationCollectionFunctionException(skyKey, e);
    }
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
    public ConfigurationCollectionFunctionException(SkyKey key,
        InvalidConfigurationException e) {
      super(key, e);
    }
  }
}
