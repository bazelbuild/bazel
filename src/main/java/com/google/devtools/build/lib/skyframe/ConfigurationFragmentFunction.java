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
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.skyframe.ConfigurationFragmentValue.ConfigurationFragmentKey;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * A builder for {@link ConfigurationFragmentValue}s.
 */
public final class ConfigurationFragmentFunction implements SkyFunction {
  private final Supplier<ImmutableList<ConfigurationFragmentFactory>> configurationFragments;

  public ConfigurationFragmentFunction(
      Supplier<ImmutableList<ConfigurationFragmentFactory>> configurationFragments) {
    this.configurationFragments = configurationFragments;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws ConfigurationFragmentFunctionException {
    ConfigurationFragmentKey configurationFragmentKey = 
        (ConfigurationFragmentKey) skyKey.argument();
    BuildOptions buildOptions = configurationFragmentKey.getBuildOptions();
    ConfigurationFragmentFactory factory = getFactory(configurationFragmentKey.getFragmentType());
    try {
      Fragment fragment = factory.create(buildOptions);

      if (env.valuesMissing()) {
        return null;
      }
      return new ConfigurationFragmentValue(fragment);
    } catch (InvalidConfigurationException e) {
      // TODO(bazel-team): Rework the control-flow here so that we're not actually throwing this
      // exception with missing Skyframe dependencies.
      if (env.valuesMissing()) {
        return null;
      }
      throw new ConfigurationFragmentFunctionException(e);
    }
  }

  private ConfigurationFragmentFactory getFactory(Class<? extends Fragment> fragmentType) {
    for (ConfigurationFragmentFactory factory : configurationFragments.get()) {
      if (factory.creates().equals(fragmentType)) {
        return factory;
      }
    }
    throw new IllegalStateException(
        "There is no factory for fragment: " + fragmentType.getSimpleName());
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link ConfigurationFragmentFunction#compute}.
   */
  private static final class ConfigurationFragmentFunctionException extends SkyFunctionException {
    public ConfigurationFragmentFunctionException(InvalidConfigurationException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
