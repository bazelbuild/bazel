// Copyright 2015 Google Inc. All rights reserved.
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

import static com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;

import com.google.common.collect.ClassToInstanceMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.MutableClassToInstanceMap;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException;

import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

/**
 * A builder for {@link BuildConfigurationValue} instances.
 */
public class BuildConfigurationFunction implements SkyFunction {

  private final BlazeDirectories directories;

  public BuildConfigurationFunction(BlazeDirectories directories) {
    this.directories = directories;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException,
      SkyFunctionException {
    BuildConfigurationValue.Key key = (BuildConfigurationValue.Key) skyKey.argument();
    Set<Fragment> fragments;
    try {
      fragments = getConfigurationFragments(key, env);
    } catch (InvalidConfigurationException e) {
      throw new BuildConfigurationFunctionException(e);
    }
    if (fragments == null) {
      return null;
    }

    ClassToInstanceMap<Fragment> fragmentsMap = MutableClassToInstanceMap.create();
    for (Fragment fragment : fragments) {
      fragmentsMap.put(fragment.getClass(), fragment);
    }

    return new BuildConfigurationValue(
        new BuildConfiguration(directories, fragmentsMap, key.getBuildOptions(),
            !key.actionsEnabled()));
  }

  private Set<Fragment> getConfigurationFragments(BuildConfigurationValue.Key key, Environment env)
      throws InvalidConfigurationException {
    // Get SkyKeys for the fragments we need to load.
    Set<SkyKey> fragmentKeys = new LinkedHashSet<>();
    for (Class<? extends BuildConfiguration.Fragment> fragmentClass : key.getFragments()) {
      fragmentKeys.add(ConfigurationFragmentValue.key(key.getBuildOptions(), fragmentClass));
    }

    // Load them as Skyframe deps.
    Map<SkyKey, ValueOrException<InvalidConfigurationException>> fragmentDeps =
        env.getValuesOrThrow(fragmentKeys, InvalidConfigurationException.class);
    if (env.valuesMissing()) {
      return null;
    }

    // Collect and return the results.
    ImmutableSet.Builder<Fragment> fragments = ImmutableSet.builder();
    for (ValueOrException<InvalidConfigurationException> value : fragmentDeps.values()) {
      fragments.add(((ConfigurationFragmentValue) value.get()).getFragment());
    }
    return fragments.build();
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  private static final class BuildConfigurationFunctionException extends SkyFunctionException {
    public BuildConfigurationFunctionException(InvalidConfigurationException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
