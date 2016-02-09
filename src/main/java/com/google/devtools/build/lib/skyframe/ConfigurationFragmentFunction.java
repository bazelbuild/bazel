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
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.PackageProviderForConfigurations;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.ConfigurationFragmentValue.ConfigurationFragmentKey;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;

/**
 * A builder for {@link ConfigurationFragmentValue}s.
 */
public class ConfigurationFragmentFunction implements SkyFunction {

  private final Supplier<ImmutableList<ConfigurationFragmentFactory>> configurationFragments;
  private final RuleClassProvider ruleClassProvider;

  public ConfigurationFragmentFunction(
      Supplier<ImmutableList<ConfigurationFragmentFactory>> configurationFragments,
      RuleClassProvider ruleClassProvider) {
    this.configurationFragments = configurationFragments;
    this.ruleClassProvider = ruleClassProvider;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException,
      ConfigurationFragmentFunctionException {
    ConfigurationFragmentKey configurationFragmentKey = 
        (ConfigurationFragmentKey) skyKey.argument();
    BuildOptions buildOptions = configurationFragmentKey.getBuildOptions();
    ConfigurationFragmentFactory factory = getFactory(configurationFragmentKey.getFragmentType());
    try {
      PackageProviderForConfigurations packageProvider = 
          new SkyframePackageLoaderWithValueEnvironment(env, ruleClassProvider);
      ConfigurationEnvironment confEnv = new ConfigurationBuilderEnvironment(packageProvider);
      Fragment fragment = factory.create(confEnv, buildOptions);
      
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
   * A {@link ConfigurationEnvironment} implementation that can create dependencies on files.
   */
  private final class ConfigurationBuilderEnvironment implements ConfigurationEnvironment {
    private final PackageProviderForConfigurations packageProvider;

    ConfigurationBuilderEnvironment(PackageProviderForConfigurations packageProvider) {
      this.packageProvider = packageProvider;
    }

    @Override
    public EventHandler getEventHandler() {
      return packageProvider.getEventHandler();
    }

    @Override
    public Target getTarget(Label label) throws NoSuchPackageException, NoSuchTargetException {
      return packageProvider.getTarget(label);
    }

    @Override
    public Path getPath(Package pkg, String fileName) {
      Path result = pkg.getPackageDirectory().getRelative(fileName);
      try {
        packageProvider.addDependency(pkg, fileName);
      } catch (IOException | LabelSyntaxException e) {
        return null;
      }
      return result;
    }

    @Override
    public <T extends Fragment> T getFragment(BuildOptions buildOptions, Class<T> fragmentType) 
        throws InvalidConfigurationException {
      return packageProvider.getFragment(buildOptions, fragmentType);
    }

    @Override
    public BlazeDirectories getBlazeDirectories() {
      return packageProvider.getDirectories();
    }
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
