// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentClassSet;
import com.google.devtools.build.lib.analysis.config.FragmentFactory;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.OutputDirectories.InvalidMnemonicException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import net.starlark.java.eval.StarlarkSemantics;

/** A builder for {@link BuildConfigurationValue} instances. */
public final class BuildConfigurationFunction implements SkyFunction {

  private final BlazeDirectories directories;
  private final ConfiguredRuleClassProvider ruleClassProvider;
  private final FragmentFactory fragmentFactory = new FragmentFactory();

  public BuildConfigurationFunction(
      BlazeDirectories directories, RuleClassProvider ruleClassProvider) {
    this.directories = directories;
    this.ruleClassProvider = (ConfiguredRuleClassProvider) ruleClassProvider;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, BuildConfigurationFunctionException {
    WorkspaceNameValue workspaceNameValue = (WorkspaceNameValue) env
        .getValue(WorkspaceNameValue.key());
    if (workspaceNameValue == null) {
      return null;
    }
    FragmentClassSet fragmentClasses = ruleClassProvider.getFragmentRegistry().getAllFragments();

    BuildConfigurationKey key = (BuildConfigurationKey) skyKey.argument();
    ImmutableSortedMap<Class<? extends Fragment>, Fragment> fragments;
    try {
      fragments = getConfigurationFragments(key, fragmentClasses);
    } catch (InvalidConfigurationException e) {
      throw new BuildConfigurationFunctionException(e);
    }

    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (starlarkSemantics == null) {
      return null;
    }

    ActionEnvironment actionEnvironment =
        ruleClassProvider.getActionEnvironmentProvider().getActionEnvironment(key.getOptions());

    try {
      return new BuildConfigurationValue(
          directories,
          fragments,
          key.getOptions(),
          ruleClassProvider.getReservedActionMnemonics(),
          actionEnvironment,
          RepositoryName.createFromValidStrippedName(workspaceNameValue.getName()),
          starlarkSemantics.getBool(BuildLanguageOptions.EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT));
    } catch (InvalidMnemonicException e) {
      throw new BuildConfigurationFunctionException(e);
    }
  }

  private ImmutableSortedMap<Class<? extends Fragment>, Fragment> getConfigurationFragments(
      BuildConfigurationKey key, FragmentClassSet fragmentClasses)
      throws InvalidConfigurationException {
    ImmutableSortedMap.Builder<Class<? extends Fragment>, Fragment> fragments =
        ImmutableSortedMap.orderedBy(FragmentClassSet.LEXICAL_FRAGMENT_SORTER);
    for (Class<? extends Fragment> fragmentClass : fragmentClasses) {
      Fragment fragment = fragmentFactory.createFragment(key.getOptions(), fragmentClass);
      if (fragment != null) {
        fragments.put(fragmentClass, fragment);
      }
    }
    return fragments.build();
  }

  private static final class BuildConfigurationFunctionException extends SkyFunctionException {
    BuildConfigurationFunctionException(InvalidConfigurationException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
