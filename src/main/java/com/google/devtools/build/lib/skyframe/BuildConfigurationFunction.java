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

import com.google.auto.value.AutoValue;
import com.google.common.base.Throwables;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ClassToInstanceMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.MutableClassToInstanceMap;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.OutputDirectories.InvalidMnemonicException;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.lang.reflect.InvocationTargetException;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import net.starlark.java.eval.StarlarkSemantics;

/** A builder for {@link BuildConfigurationValue} instances. */
public final class BuildConfigurationFunction implements SkyFunction {

  /** Cache with weak values can't have null values. */
  private static final Fragment NULL_MARKER = new Fragment() {};

  private final BlazeDirectories directories;
  private final ConfiguredRuleClassProvider ruleClassProvider;
  private final LoadingCache<FragmentKey, Fragment> fragmentCache =
      CacheBuilder.newBuilder()
          .weakValues()
          .build(
              new CacheLoader<FragmentKey, Fragment>() {
                @Override
                public Fragment load(FragmentKey key) throws InvalidConfigurationException {
                  return makeFragment(key);
                }
              });

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

    BuildConfigurationValue.Key key = (BuildConfigurationValue.Key) skyKey.argument();
    Set<Fragment> fragments;
    try {
      fragments = getConfigurationFragments(key);
    } catch (InvalidConfigurationException e) {
      throw new BuildConfigurationFunctionException(e);
    }
    if (fragments == null) {
      return null;
    }

    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (starlarkSemantics == null) {
      return null;
    }

    ClassToInstanceMap<Fragment> fragmentsMap = MutableClassToInstanceMap.create();
    for (Fragment fragment : fragments) {
      fragmentsMap.put(fragment.getClass(), fragment);
    }

    ActionEnvironment actionEnvironment =
        ruleClassProvider.getActionEnvironmentProvider().getActionEnvironment(key.getOptions());

    try {
      return new BuildConfigurationValue(
          new BuildConfiguration(
              directories,
              fragmentsMap,
              key.getOptions(),
              ruleClassProvider.getReservedActionMnemonics(),
              actionEnvironment,
              workspaceNameValue.getName(),
              starlarkSemantics.getBool(
                  BuildLanguageOptions.EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT)));
    } catch (InvalidMnemonicException e) {
      throw new BuildConfigurationFunctionException(e);
    }
  }

  private Set<Fragment> getConfigurationFragments(BuildConfigurationValue.Key key)
      throws InvalidConfigurationException {
    ImmutableSortedSet<Class<? extends Fragment>> fragmentClasses = key.getFragments();
    ImmutableSet.Builder<Fragment> fragments =
        ImmutableSet.builderWithExpectedSize(fragmentClasses.size());
    for (Class<? extends Fragment> fragmentClass : fragmentClasses) {
      BuildOptions trimmedOptions =
          key.getOptions()
              .trim(
                  BuildConfiguration.getOptionsClasses(
                      ImmutableList.of(fragmentClass), ruleClassProvider));
      Fragment fragment;
      FragmentKey fragmentKey = FragmentKey.create(trimmedOptions, fragmentClass);
      try {
        fragment = fragmentCache.get(fragmentKey);
      } catch (ExecutionException e) {
        Throwables.propagateIfPossible(e.getCause(), InvalidConfigurationException.class);
        throw new IllegalStateException(e);
      }
      if (fragment != NULL_MARKER) {
        fragments.add(fragment);
      } else {
        // NULL_MARKER is never GC'ed, so this entry will stay in cache forever unless we delete it
        // ourselves. Since it's a cheap computation we don't care about recomputing it.
        fragmentCache.invalidate(fragmentKey);
      }
    }
    return fragments.build();
  }

  @AutoValue
  abstract static class FragmentKey {
    abstract BuildOptions getBuildOptions();

    abstract Class<? extends Fragment> getFragmentClass();

    private static FragmentKey create(
        BuildOptions buildOptions, Class<? extends Fragment> fragmentClass) {
      return new AutoValue_BuildConfigurationFunction_FragmentKey(buildOptions, fragmentClass);
    }
  }

  private static Fragment makeFragment(FragmentKey fragmentKey)
      throws InvalidConfigurationException {
    BuildOptions buildOptions = fragmentKey.getBuildOptions();
    Class<? extends Fragment> fragmentClass = fragmentKey.getFragmentClass();
    String noConstructorPattern = "%s lacks constructor(BuildOptions)";
    try {
      Fragment fragment =
          fragmentClass.getConstructor(BuildOptions.class).newInstance(buildOptions);
      return fragment.shouldInclude() ? fragment : NULL_MARKER;
    } catch (InvocationTargetException e) {
      if (e.getCause() instanceof InvalidConfigurationException) {
        throw (InvalidConfigurationException) e.getCause();
      }
      throw new IllegalStateException(String.format(noConstructorPattern, fragmentClass), e);
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException(String.format(noConstructorPattern, fragmentClass), e);
    }
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  private static final class BuildConfigurationFunctionException extends SkyFunctionException {
    BuildConfigurationFunctionException(InvalidConfigurationException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
