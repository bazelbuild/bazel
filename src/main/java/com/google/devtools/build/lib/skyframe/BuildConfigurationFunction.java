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

import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.google.auto.value.AutoValue;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentClassSet;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.OutputDirectories.InvalidMnemonicException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.lang.reflect.InvocationTargetException;
import java.util.Set;
import java.util.concurrent.CompletionException;
import net.starlark.java.eval.StarlarkSemantics;

/** A builder for {@link BuildConfigurationValue} instances. */
public final class BuildConfigurationFunction implements SkyFunction {

  /** Cache with weak values can't have null values. */
  private static final Fragment NULL_MARKER = new Fragment() {};

  private final BlazeDirectories directories;
  private final ConfiguredRuleClassProvider ruleClassProvider;
  private final LoadingCache<FragmentKey, Fragment> fragmentCache =
      Caffeine.newBuilder().weakValues().build(BuildConfigurationFunction::makeFragment);

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
      BuildOptions trimmedOptions = trimToRequiredOptions(key.getOptions(), fragmentClass);
      Fragment fragment;
      FragmentKey fragmentKey = FragmentKey.create(trimmedOptions, fragmentClass);
      try {
        fragment = fragmentCache.get(fragmentKey);
      } catch (CompletionException e) {
        Throwables.propagateIfPossible(e.getCause(), InvalidConfigurationException.class);
        throw e;
      }
      if (fragment != NULL_MARKER) {
        fragments.put(fragmentClass, fragment);
      } else {
        // NULL_MARKER is never GC'ed, so this entry will stay in cache forever unless we delete it
        // ourselves. Since it's a cheap computation we don't care about recomputing it.
        fragmentCache.invalidate(fragmentKey);
      }
    }
    return fragments.build();
  }

  private static BuildOptions trimToRequiredOptions(
      BuildOptions original, Class<? extends Fragment> fragment) {
    BuildOptions.Builder trimmed = BuildOptions.builder();
    Set<Class<? extends FragmentOptions>> requiredOptions = Fragment.requiredOptions(fragment);
    for (FragmentOptions options : original.getNativeOptions()) {
      // CoreOptions is implicitly required by all fragments.
      if (options instanceof CoreOptions || requiredOptions.contains(options.getClass())) {
        trimmed.addFragmentOptions(options);
      }
    }
    if (Fragment.requiresStarlarkOptions(fragment)) {
      trimmed.addStarlarkOptions(original.getStarlarkOptions());
    }
    return trimmed.build();
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
