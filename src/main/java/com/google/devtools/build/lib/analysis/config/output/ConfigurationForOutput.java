// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.config.output;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableSortedMap.toImmutableSortedMap;
import static com.google.common.collect.ImmutableSortedSet.toImmutableSortedSet;
import static java.util.Comparator.comparing;
import static java.util.stream.Collectors.toList;

import com.google.common.base.Verify;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentClassSet;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SortedSet;
import javax.annotation.Nullable;

/**
 * Data structure defining a {@link BuildConfigurationValue} for the purpose of returning user
 * output about the configuration.
 *
 * <p>Includes all data representing a "configuration" and defines their relative structure and list
 * order.
 *
 * <p>A {@link com.google.devtools.build.lib.runtime.commands.ConfigCommandOutputFormatter} uses
 * this to lightly format output from a logically consistent core structure.
 */
public class ConfigurationForOutput {
  private final String skyKey;
  private final String configHash;
  private final String mnemonic;
  private final boolean isExec;
  private final List<FragmentForOutput> fragments;
  private final List<FragmentOptionsForOutput> fragmentOptions;

  public ConfigurationForOutput(
      String skyKey,
      String configHash,
      String mnemonic,
      boolean isExec,
      List<FragmentForOutput> fragments,
      List<FragmentOptionsForOutput> fragmentOptions) {
    this.skyKey = skyKey;
    this.configHash = configHash;
    this.mnemonic = mnemonic;
    this.isExec = isExec;
    this.fragments = fragments;
    this.fragmentOptions = fragmentOptions;
  }

  public String getSkyKey() {
    return skyKey;
  }

  public String getConfigHash() {
    return configHash;
  }

  public String getMnemonic() {
    return mnemonic;
  }

  public boolean isExec() {
    return isExec;
  }

  public List<FragmentForOutput> getFragments() {
    return fragments;
  }

  /**
   * The union of {@link FragmentOptionsForOutput} used by the Fragments associated with this
   * configuration, sorted by FragmentOptionsForOutput name.
   */
  public List<FragmentOptionsForOutput> getFragmentOptions() {
    return fragmentOptions;
  }

  @Nullable
  public FragmentOptionsForOutput fragment(String fragmentName) {
    return this.fragmentOptions.stream()
        .filter(fo -> fo.getName().equals(fragmentName))
        .findFirst()
        .orElse(null);
  }

  public SortedSet<String> fragmentOptionNames() {
    return this.fragmentOptions.stream()
        .map(FragmentOptionsForOutput::getName)
        .collect(toImmutableSortedSet(Ordering.natural()));
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof ConfigurationForOutput) {
      ConfigurationForOutput other = (ConfigurationForOutput) o;
      return other.skyKey.equals(skyKey)
          && other.configHash.equals(configHash)
          && other.fragments.equals(fragments)
          && other.fragmentOptions.equals(fragmentOptions);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return Objects.hash(skyKey, configHash, fragments, fragmentOptions);
  }

  /** Constructs a {@link ConfigurationForOutput} from the given {@link BuildConfigurationValue}. */
  public static ConfigurationForOutput getConfigurationForOutput(
      BuildConfigurationValue buildConfigurationValue) {
    ImmutableSortedMap<
            Class<? extends Fragment>, ImmutableSortedSet<Class<? extends FragmentOptions>>>
        fragmentDefs =
            buildConfigurationValue.getFragments().keySet().stream()
                .collect(
                    toImmutableSortedMap(
                        FragmentClassSet.LEXICAL_FRAGMENT_SORTER,
                        fragment -> fragment,
                        fragment ->
                            ImmutableSortedSet.copyOf(
                                comparing(Class::getName), Fragment.requiredOptions(fragment))));

    return getConfigurationForOutput(
        buildConfigurationValue.getKey(),
        buildConfigurationValue.checksum(),
        buildConfigurationValue,
        fragmentDefs);
  }

  /** Constructs a {@link ConfigurationForOutput} from the given input data. */
  public static ConfigurationForOutput getConfigurationForOutput(
      BuildConfigurationKey skyKey,
      String configHash,
      BuildConfigurationValue config,
      ImmutableSortedMap<
              Class<? extends Fragment>, ImmutableSortedSet<Class<? extends FragmentOptions>>>
          fragmentDefs) {

    ImmutableSortedSet.Builder<FragmentForOutput> fragments =
        ImmutableSortedSet.orderedBy(comparing(e -> e.getName()));
    for (Map.Entry<Class<? extends Fragment>, ImmutableSortedSet<Class<? extends FragmentOptions>>>
        entry : fragmentDefs.entrySet()) {
      fragments.add(
          new FragmentForOutput(
              entry.getKey().getName(),
              entry.getValue().stream().map(Class::getName).collect(toImmutableList())));
    }
    fragmentDefs.entrySet().stream()
        .filter(entry -> config.hasFragment(entry.getKey()))
        .forEach(
            entry ->
                fragments.add(
                    new FragmentForOutput(
                        entry.getKey().getName(),
                        entry.getValue().stream().map(Class::getName).collect(toList()))));

    ImmutableSortedSet.Builder<FragmentOptionsForOutput> fragmentOptions =
        ImmutableSortedSet.orderedBy(comparing(e -> e.getName()));
    config.getOptions().getFragmentClasses().stream()
        .map(optionsClass -> config.getOptions().get(optionsClass))
        .forEach(
            fragmentOptionsInstance ->
                fragmentOptions.add(
                    new FragmentOptionsForOutput(
                        fragmentOptionsInstance.getClass().getName(),
                        getOrderedNativeOptions(fragmentOptionsInstance))));
    fragmentOptions.add(
        new FragmentOptionsForOutput(
            UserDefinedFragment.DESCRIPTIVE_NAME, getOrderedUserDefinedOptions(config)));

    return new ConfigurationForOutput(
        skyKey.toString(),
        configHash,
        config.getMnemonic(),
        config.isExecConfiguration(),
        fragments.build().asList(),
        fragmentOptions.build().asList());
  }

  /**
   * Returns a {@link FragmentOptions}'s native option settings in canonical order.
   *
   * <p>While actual option values are objects, we serialize them to strings to prevent command
   * output from interpreting them more deeply than we want for simple "name=value" output.
   */
  private static ImmutableSortedMap<String, String> getOrderedNativeOptions(
      FragmentOptions options) {
    return options.asMap().entrySet().stream()
        // While technically part of CoreOptions, --define is practically a user-definable flag so
        // we include it in the user-defined fragment for clarity. See getOrderedUserDefinedOptions.
        .filter(
            entry ->
                !(options.getClass().equals(CoreOptions.class) && entry.getKey().equals("define")))
        .collect(
            toImmutableSortedMap(
                Ordering.natural(), Map.Entry::getKey, e -> String.valueOf(e.getValue())));
  }

  /**
   * Returns a configuration's user-definable settings in canonical order.
   *
   * <p>While actual option values are objects, we serialize them to strings to prevent command
   * output from interpreting them more deeply than we want for simple "name=value" output.
   */
  private static ImmutableSortedMap<String, String> getOrderedUserDefinedOptions(
      BuildConfigurationValue config) {
    ImmutableSortedMap.Builder<String, String> ans = ImmutableSortedMap.naturalOrder();

    // Starlark-defined options:
    for (Map.Entry<Label, Object> entry : config.getOptions().getStarlarkOptions().entrySet()) {
      ans.put(entry.getKey().toString(), String.valueOf(entry.getValue()));
    }

    // --define:
    for (Map.Entry<String, String> entry :
        config
            .getOptions()
            .get(CoreOptions.class)
            .getNormalizedCommandLineBuildVariables()
            .entrySet()) {
      ans.put("--define:" + entry.getKey(), Verify.verifyNotNull(entry.getValue()));
    }
    return ans.buildOrThrow();
  }

  /**
   * Starlark options don't have configuration fragments. This is just to keep their output
   * consistent with native options, i.e. to include "user-defined" section in the output list.
   */
  static class UserDefinedFragment extends FragmentOptions {
    static final String DESCRIPTIVE_NAME = "user-defined";
    // Intentionally empty: we read the actual options directly from BuildOptions.
  }
}
