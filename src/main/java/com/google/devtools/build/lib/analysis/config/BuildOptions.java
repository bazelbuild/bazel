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

package com.google.devtools.build.lib.analysis.config;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Objects;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.flags.InvocationPolicyEnforcer;
import com.google.devtools.build.lib.packages.Attribute.SplitTransition;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsClassProvider;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import javax.annotation.Nullable;

/**
 * Stores the command-line options from a set of configuration fragments.
 */
public final class BuildOptions implements Cloneable, Serializable {
  /**
   * Creates a BuildOptions object with all options set to their default values, processed by the
   * given {@code invocationPolicy}.
   */
  static BuildOptions createDefaults(
      Iterable<Class<? extends FragmentOptions>> options, InvocationPolicy invocationPolicy) {
    return of(options, createDefaultParser(options, invocationPolicy));
  }

  private static OptionsParser createDefaultParser(
      Iterable<Class<? extends FragmentOptions>> options, InvocationPolicy invocationPolicy) {
    OptionsParser optionsParser = OptionsParser.newOptionsParser(options);
    try {
      new InvocationPolicyEnforcer(invocationPolicy).enforce(optionsParser);
    } catch (OptionsParsingException e) {
      throw new IllegalStateException(e);
    }
    return optionsParser;
  }

  /**
   * Creates a new BuildOptions instance for host.
   *
   * @param fallback if true, we have already tried the user specified hostCpu options
   *                 and it didn't work, so now we try the default options instead.
   */
  public BuildOptions createHostOptions(boolean fallback) {
    Builder builder = builder();
    for (FragmentOptions options : fragmentOptionsMap.values()) {
      builder.add(options.getHost(fallback));
    }
    return builder.build();
  }

  /**
   * Returns a list of potential split configuration transitions by calling {@link
   * FragmentOptions#getPotentialSplitTransitions} on all the fragments.
   */
  public List<SplitTransition<BuildOptions>> getPotentialSplitTransitions() {
    List<SplitTransition<BuildOptions>> result = new ArrayList<>();
    for (FragmentOptions options : fragmentOptionsMap.values()) {
      result.addAll(options.getPotentialSplitTransitions());
    }
    return result;
  }

  /**
   * Returns an equivalent instance to this one with only options from the given
   * {@link FragmentOptions} classes.
   */
  public BuildOptions trim(Set<Class<? extends FragmentOptions>> optionsClasses) {
    Builder builder = builder();
    for (FragmentOptions options : fragmentOptionsMap.values()) {
      if (optionsClasses.contains(options.getClass())
          || options instanceof BuildConfiguration.Options) {
        builder.add(options);
      }
    }
    return builder.build();
  }

  /**
   * Creates a BuildOptions class by taking the option values from an options provider
   * (eg. an OptionsParser).
   */
  public static BuildOptions of(
      Iterable<Class<? extends FragmentOptions>> optionsList, OptionsClassProvider provider) {
    Builder builder = builder();
    for (Class<? extends FragmentOptions> optionsClass : optionsList) {
      builder.add(provider.getOptions(optionsClass));
    }
    return builder.build();
  }

  /**
   * Creates a BuildOptions class by taking the option values from command-line arguments
   */
  @VisibleForTesting
  public static BuildOptions of(List<Class<? extends FragmentOptions>> optionsList, String... args)
      throws OptionsParsingException {
    return of(optionsList, null, args);
  }

  /**
   * Creates a BuildOptions class by taking the option values from command-line arguments and
   * applying the specified original options.
   */
  @VisibleForTesting
  static BuildOptions of(List<Class<?extends FragmentOptions>> optionsList,
      BuildOptions originalOptions, String... args) throws OptionsParsingException {
    Builder builder = builder();
    OptionsParser parser = OptionsParser.newOptionsParser(
        ImmutableList.<Class<? extends OptionsBase>>copyOf(optionsList));
    parser.parse(args);
    for (Class<? extends FragmentOptions> optionsClass : optionsList) {
      builder.add(parser.getOptions(optionsClass));
    }
    builder.setOriginalOptions(originalOptions);
    return builder.build();
  }

  /**
   * Returns a cloned instance that disables dynamic configurations if both
   * {@link BuildConfiguration.Options.DynamicConfigsMode} is {@code NOTRIM_PARTIAL} and
   * {@link #useStaticConfigurationsOverride()} is true. Otherwise it returns the input
   * instance unchanged.
   */
  public static BuildOptions applyStaticConfigOverride(BuildOptions buildOptions) {
    if (buildOptions.useStaticConfigurationsOverride()
        && buildOptions.get(BuildConfiguration.Options.class).useDynamicConfigurations
            == BuildConfiguration.Options.DynamicConfigsMode.NOTRIM_PARTIAL) {
      // It's not, generally speaking, safe to mutate BuildOptions instances when the original
      // reference might persist.
      buildOptions = buildOptions.clone();
      buildOptions.get(BuildConfiguration.Options.class).useDynamicConfigurations =
          BuildConfiguration.Options.DynamicConfigsMode.OFF;
    }
    return buildOptions;
  }

  /**
   * Returns the actual instance of a FragmentOptions class.
   */
  public <T extends FragmentOptions> T get(Class<T> optionsClass) {
    FragmentOptions options = fragmentOptionsMap.get(optionsClass);
    Preconditions.checkNotNull(options, "fragment options unavailable: " + optionsClass.getName());
    return optionsClass.cast(options);
  }

  /**
   * Returns true if these options contain the given {@link FragmentOptions}.
   */
  public boolean contains(Class<? extends FragmentOptions> optionsClass) {
    return fragmentOptionsMap.containsKey(optionsClass);
  }

  /**
   * Returns a multimap of all labels that were specified as options, keyed by the name to be
   * displayed to the user if something goes wrong. This should be the set of all labels
   * mentioned in explicit command line options that are not already covered by the
   * tools/defaults package (see the DefaultsPackage class), and nothing else.
   */
  public ListMultimap<String, Label> getAllLabels() {
    ListMultimap<String, Label> labels = ArrayListMultimap.create();
    for (FragmentOptions optionsBase : fragmentOptionsMap.values()) {
      optionsBase.addAllLabels(labels);
    }
    return labels;
  }

  // It would be very convenient to use a Multimap here, but we cannot do that because we need to
  // support defaults labels that have zero elements.
  ImmutableMap<String, ImmutableSet<Label>> getDefaultsLabels() {
    BuildConfiguration.Options opts = get(BuildConfiguration.Options.class);
    Map<String, Set<Label>> collector  = new TreeMap<>();
    for (FragmentOptions fragment : fragmentOptionsMap.values()) {
      for (Map.Entry<String, Set<Label>> entry : fragment.getDefaultsLabels(opts).entrySet()) {
        if (!collector.containsKey(entry.getKey())) {
          collector.put(entry.getKey(), new TreeSet<Label>());
        }
        collector.get(entry.getKey()).addAll(entry.getValue());
      }
    }

    ImmutableMap.Builder<String, ImmutableSet<Label>> result = new ImmutableMap.Builder<>();
    for (Map.Entry<String, Set<Label>> entry : collector.entrySet()) {
      result.put(entry.getKey(), ImmutableSet.copyOf(entry.getValue()));
    }

    return result.build();
  }

  ImmutableList<String> getDefaultsRules() {
    ImmutableList.Builder<String> result = ImmutableList.builder();
    for (FragmentOptions fragment : fragmentOptionsMap.values()) {
      result.addAll(fragment.getDefaultsRules());
    }

    return result.build();
  }

  /**
   * Returns {@code true} if static configurations should be used with
   * {@link BuildConfiguration.Options.DynamicConfigsMode.NOTRIM_PARTIAL}.
   */
  public boolean useStaticConfigurationsOverride() {
    for (FragmentOptions fragment : fragmentOptionsMap.values()) {
      if (fragment.useStaticConfigurationsOverride()) {
        return true;
      }
    }
    return false;
  }

  /**
   * The cache key for the options collection. Recomputes cache key every time it's called.
   */
  public String computeCacheKey() {
    StringBuilder keyBuilder = new StringBuilder();
    for (FragmentOptions options : fragmentOptionsMap.values()) {
      keyBuilder.append(options.cacheKey());
    }
    return keyBuilder.toString();
  }

  /**
   * String representation of build options.
   */
  @Override
  public String toString() {
    StringBuilder stringBuilder = new StringBuilder();
    for (FragmentOptions options : fragmentOptionsMap.values()) {
      stringBuilder.append(options);
    }
    return stringBuilder.toString();
  }

  /**
   * Returns the options contained in this collection.
   */
  public Collection<FragmentOptions> getOptions() {
    return fragmentOptionsMap.values();
  }

  /**
   * Creates a copy of the BuildOptions object that contains copies of the FragmentOptions.
   */
  @Override
  public BuildOptions clone() {
    return clone(null);
  }

  /**
   * Creates a copy of the BuildOptions object that stores a set of original options. This can
   * be used to power "reversion" of options changes.
   */
  public BuildOptions clone(@Nullable BuildOptions originalOptions) {
    ImmutableMap.Builder<Class<? extends FragmentOptions>, FragmentOptions> builder =
        ImmutableMap.builder();
    for (Map.Entry<Class<? extends FragmentOptions>, FragmentOptions> entry :
        fragmentOptionsMap.entrySet()) {
      builder.put(entry.getKey(), entry.getValue().clone());
    }
    // TODO(bazel-team): only store the diff between the current options and its original
    // options. This may be easier with immutable options.
    return new BuildOptions(builder.build(), originalOptions);
  }

  /**
   * Returns the original options these options were spawned from, or null if this info wasn't
   * recorded.
   */
  public BuildOptions getOriginal() {
    return originalOptions;
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    } else if (!(other instanceof BuildOptions)) {
      return false;
    } else {
      BuildOptions otherOptions = (BuildOptions) other;
      return fragmentOptionsMap.equals(otherOptions.fragmentOptionsMap)
          && Objects.equal(originalOptions, otherOptions.originalOptions);
    }
  }

  @Override
  public int hashCode() {
    return fragmentOptionsMap.hashCode();
  }

  /**
   * Maps options class definitions to FragmentOptions objects
   */
  private final ImmutableMap<Class<? extends FragmentOptions>, FragmentOptions> fragmentOptionsMap;

  /**
   * Records an original set of options these options came from. When set, this
   * provides the ability to "revert" options back to a previous form.
   */
  @Nullable private final BuildOptions originalOptions;

  private BuildOptions(
      ImmutableMap<Class<? extends FragmentOptions>, FragmentOptions> fragmentOptionsMap,
      BuildOptions originalOptions) {
    this.fragmentOptionsMap = fragmentOptionsMap;
    this.originalOptions = originalOptions;
  }

  /**
   * Creates a builder object for BuildOptions
   */
  public static Builder builder() {
    return new Builder();
  }

  /**
   * Builder class for BuildOptions.
   */
  public static class Builder {
    /**
     * Adds a new FragmentOptions instance to the builder. Overrides previous instances of the
     * exact same subclass of FragmentOptions.
     */
    public <T extends FragmentOptions> Builder add(T options) {
      builderMap.put(options.getClass(), options);
      return this;
    }

    /**
     * Specify the original options these options were branched from. This should only be used
     * when there's a desire to revert back to the old options, e.g. for a parent transition.
     */
    public Builder setOriginalOptions(BuildOptions originalOptions) {
      this.originalOptions = originalOptions;
      return this;
    }

    public BuildOptions build() {
      return new BuildOptions(ImmutableMap.copyOf(builderMap), originalOptions);
    }

    private Map<Class<? extends FragmentOptions>, FragmentOptions> builderMap;
    @Nullable private BuildOptions originalOptions;

    private Builder() {
      builderMap = new HashMap<>();
    }
  }
}
