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

package com.google.devtools.build.lib.analysis.config;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.packages.Attribute.SplitTransition;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsClassProvider;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * This is a collection of command-line options from all configuration fragments. Contains
 * a single instance for all FragmentOptions classes provided by Blaze language modules.
 */
public final class BuildOptions implements Cloneable, Serializable {
  /**
   * Creates a BuildOptions object with all options set to its default value.
   */
  public static BuildOptions createDefaults(Iterable<Class<? extends FragmentOptions>> options) {
    Builder builder = builder();
    for (Class<? extends FragmentOptions> optionsClass : options) {
      builder.add(Options.getDefaults(optionsClass));
    }
    return builder.build();
  }

  /**
   * This function creates a new BuildOptions instance for host.
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
   * Creates an BuildOptions class by taking the option values from an options provider
   * (eg. an OptionsParser).
   */
  public static BuildOptions of(List<Class<? extends FragmentOptions>> optionsList,
      OptionsClassProvider provider) {
    Builder builder = builder();
    for (Class<? extends FragmentOptions> optionsClass : optionsList) {
      builder.add(provider.getOptions(optionsClass));
    }
    return builder.build();
  }

  /**
   * Creates an BuildOptions class by taking the option values from command-line arguments
   */
  @VisibleForTesting
  public static BuildOptions of(List<Class<? extends FragmentOptions>> optionsList, String... args)
      throws OptionsParsingException {
    Builder builder = builder();
    OptionsParser parser = OptionsParser.newOptionsParser(
        ImmutableList.<Class<? extends OptionsBase>>copyOf(optionsList));
    parser.parse(args);
    for (Class<? extends FragmentOptions> optionsClass : optionsList) {
      builder.add(parser.getOptions(optionsClass));
    }
    return builder.build();
  }

  /**
   * Returns the actual instance of a FragmentOptions class.
   */
  @SuppressWarnings("unchecked")
  public <T extends FragmentOptions> T get(Class<T> optionsClass) {
    FragmentOptions options = fragmentOptionsMap.get(optionsClass);
    Preconditions.checkNotNull(options);
    Preconditions.checkArgument(optionsClass.isAssignableFrom(options.getClass()));
    return (T) options;
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
  public Iterable<FragmentOptions> getOptions() {
    return fragmentOptionsMap.values();
  }

  /**
   * Creates a copy of the BuildOptions object that contains copies of the FragmentOptions.
   */
  @Override
  public BuildOptions clone() {
    ImmutableMap.Builder<Class<? extends FragmentOptions>, FragmentOptions> builder =
        ImmutableMap.builder();
    for (Map.Entry<Class<? extends FragmentOptions>, FragmentOptions> entry :
        fragmentOptionsMap.entrySet()) {
      builder.put(entry.getKey(), entry.getValue().clone());
    }
    return new BuildOptions(builder.build());
  }

  @Override
  public boolean equals(Object other) {
    return (this == other) || (other instanceof BuildOptions &&
        fragmentOptionsMap.equals(((BuildOptions) other).fragmentOptionsMap));
  }

  @Override
  public int hashCode() {
    return fragmentOptionsMap.hashCode();
  }

  /**
   * Maps options class definitions to FragmentOptions objects
   */
  private final ImmutableMap<Class<? extends FragmentOptions>, FragmentOptions> fragmentOptionsMap;

  private BuildOptions(
      ImmutableMap<Class<? extends FragmentOptions>, FragmentOptions> fragmentOptionsMap) {
    this.fragmentOptionsMap = fragmentOptionsMap;
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

    public BuildOptions build() {
      return new BuildOptions(ImmutableMap.copyOf(builderMap));
    }

    private Map<Class<? extends FragmentOptions>, FragmentOptions> builderMap;

    private Builder() {
      builderMap = new HashMap<>();
    }
  }
}
