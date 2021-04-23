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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Lists;
import com.google.common.collect.MapDifference;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.common.collect.Ordering;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParsingResult;
import com.google.devtools.common.options.OptionsProvider;
import com.google.devtools.common.options.ParsedOptionDescription;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** Stores the command-line options from a set of configuration fragments. */
// TODO(janakr): If overhead of FragmentOptions class names is too high, add constructor that just
// takes fragments and gets names from them.
public final class BuildOptions implements Cloneable, Serializable {
  private static final Comparator<Class<? extends FragmentOptions>>
      lexicalFragmentOptionsComparator = Comparator.comparing(Class::getName);
  private static final Comparator<Label> starlarkOptionsComparator = Ordering.natural();

  public static Map<Label, Object> labelizeStarlarkOptions(Map<String, Object> starlarkOptions) {
    return starlarkOptions.entrySet().stream()
        .collect(
            Collectors.toMap(e -> Label.parseAbsoluteUnchecked(e.getKey()), Map.Entry::getValue));
  }

  public static BuildOptions getDefaultBuildOptionsForFragments(
      List<Class<? extends FragmentOptions>> fragmentClasses) {
    try {
      return BuildOptions.of(fragmentClasses);
    } catch (OptionsParsingException e) {
      throw new IllegalArgumentException("Failed to parse empty options", e);
    }
  }

  /** Creates a new BuildOptions instance for host. */
  public BuildOptions createHostOptions() {
    Builder builder = builder();
    for (FragmentOptions options : fragmentOptionsMap.values()) {
      builder.addFragmentOptions(options.getHost());
    }
    return builder.addStarlarkOptions(starlarkOptionsMap).build();
  }

  /**
   * Returns {@code BuildOptions} that are otherwise identical to this one, but contain only options
   * from the given {@link FragmentOptions} classes (plus build configuration options).
   *
   * <p>If nothing needs to be trimmed, this instance is returned.
   */
  public BuildOptions trim(Set<Class<? extends FragmentOptions>> optionsClasses) {
    List<FragmentOptions> retainedOptions =
        Lists.newArrayListWithExpectedSize(optionsClasses.size() + 1);
    for (FragmentOptions options : fragmentOptionsMap.values()) {
      if (optionsClasses.contains(options.getClass())
          // TODO(bazel-team): make this non-hacky while not requiring CoreOptions access
          // to BuildOptions.
          || options.getClass().getName().endsWith("CoreOptions")) {
        retainedOptions.add(options);
      }
    }
    if (retainedOptions.size() == fragmentOptionsMap.size()) {
      return this; // Nothing to trim.
    }
    Builder builder = builder();
    for (FragmentOptions options : retainedOptions) {
      builder.addFragmentOptions(options);
    }
    return builder.addStarlarkOptions(starlarkOptionsMap).build();
  }

  /**
   * Creates a BuildOptions class by taking the option values from an options provider (eg. an
   * OptionsParser).
   */
  public static BuildOptions of(
      Iterable<Class<? extends FragmentOptions>> optionsList, OptionsProvider provider) {
    Builder builder = builder();
    for (Class<? extends FragmentOptions> optionsClass : optionsList) {
      builder.addFragmentOptions(provider.getOptions(optionsClass));
    }
    return builder
        .addStarlarkOptions(labelizeStarlarkOptions(provider.getStarlarkOptions()))
        .build();
  }

  /**
   * Creates a BuildOptions class by taking the option values from command-line arguments. Returns a
   * BuildOptions class that only has native options.
   */
  @VisibleForTesting
  public static BuildOptions of(List<Class<? extends FragmentOptions>> optionsList, String... args)
      throws OptionsParsingException {
    Builder builder = builder();
    OptionsParser parser =
        OptionsParser.builder().optionsClasses(ImmutableList.copyOf(optionsList)).build();
    parser.parse(args);
    for (Class<? extends FragmentOptions> optionsClass : optionsList) {
      builder.addFragmentOptions(parser.getOptions(optionsClass));
    }
    return builder.build();
  }

  /*
   * Returns a BuildOptions class that only has Starlark options.
   */
  @VisibleForTesting
  public static BuildOptions of(Map<Label, Object> starlarkOptions) {
    return builder().addStarlarkOptions(starlarkOptions).build();
  }

  /** Returns the actual instance of a FragmentOptions class. */
  public <T extends FragmentOptions> T get(Class<T> optionsClass) {
    FragmentOptions options = fragmentOptionsMap.get(optionsClass);
    checkNotNull(options, "fragment options unavailable: %s", optionsClass);
    return optionsClass.cast(options);
  }

  /** Returns true if these options contain the given {@link FragmentOptions}. */
  public boolean contains(Class<? extends FragmentOptions> optionsClass) {
    return fragmentOptionsMap.containsKey(optionsClass);
  }

  /** Returns a hex digest string uniquely identifying the build options. */
  public String checksum() {
    if (checksum == null) {
      synchronized (this) {
        if (checksum == null) {
          Fingerprint fingerprint = new Fingerprint();
          for (FragmentOptions options : fragmentOptionsMap.values()) {
            fingerprint.addString(options.cacheKey());
          }
          fingerprint.addString(OptionsBase.mapToCacheKey(starlarkOptionsMap));
          checksum = fingerprint.hexDigestAndReset();
        }
      }
    }
    return checksum;
  }

  /** String representation of build options. */
  @Override
  public String toString() {
    StringBuilder stringBuilder = new StringBuilder();
    for (FragmentOptions options : fragmentOptionsMap.values()) {
      stringBuilder.append(options);
    }
    return stringBuilder.toString();
  }

  /** Returns the options contained in this collection. */
  public ImmutableCollection<FragmentOptions> getNativeOptions() {
    return fragmentOptionsMap.values();
  }

  /** Returns the set of fragment classes contained in these options. */
  public ImmutableSet<Class<? extends FragmentOptions>> getFragmentClasses() {
    return fragmentOptionsMap.keySet();
  }

  public ImmutableMap<Label, Object> getStarlarkOptions() {
    return starlarkOptionsMap;
  }

  /**
   * Creates a copy of the BuildOptions object that contains copies of the FragmentOptions and
   * Starlark options.
   */
  @Override
  public BuildOptions clone() {
    ImmutableMap.Builder<Class<? extends FragmentOptions>, FragmentOptions> nativeOptionsBuilder =
        ImmutableMap.builderWithExpectedSize(fragmentOptionsMap.size());
    for (Map.Entry<Class<? extends FragmentOptions>, FragmentOptions> entry :
        fragmentOptionsMap.entrySet()) {
      nativeOptionsBuilder.put(entry.getKey(), entry.getValue().clone());
    }
    return new BuildOptions(nativeOptionsBuilder.build(), ImmutableMap.copyOf(starlarkOptionsMap));
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof BuildOptions)) {
      return false;
    }
    return checksum().equals(((BuildOptions) other).checksum());
  }

  @Override
  public int hashCode() {
    return 31 + checksum().hashCode();
  }

  /** Maps options class definitions to FragmentOptions objects. */
  private final ImmutableMap<Class<? extends FragmentOptions>, FragmentOptions> fragmentOptionsMap;
  /** Maps Starlark options names to Starlark options values. */
  private final ImmutableMap<Label, Object> starlarkOptionsMap;

  // Lazily initialized both for performance and correctness - BuildOptions instances may be mutated
  // after construction but before consumption. Access via checksum() to ensure initialization. This
  // field is volatile as per https://errorprone.info/bugpattern/DoubleCheckedLocking, which
  // encourages using volatile even for immutable objects.
  @Nullable private volatile String checksum = null;

  private BuildOptions(
      ImmutableMap<Class<? extends FragmentOptions>, FragmentOptions> fragmentOptionsMap,
      ImmutableMap<Label, Object> starlarkOptionsMap) {
    this.fragmentOptionsMap = fragmentOptionsMap;
    this.starlarkOptionsMap = starlarkOptionsMap;
  }

  /**
   * Applies any options set in the parsing result on top of these options, returning the resulting
   * build options.
   *
   * <p>To preserve fragment trimming, this method will not expand the set of included native
   * fragments. If the parsing result contains native options whose owning fragment is not part of
   * these options they will be ignored (i.e. not set on the resulting options). Starlark options
   * are not affected by this restriction.
   *
   * @param parsingResult any options that are being modified
   * @return the new options after applying the parsing result to the original options
   */
  public BuildOptions applyParsingResult(OptionsParsingResult parsingResult) {
    Map<Class<? extends FragmentOptions>, FragmentOptions> modifiedFragments =
        toModifiedFragments(parsingResult);

    BuildOptions.Builder builder = builder();
    for (Map.Entry<Class<? extends FragmentOptions>, FragmentOptions> classAndFragment :
        fragmentOptionsMap.entrySet()) {
      Class<? extends FragmentOptions> fragmentClass = classAndFragment.getKey();
      if (modifiedFragments.containsKey(fragmentClass)) {
        builder.addFragmentOptions(modifiedFragments.get(fragmentClass));
      } else {
        builder.addFragmentOptions(classAndFragment.getValue());
      }
    }

    Map<Label, Object> starlarkOptions = new HashMap<>(starlarkOptionsMap);
    Map<Label, Object> parsedStarlarkOptions =
        labelizeStarlarkOptions(parsingResult.getStarlarkOptions());
    for (Map.Entry<Label, Object> starlarkOption : parsedStarlarkOptions.entrySet()) {
      starlarkOptions.put(starlarkOption.getKey(), starlarkOption.getValue());
    }
    builder.addStarlarkOptions(starlarkOptions);
    return builder.build();
  }

  private Map<Class<? extends FragmentOptions>, FragmentOptions> toModifiedFragments(
      OptionsParsingResult parsingResult) {
    Map<Class<? extends FragmentOptions>, FragmentOptions> replacedOptions = new HashMap<>();
    for (ParsedOptionDescription parsedOption : parsingResult.asListOfExplicitOptions()) {
      OptionDefinition optionDefinition = parsedOption.getOptionDefinition();

      // All options obtained from an options parser are guaranteed to have been defined in an
      // FragmentOptions class.
      @SuppressWarnings("unchecked")
      Class<? extends FragmentOptions> fragmentOptionClass =
          (Class<? extends FragmentOptions>) optionDefinition.getField().getDeclaringClass();

      FragmentOptions originalFragment = fragmentOptionsMap.get(fragmentOptionClass);
      if (originalFragment == null) {
        // Preserve trimming by ignoring fragments not present in the original options.
        continue;
      }
      FragmentOptions newOptions =
          replacedOptions.computeIfAbsent(fragmentOptionClass, unused -> originalFragment.clone());
      try {
        Object value =
            parsingResult.getOptionValueDescription(optionDefinition.getOptionName()).getValue();
        optionDefinition.getField().set(newOptions, value);
      } catch (IllegalAccessException e) {
        throw new IllegalStateException("Couldn't set " + optionDefinition.getField(), e);
      }
    }

    return replacedOptions;
  }

  /**
   * Returns true if the passed parsing result's options have the same value as these options.
   *
   * <p>If a native parsed option is passed whose fragment has been trimmed in these options it is
   * considered to match.
   *
   * <p>If no options are present in the parsing result or all options in the parsing result have
   * been trimmed the result is considered not to match. This is because otherwise the parsing
   * result would match any options in a similar trimmed state, regardless of contents.
   *
   * @param parsingResult parsing result to be compared to these options
   * @return true if all non-trimmed values match
   * @throws OptionsParsingException if options cannot be parsed
   */
  public boolean matches(OptionsParsingResult parsingResult) throws OptionsParsingException {
    Set<OptionDefinition> ignoredDefinitions = new HashSet<>();
    for (ParsedOptionDescription parsedOption : parsingResult.asListOfExplicitOptions()) {
      OptionDefinition optionDefinition = parsedOption.getOptionDefinition();

      // All options obtained from an options parser are guaranteed to have been defined in an
      // FragmentOptions class.
      @SuppressWarnings("unchecked")
      Class<? extends FragmentOptions> fragmentClass =
          (Class<? extends FragmentOptions>) optionDefinition.getField().getDeclaringClass();

      FragmentOptions originalFragment = fragmentOptionsMap.get(fragmentClass);
      if (originalFragment == null) {
        // Ignore flags set in trimmed fragments.
        ignoredDefinitions.add(optionDefinition);
        continue;
      }
      Object originalValue = originalFragment.asMap().get(optionDefinition.getOptionName());
      if (!Objects.equals(originalValue, parsedOption.getConvertedValue())) {
        return false;
      }
    }

    Map<Label, Object> starlarkOptions =
        labelizeStarlarkOptions(parsingResult.getStarlarkOptions());
    MapDifference<Label, Object> starlarkDifference =
        Maps.difference(starlarkOptionsMap, starlarkOptions);
    if (starlarkDifference.entriesInCommon().size() < starlarkOptions.size()) {
      return false;
    }

    if (ignoredDefinitions.size() == parsingResult.asListOfExplicitOptions().size()
        && starlarkOptions.isEmpty()) {
      // Zero options were compared, either because none were passed or because all of them were
      // trimmed.
      return false;
    }

    return true;
  }

  /** Creates a builder object for BuildOptions */
  public static Builder builder() {
    return new Builder();
  }

  /** Creates a builder operating on a clone of this BuildOptions. */
  public Builder toBuilder() {
    return builder().merge(clone());
  }

  /** Builder class for BuildOptions. */
  public static class Builder {
    /**
     * Merges the given BuildOptions into this builder, overriding any previous instances of
     * Starlark options or FragmentOptions subclasses found in the new BuildOptions.
     */
    public Builder merge(BuildOptions options) {
      for (FragmentOptions fragment : options.getNativeOptions()) {
        this.addFragmentOptions(fragment);
      }
      this.addStarlarkOptions(options.getStarlarkOptions());
      return this;
    }

    /**
     * Adds a new {@link FragmentOptions} instance to the builder.
     *
     * <p>Overrides previous instances of the exact same subclass of {@code FragmentOptions}.
     *
     * <p>The options get preprocessed with {@link FragmentOptions#getNormalized}.
     */
    public <T extends FragmentOptions> Builder addFragmentOptions(T options) {
      fragmentOptions.put(options.getClass(), options.getNormalized());
      return this;
    }

    /**
     * Adds multiple Starlark options to the builder. Overrides previous instances of the same key.
     */
    public Builder addStarlarkOptions(Map<Label, Object> options) {
      starlarkOptions.putAll(options);
      return this;
    }

    /** Adds a Starlark option to the builder. Overrides previous instances of the same key. */
    public Builder addStarlarkOption(Label key, Object value) {
      starlarkOptions.put(key, value);
      return this;
    }

    /** Removes the value for the {@link FragmentOptions} with the given FragmentOptions class. */
    public Builder removeFragmentOptions(Class<? extends FragmentOptions> key) {
      fragmentOptions.remove(key);
      return this;
    }

    /** Removes the value for the Starlark option with the given key. */
    public Builder removeStarlarkOption(Label key) {
      starlarkOptions.remove(key);
      return this;
    }

    public BuildOptions build() {
      return new BuildOptions(
          ImmutableSortedMap.copyOf(fragmentOptions, lexicalFragmentOptionsComparator),
          ImmutableSortedMap.copyOf(starlarkOptions, starlarkOptionsComparator));
    }

    private final Map<Class<? extends FragmentOptions>, FragmentOptions> fragmentOptions;
    private final Map<Label, Object> starlarkOptions;

    private Builder() {
      fragmentOptions = new HashMap<>();
      starlarkOptions = new HashMap<>();
    }
  }

  /** Returns the difference between two BuildOptions in a new {@link BuildOptions.OptionsDiff}. */
  public static OptionsDiff diff(BuildOptions first, BuildOptions second) {
    return diff(new OptionsDiff(), first, second);
  }

  /**
   * Returns the difference between two BuildOptions in a pre-existing {@link
   * BuildOptions.OptionsDiff}.
   *
   * <p>In a single pass through this method, the method can only compare a single "first" {@link
   * BuildOptions} and single "second" BuildOptions; but an OptionsDiff instance can store the diff
   * between a single "first" BuildOptions and multiple "second" BuildOptions. Being able to
   * maintain a single OptionsDiff over multiple calls to diff is useful for, for example,
   * aggregating the difference between a single BuildOptions and the results of applying a {@link
   * com.google.devtools.build.lib.analysis.config.transitions.SplitTransition}) to it.
   */
  @SuppressWarnings("ReferenceEquality") // See comment above == comparison.
  public static OptionsDiff diff(OptionsDiff diff, BuildOptions first, BuildOptions second) {
    checkArgument(
        !diff.hasStarlarkOptions,
        "OptionsDiff cannot handle multiple 'second' BuildOptions with Starlark options and is"
            + " trying to diff against %s",
        diff);
    checkNotNull(first);
    checkNotNull(second);
    if (first.equals(second)) {
      return diff;
    }

    // Check and report if either class has been trimmed of an options class that exists in the
    // other.
    ImmutableSet<Class<? extends FragmentOptions>> firstOptionClasses =
        first.getNativeOptions().stream()
            .map(FragmentOptions::getClass)
            .collect(ImmutableSet.toImmutableSet());
    ImmutableSet<Class<? extends FragmentOptions>> secondOptionClasses =
        second.getNativeOptions().stream()
            .map(FragmentOptions::getClass)
            .collect(ImmutableSet.toImmutableSet());
    Sets.difference(firstOptionClasses, secondOptionClasses).forEach(diff::addExtraFirstFragment);
    Sets.difference(secondOptionClasses, firstOptionClasses).stream()
        .map(second::get)
        .forEach(diff::addExtraSecondFragment);

    // For fragments in common, report differences.
    for (Class<? extends FragmentOptions> clazz :
        Sets.intersection(firstOptionClasses, secondOptionClasses)) {
      FragmentOptions firstOptions = first.get(clazz);
      FragmentOptions secondOptions = second.get(clazz);
      // We avoid calling #equals because we are going to do a field-by-field comparison anyway.
      if (firstOptions == secondOptions) {
        continue;
      }
      for (OptionDefinition definition : OptionsParser.getOptionDefinitions(clazz)) {
        Object firstValue = firstOptions.getValueFromDefinition(definition);
        Object secondValue = secondOptions.getValueFromDefinition(definition);
        if (!Objects.equals(firstValue, secondValue)) {
          diff.addDiff(clazz, definition, firstValue, secondValue);
        }
      }
    }

    // Compare Starlark options for the two classes.
    Map<Label, Object> starlarkFirst = first.getStarlarkOptions();
    Map<Label, Object> starlarkSecond = second.getStarlarkOptions();
    for (Label buildSetting : Sets.union(starlarkFirst.keySet(), starlarkSecond.keySet())) {
      if (starlarkFirst.get(buildSetting) == null) {
        diff.addExtraSecondStarlarkOption(buildSetting, starlarkSecond.get(buildSetting));
      } else if (starlarkSecond.get(buildSetting) == null) {
        diff.addExtraFirstStarlarkOption(buildSetting);
      } else if (!starlarkFirst.get(buildSetting).equals(starlarkSecond.get(buildSetting))) {
        diff.putStarlarkDiff(
            buildSetting, starlarkFirst.get(buildSetting), starlarkSecond.get(buildSetting));
      }
    }
    return diff;
  }

  /**
   * A diff class for BuildOptions. Fields are meant to be populated and returned by {@link
   * BuildOptions#diff}.
   */
  public static final class OptionsDiff {
    private final Multimap<Class<? extends FragmentOptions>, OptionDefinition> differingOptions =
        ArrayListMultimap.create();
    // The keyset for the {@link first} and {@link second} maps are identical and indicate which
    // specific options differ between the first and second built options.
    private final Map<OptionDefinition, Object> first = new LinkedHashMap<>();
    // Since this class can be used to track the result of transitions, {@link second} is a multimap
    // to be able to handle {@link SplitTransition}s.
    private final Multimap<OptionDefinition, Object> second = OrderedSetMultimap.create();
    // List of "extra" fragments for each BuildOption aka fragments that were trimmed off one
    // BuildOption but not the other.
    private final Set<Class<? extends FragmentOptions>> extraFirstFragments = new HashSet<>();
    private final Set<FragmentOptions> extraSecondFragments = new HashSet<>();

    private final Map<Label, Object> starlarkFirst = new LinkedHashMap<>();
    // TODO(b/112041323): This should also be multimap but we don't diff multiple times with
    // Starlark options anywhere yet so add that feature when necessary.
    private final Map<Label, Object> starlarkSecond = new LinkedHashMap<>();

    private final List<Label> extraStarlarkOptionsFirst = new ArrayList<>();
    private final Map<Label, Object> extraStarlarkOptionsSecond = new HashMap<>();

    private boolean hasStarlarkOptions = false;

    @VisibleForTesting
    Set<Class<? extends FragmentOptions>> getExtraFirstFragmentClassesForTesting() {
      return extraFirstFragments;
    }

    @VisibleForTesting
    Set<FragmentOptions> getExtraSecondFragmentsForTesting() {
      return extraSecondFragments;
    }

    public Map<OptionDefinition, Object> getFirst() {
      return first;
    }

    public Multimap<OptionDefinition, Object> getSecond() {
      return second;
    }

    private void addDiff(
        Class<? extends FragmentOptions> fragmentOptionsClass,
        OptionDefinition option,
        Object firstValue,
        Object secondValue) {
      differingOptions.put(fragmentOptionsClass, option);
      first.put(option, firstValue);
      second.put(option, secondValue);
    }

    private void addExtraFirstFragment(Class<? extends FragmentOptions> options) {
      extraFirstFragments.add(options);
    }

    private void addExtraSecondFragment(FragmentOptions options) {
      extraSecondFragments.add(options);
    }

    private void putStarlarkDiff(Label buildSetting, Object firstValue, Object secondValue) {
      starlarkFirst.put(buildSetting, firstValue);
      starlarkSecond.put(buildSetting, secondValue);
      hasStarlarkOptions = true;
    }

    private void addExtraFirstStarlarkOption(Label buildSetting) {
      extraStarlarkOptionsFirst.add(buildSetting);
      hasStarlarkOptions = true;
    }

    private void addExtraSecondStarlarkOption(Label buildSetting, Object value) {
      extraStarlarkOptionsSecond.put(buildSetting, value);
      hasStarlarkOptions = true;
    }

    /**
     * Returns the labels of all starlark options that caused a difference between the first and
     * second options set.
     */
    public Set<Label> getChangedStarlarkOptions() {
      return ImmutableSet.<Label>builder()
          .addAll(starlarkFirst.keySet())
          .addAll(starlarkSecond.keySet())
          .addAll(extraStarlarkOptionsFirst)
          .addAll(extraStarlarkOptionsSecond.keySet())
          .build();
    }

    @VisibleForTesting
    Map<Label, Object> getStarlarkFirstForTesting() {
      return starlarkFirst;
    }

    @VisibleForTesting
    Map<Label, Object> getStarlarkSecondForTesting() {
      return starlarkSecond;
    }

    @VisibleForTesting
    List<Label> getExtraStarlarkOptionsFirstForTesting() {
      return extraStarlarkOptionsFirst;
    }

    @VisibleForTesting
    Map<Label, Object> getExtraStarlarkOptionsSecondForTesting() {
      return extraStarlarkOptionsSecond;
    }

    /**
     * Note: it's not enough for first and second to be empty, with trimming, they must also contain
     * the same options classes.
     */
    boolean areSame() {
      return first.isEmpty()
          && second.isEmpty()
          && extraSecondFragments.isEmpty()
          && extraFirstFragments.isEmpty()
          && differingOptions.isEmpty()
          && starlarkFirst.isEmpty()
          && starlarkSecond.isEmpty()
          && extraStarlarkOptionsFirst.isEmpty()
          && extraStarlarkOptionsSecond.isEmpty();
    }

    public String prettyPrint() {
      StringBuilder toReturn = new StringBuilder();
      for (String diff : getPrettyPrintList()) {
        toReturn.append(diff).append(System.lineSeparator());
      }
      return toReturn.toString();
    }

    public List<String> getPrettyPrintList() {
      List<String> toReturn = new ArrayList<>();
      first.forEach(
          (option, value) ->
              toReturn.add(option.getOptionName() + ":" + value + " -> " + second.get(option)));
      starlarkFirst.forEach(
          (option, value) -> toReturn.add(option + ":" + value + starlarkSecond.get(option)));
      return toReturn;
    }
  }

  @SuppressWarnings("unused") // Used reflectively.
  private static final class Codec implements ObjectCodec<BuildOptions> {

    @Override
    public Class<BuildOptions> getEncodedClass() {
      return BuildOptions.class;
    }

    @Override
    public void serialize(
        SerializationContext context, BuildOptions options, CodedOutputStream codedOut)
        throws IOException {
      context.getDependency(OptionsChecksumCache.class).prime(options);
      codedOut.writeStringNoTag(options.checksum());
    }

    @Override
    public BuildOptions deserialize(DeserializationContext context, CodedInputStream codedIn)
        throws IOException {
        String checksum = codedIn.readString();
      return checkNotNull(
          context.getDependency(OptionsChecksumCache.class).getOptions(checksum),
          "No options instance for %s",
          checksum);
    }
  }

  /**
   * Provides {@link BuildOptions} instances when requested via their {@linkplain
   * BuildOptions#checksum() checksum}.
   */
  public interface OptionsChecksumCache {

    /**
     * Called during deserialization to transform a checksum into a {@link BuildOptions} instance.
     */
    BuildOptions getOptions(String checksum);

    /**
     * Notifies the cache that it may be necessary to deserialize the given options diff's checksum.
     *
     * <p>Called each time an {@link BuildOptions} instance is serialized.
     */
    void prime(BuildOptions options);
  }

  /**
   * Simple {@link OptionsChecksumCache} backed by a {@link ConcurrentMap}.
   *
   * <p>Checksum mappings are retained indefinitely.
   */
  public static final class MapBackedChecksumCache implements OptionsChecksumCache {
    private final ConcurrentMap<String, BuildOptions> map = new ConcurrentHashMap<>();

    @Override
    public BuildOptions getOptions(String checksum) {
      return map.get(checksum);
    }

    @Override
    public void prime(BuildOptions options) {
      map.putIfAbsent(options.checksum(), options);
    }
  }
}
