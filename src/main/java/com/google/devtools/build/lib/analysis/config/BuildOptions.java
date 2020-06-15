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
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.MapDifference;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import com.google.common.collect.Ordering;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.trimming.ConfigurationComparer;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParsingResult;
import com.google.devtools.common.options.OptionsProvider;
import com.google.devtools.common.options.ParsedOptionDescription;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.io.Serializable;
import java.lang.ref.SoftReference;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** Stores the command-line options from a set of configuration fragments. */
// TODO(janakr): If overhead of FragmentOptions class names is too high, add constructor that just
// takes fragments and gets names from them.
@AutoCodec
public final class BuildOptions implements Cloneable, Serializable {
  private static final Comparator<Class<? extends FragmentOptions>>
      lexicalFragmentOptionsComparator = Comparator.comparing(Class::getName);
  private static final Comparator<Label> starlarkOptionsComparator = Ordering.natural();
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  public static Map<Label, Object> labelizeStarlarkOptions(Map<String, Object> starlarkOptions) {
    return starlarkOptions.entrySet().stream()
        .collect(
            Collectors.toMap(e -> Label.parseAbsoluteUnchecked(e.getKey()), Map.Entry::getValue));
  }

  public static BuildOptions getDefaultBuildOptionsForFragments(
      List<Class<? extends FragmentOptions>> fragmentClasses) {
    ArrayList<String> collector = new ArrayList<>();
    try {
      String[] stringCollector = new String[collector.size()];
      return BuildOptions.of(fragmentClasses, collector.toArray(stringCollector));
    } catch (OptionsParsingException e) {
      throw new IllegalArgumentException("Failed to parse default options", e);
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
        OptionsParser.builder()
            .optionsClasses(ImmutableList.<Class<? extends OptionsBase>>copyOf(optionsList))
            .build();
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
    Preconditions.checkNotNull(options, "fragment options unavailable: " + optionsClass.getName());
    return optionsClass.cast(options);
  }

  /** Returns true if these options contain the given {@link FragmentOptions}. */
  public boolean contains(Class<? extends FragmentOptions> optionsClass) {
    return fragmentOptionsMap.containsKey(optionsClass);
  }

  /** The cache key for the options collection. Recomputes cache key every time it's called. */
  public String computeCacheKey() {
    StringBuilder keyBuilder = new StringBuilder();
    for (FragmentOptions options : fragmentOptionsMap.values()) {
      keyBuilder.append(options.cacheKey());
    }
    keyBuilder.append(
        OptionsBase.mapToCacheKey(
            starlarkOptionsMap.entrySet().stream()
                .collect(Collectors.toMap(e -> e.getKey().toString(), Map.Entry::getValue))));
    return keyBuilder.toString();
  }

  public String computeChecksum() {
    return Fingerprint.getHexDigest(computeCacheKey());
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
  public Collection<FragmentOptions> getNativeOptions() {
    return fragmentOptionsMap.values();
  }

  /** Returns the set of fragment classes contained in these options. */
  public Set<Class<? extends FragmentOptions>> getFragmentClasses() {
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
        ImmutableMap.builder();
    for (Map.Entry<Class<? extends FragmentOptions>, FragmentOptions> entry :
        fragmentOptionsMap.entrySet()) {
      nativeOptionsBuilder.put(entry.getKey(), entry.getValue().clone());
    }
    return new BuildOptions(nativeOptionsBuilder.build(), ImmutableMap.copyOf(starlarkOptionsMap));
  }

  private boolean fingerprintAndHashCodeInitialized() {
    return fingerprint != null;
  }

  /**
   * Lazily initialize {@link #fingerprint} and {@link #hashCode}. Keeps computation off critical
   * path of build, while still avoiding expensive computation for equality and hash code each time.
   *
   * <p>We check {@link #fingerprintAndHashCodeInitialized} to see if this method has already been
   * called. Using {@link #hashCode} after this method is called is safe because it is set here
   * before {@link #fingerprint} is set, so if {@link #fingerprint} is non-null then {@link
   * #hashCode} is definitely set.
   */
  private void maybeInitializeFingerprintAndHashCode() {
    if (fingerprintAndHashCodeInitialized()) {
      return;
    }
    synchronized (this) {
      if (fingerprintAndHashCodeInitialized()) {
        return;
      }
      Fingerprint fingerprint = new Fingerprint();
      for (Map.Entry<Class<? extends FragmentOptions>, FragmentOptions> entry :
          fragmentOptionsMap.entrySet()) {
        fingerprint.addString(entry.getKey().getName());
        fingerprint.addString(entry.getValue().cacheKey());
      }
      for (Map.Entry<Label, Object> entry : starlarkOptionsMap.entrySet()) {
        fingerprint.addString(entry.getKey().toString());
        fingerprint.addString(entry.getValue().toString());
      }
      byte[] computedFingerprint = fingerprint.digestAndReset();
      hashCode = Arrays.hashCode(computedFingerprint);
      this.fingerprint = computedFingerprint;
    }
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    } else if (!(other instanceof BuildOptions)) {
      return false;
    } else {
      maybeInitializeFingerprintAndHashCode();
      BuildOptions otherOptions = (BuildOptions) other;
      otherOptions.maybeInitializeFingerprintAndHashCode();
      return Arrays.equals(this.fingerprint, otherOptions.fingerprint);
    }
  }

  @Override
  public int hashCode() {
    maybeInitializeFingerprintAndHashCode();
    return hashCode;
  }

  // Lazily initialized.
  @Nullable private volatile byte[] fingerprint;
  private volatile int hashCode;

  /** Maps options class definitions to FragmentOptions objects. */
  private final ImmutableMap<Class<? extends FragmentOptions>, FragmentOptions> fragmentOptionsMap;
  /** Maps Starlark options names to Starlark options values. */
  private final ImmutableMap<Label, Object> starlarkOptionsMap;

  @AutoCodec.VisibleForSerialization
  BuildOptions(
      ImmutableMap<Class<? extends FragmentOptions>, FragmentOptions> fragmentOptionsMap,
      ImmutableMap<Label, Object> starlarkOptionsMap) {
    this.fragmentOptionsMap = fragmentOptionsMap;
    this.starlarkOptionsMap = starlarkOptionsMap;
  }

  public BuildOptions applyDiff(OptionsDiffForReconstruction optionsDiff) {
    if (optionsDiff.isEmpty()) {
      return this;
    }
    maybeInitializeFingerprintAndHashCode();
    if (!Arrays.equals(fingerprint, optionsDiff.baseFingerprint)) {
      throw new IllegalArgumentException("Cannot reconstruct BuildOptions with a different base.");
    }
    BuildOptions reconstructedOptions = optionsDiff.cachedReconstructed.get();
    if (reconstructedOptions != null) {
      return reconstructedOptions;
    }
    Builder builder = builder();
    for (FragmentOptions options : fragmentOptionsMap.values()) {
      FragmentOptions newOptions = optionsDiff.transformOptions(options);
      if (newOptions != null) {
        builder.addFragmentOptions(newOptions);
      }
    }
    for (FragmentOptions extraSecondFragment : optionsDiff.extraSecondFragments) {
      builder.addFragmentOptions(extraSecondFragment);
    }

    Map<Label, Object> starlarkOptions = new HashMap<>();
    for (Map.Entry<Label, Object> buildSettingAndValue : starlarkOptionsMap.entrySet()) {
      Label buildSetting = buildSettingAndValue.getKey();
      if (optionsDiff.extraFirstStarlarkOptions.contains(buildSetting)) {
        continue;
      } else if (optionsDiff.differingStarlarkOptions.containsKey(buildSetting)) {
        starlarkOptions.put(buildSetting, optionsDiff.differingStarlarkOptions.get(buildSetting));
      } else {
        starlarkOptions.put(buildSetting, starlarkOptionsMap.get(buildSetting));
      }
    }
    starlarkOptions.putAll(optionsDiff.extraSecondStarlarkOptions);
    reconstructedOptions = builder.addStarlarkOptions(starlarkOptions).build();
    optionsDiff.cachedReconstructed = new SoftReference<>(reconstructedOptions);
    return reconstructedOptions;
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
          replacedOptions.computeIfAbsent(
              fragmentOptionClass,
              (Class<? extends FragmentOptions> k) -> originalFragment.clone());
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

    /** Returns whether the builder contains a particular Starlark option. */
    boolean contains(Label key) {
      return starlarkOptions.containsKey(key);
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
  @SuppressWarnings("ReferenceEquality") // See comments above == comparisons.
  public static OptionsDiff diff(OptionsDiff diff, BuildOptions first, BuildOptions second) {
    if (diff.hasStarlarkOptions) {
      throw new IllegalStateException(
          "OptionsDiff cannot handle multiple 'second' BuildOptions with Starlark options "
              + "and is trying to diff against a second BuildOptions with Starlark options.");
    }
    if (first == null || second == null) {
      throw new IllegalArgumentException("Cannot diff null BuildOptions");
    }
    // For performance reasons, we avoid calling #equals unless both instances have had their
    // fingerprint and hash code initialized. We don't typically encounter value-equal instances
    // here anyway.
    if (first == second
        || (first.fingerprintAndHashCodeInitialized()
            && second.fingerprintAndHashCodeInitialized()
            && first.equals(second))) {
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
      // Similar to above, we avoid calling #equals because we are going to do a field-by-field
      // comparison anyway.
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
   * Cache for {@link OptionsDiffForReconstruction}, which is expensive to compute.
   *
   * <p>The reason for using {@linkplain CacheBuilder#weakKeys weak keys} is twofold: we want
   * objects in the cache to be garbage collected, and we also want to use reference equality to
   * avoid the expensive initialization in {@link #maybeInitializeFingerprintAndHashCode}.
   */
  private static final Cache<BuildOptions, OptionsDiffForReconstruction>
      diffForReconstructionCache = CacheBuilder.newBuilder().weakKeys().build();

  /**
   * Returns a {@link OptionsDiffForReconstruction} object that can be applied to {@code first} via
   * {@link #applyDiff} to get a {@link BuildOptions} object equal to {@code second}.
   */
  public static OptionsDiffForReconstruction diffForReconstruction(
      BuildOptions first, BuildOptions second) {
    OptionsDiffForReconstruction diff;
    try {
      diff =
          diffForReconstructionCache.get(second, () -> createDiffForReconstruction(first, second));
    } catch (ExecutionException e) {
      throw new IllegalStateException(e);
    }

    // We need to ensure that the possibly cached diff was computed against the same base options.
    // In practice this should always be the case, since callers pass in a "default" options
    // instance as "first". To be safe however, we create an uncached diff if there is a mismatch.
    // Note that this check should be fast because the fingerprints should be reference-equal.
    return Arrays.equals(first.fingerprint, diff.baseFingerprint)
        ? diff
        : createDiffForReconstruction(first, second);
  }

  private static OptionsDiffForReconstruction createDiffForReconstruction(
      BuildOptions first, BuildOptions second) {
    OptionsDiff diff = diff(first, second);
    if (diff.areSame()) {
      first.maybeInitializeFingerprintAndHashCode();
      return OptionsDiffForReconstruction.getEmpty(first.fingerprint, second.computeChecksum());
    }
    LinkedHashMap<Class<? extends FragmentOptions>, Map<String, Object>> differingOptions =
        new LinkedHashMap<>(diff.differingOptions.keySet().size());
    for (Class<? extends FragmentOptions> clazz :
        diff.differingOptions.keySet().stream()
            .sorted(lexicalFragmentOptionsComparator)
            .collect(Collectors.toList())) {
      Collection<OptionDefinition> fields = diff.differingOptions.get(clazz);
      LinkedHashMap<String, Object> valueMap = new LinkedHashMap<>(fields.size());
      for (OptionDefinition optionDefinition :
          fields.stream()
              .sorted(Comparator.comparing(o -> o.getField().getName()))
              .collect(Collectors.toList())) {
        Object secondValue;
        try {
          secondValue = Iterables.getOnlyElement(diff.second.get(optionDefinition));
        } catch (IllegalArgumentException e) {
          // TODO(janakr): Currently this exception should never be thrown since diff is never
          // constructed using the diff method that takes in a preexisting OptionsDiff. If this
          // changes, add a test verifying this error catching works properly.
          throw new IllegalStateException(
              "OptionsDiffForReconstruction can only handle a single first BuildOptions and a "
                  + "single second BuildOptions and has encountered multiple second BuildOptions",
              e);
        }
        valueMap.put(optionDefinition.getField().getName(), secondValue);
      }
      differingOptions.put(clazz, valueMap);
    }
    first.maybeInitializeFingerprintAndHashCode();
    return new OptionsDiffForReconstruction(
        differingOptions,
        diff.extraFirstFragments.stream()
            .sorted(lexicalFragmentOptionsComparator)
            .collect(ImmutableSet.toImmutableSet()),
        ImmutableList.sortedCopyOf(
            Comparator.comparing(o -> o.getClass().getName()), diff.extraSecondFragments),
        first.fingerprint,
        second.computeChecksum(),
        diff.starlarkSecond,
        diff.extraStarlarkOptionsFirst,
        diff.extraStarlarkOptionsSecond,
        second);
  }

  /**
   * A diff class for BuildOptions. Fields are meant to be populated and returned by {@link
   * BuildOptions#diff}
   */
  public static class OptionsDiff {
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

  /**
   * An object that encapsulates the data needed to transform one {@link BuildOptions} object into
   * another: the full fragments of the second one, the fragment classes of the first that should be
   * omitted, and the values of any fields that should be changed.
   */
  public static final class OptionsDiffForReconstruction {
    private final Map<Class<? extends FragmentOptions>, Map<String, Object>> differingOptions;
    private final ImmutableSet<Class<? extends FragmentOptions>> extraFirstFragmentClasses;
    private final ImmutableList<FragmentOptions> extraSecondFragments;
    private final byte[] baseFingerprint;
    private final String checksum;

    private final Map<Label, Object> differingStarlarkOptions;
    private final List<Label> extraFirstStarlarkOptions;
    private final Map<Label, Object> extraSecondStarlarkOptions;

    /**
     * A soft reference to the reconstructed build options to save work and garbage creation in
     * {@link #applyDiff}.
     *
     * <p>Promotes reuse of a single {@code BuildOptions} instance to preserve reference equality
     * and limit fingerprint/hashCode initialization.
     */
    private SoftReference<BuildOptions> cachedReconstructed;

    public OptionsDiffForReconstruction(
        Map<Class<? extends FragmentOptions>, Map<String, Object>> differingOptions,
        ImmutableSet<Class<? extends FragmentOptions>> extraFirstFragmentClasses,
        ImmutableList<FragmentOptions> extraSecondFragments,
        byte[] baseFingerprint,
        String checksum,
        Map<Label, Object> differingStarlarkOptions,
        List<Label> extraFirstStarlarkOptions,
        Map<Label, Object> extraSecondStarlarkOptions,
        @Nullable BuildOptions original) {
      this.differingOptions = differingOptions;
      this.extraFirstFragmentClasses = extraFirstFragmentClasses;
      this.extraSecondFragments = extraSecondFragments;
      this.baseFingerprint = baseFingerprint;
      this.checksum = checksum;
      this.differingStarlarkOptions = differingStarlarkOptions;
      this.extraFirstStarlarkOptions = extraFirstStarlarkOptions;
      this.extraSecondStarlarkOptions = extraSecondStarlarkOptions;
      this.cachedReconstructed = new SoftReference<>(original);
    }

    private static OptionsDiffForReconstruction getEmpty(byte[] baseFingerprint, String checksum) {
      return new OptionsDiffForReconstruction(
          ImmutableMap.of(),
          ImmutableSet.of(),
          ImmutableList.of(),
          baseFingerprint,
          checksum,
          ImmutableMap.of(),
          ImmutableList.of(),
          ImmutableMap.of(),
          /*original=*/ null);
    }

    @Nullable
    @VisibleForTesting
    FragmentOptions transformOptions(FragmentOptions input) {
      Class<? extends FragmentOptions> clazz = input.getClass();
      if (extraFirstFragmentClasses.contains(clazz)) {
        return null;
      }
      Map<String, Object> changedOptions = differingOptions.get(clazz);
      if (changedOptions == null || changedOptions.isEmpty()) {
        return input;
      }
      FragmentOptions newOptions = input.clone();
      for (Map.Entry<String, Object> entry : changedOptions.entrySet()) {
        try {
          clazz.getField(entry.getKey()).set(newOptions, entry.getValue());
        } catch (IllegalAccessException | NoSuchFieldException e) {
          throw new IllegalStateException("Couldn't set " + entry + " for " + newOptions, e);
        }
      }
      return newOptions;
    }

    public String getChecksum() {
      return checksum;
    }

    private boolean isEmpty() {
      return differingOptions.isEmpty()
          && extraFirstFragmentClasses.isEmpty()
          && extraSecondFragments.isEmpty()
          && differingStarlarkOptions.isEmpty()
          && extraFirstStarlarkOptions.isEmpty()
          && extraSecondStarlarkOptions.isEmpty();
    }

    /**
     * Compares the fragment sets in the options described by two diffs with the same base.
     *
     * @see ConfigurationComparer
     */
    public static ConfigurationComparer.Result compareFragments(
        OptionsDiffForReconstruction left, OptionsDiffForReconstruction right) {
      // TODO: Add support for marking Starlark options as known default when trimming
      // (sentinel object?)
      Preconditions.checkArgument(
          Arrays.equals(left.baseFingerprint, right.baseFingerprint),
          "Can't compare diffs with different bases: %s and %s",
          left,
          right);
      // This code effectively looks up each piece of data (native fragment or Starlark option) in
      // this table (numbers reference comments in the code below):
      // ▼left  right▶  (none)           extraSecond      extraFirst      differing
      // (none)          equal            right only (#4)  left only (#4)  different (#1)
      // extraSecond     left only (#4)   compare (#3)     (impossible)    (impossible)
      // extraFirst      right only (#4)  (impossible)     equal           right only (#4)
      // differing       different (#1)   (impossible)     left only (#4)  compare (#2)

      // Any difference in shared data is grounds to return DIFFERENT, which happens if:
      // 1a. any starlark option was changed by one diff, but is neither changed nor removed by
      // the other
      if (left.hasChangeToStarlarkOptionUnchangedIn(right)
          || right.hasChangeToStarlarkOptionUnchangedIn(left)) {
        return ConfigurationComparer.Result.DIFFERENT;
      }
      // 1b. any native fragment was changed by one diff, but is neither changed nor removed by
      // the other
      if (left.hasChangeToNativeFragmentUnchangedIn(right)
          || right.hasChangeToNativeFragmentUnchangedIn(left)) {
        return ConfigurationComparer.Result.DIFFERENT;
      }
      // 2a. any starlark option was changed by both diffs, but to different values
      if (!commonKeysHaveEqualValues(
          left.differingStarlarkOptions, right.differingStarlarkOptions)) {
        return ConfigurationComparer.Result.DIFFERENT;
      }
      // 2b. any native fragment was changed by both diffs, but to different values
      if (!commonKeysHaveEqualValues(left.differingOptions, right.differingOptions)) {
        return ConfigurationComparer.Result.DIFFERENT;
      }
      // 3a. any starlark option was added by both diffs, but with different values
      if (!commonKeysHaveEqualValues(
          left.extraSecondStarlarkOptions, right.extraSecondStarlarkOptions)) {
        return ConfigurationComparer.Result.DIFFERENT;
      }
      // 3b. any native fragment was added by both diffs, but with different values
      if (!commonKeysHaveEqualValues(
          left.getExtraSecondFragmentsByClass(), right.getExtraSecondFragmentsByClass())) {
        return ConfigurationComparer.Result.DIFFERENT;
      }

      // At this point DIFFERENT is definitely not the result, so depending on which side(s) have
      // extra data, we can decide which of the remaining choices to return. (#4)
      boolean leftHasExtraData = left.hasExtraNativeFragmentsOrStarlarkOptionsNotIn(right);
      boolean rightHasExtraData = right.hasExtraNativeFragmentsOrStarlarkOptionsNotIn(left);

      if (leftHasExtraData && rightHasExtraData) {
        // If both have data that the other does not, all-shared-fragments-are-equal is all
        // that can be said.
        return ConfigurationComparer.Result.ALL_SHARED_FRAGMENTS_EQUAL;
      } else if (leftHasExtraData) {
        // If only the left instance has extra data, left is a superset of right.
        return ConfigurationComparer.Result.SUPERSET;
      } else if (rightHasExtraData) {
        // If only the right instance has extra data, left is a subset of right.
        return ConfigurationComparer.Result.SUBSET;
      } else {
        // If there is no extra data, the two options described by these diffs are equal.
        return ConfigurationComparer.Result.EQUAL;
      }
    }

    /**
     * Clears {@link #cachedReconstructed} so that tests can cover the core logic of {@link
     * #applyDiff}.
     */
    @VisibleForTesting
    void clearCachedReconstructedForTesting() {
      cachedReconstructed = new SoftReference<>(null);
    }

    private boolean hasChangeToStarlarkOptionUnchangedIn(OptionsDiffForReconstruction that) {
      Set<Label> starlarkOptionsChangedOrRemovedInThat =
          Sets.union(
              that.differingStarlarkOptions.keySet(),
              ImmutableSet.copyOf(that.extraFirstStarlarkOptions));
      return !starlarkOptionsChangedOrRemovedInThat.containsAll(
          this.differingStarlarkOptions.keySet());
    }

    private boolean hasChangeToNativeFragmentUnchangedIn(OptionsDiffForReconstruction that) {
      Set<Class<? extends FragmentOptions>> nativeFragmentsChangedOrRemovedInThat =
          Sets.union(that.differingOptions.keySet(), that.extraFirstFragmentClasses);
      return !nativeFragmentsChangedOrRemovedInThat.containsAll(this.differingOptions.keySet());
    }

    private Map<Class<? extends FragmentOptions>, FragmentOptions>
        getExtraSecondFragmentsByClass() {
      ImmutableMap.Builder<Class<? extends FragmentOptions>, FragmentOptions> builder =
          new ImmutableMap.Builder<>();
      for (FragmentOptions options : extraSecondFragments) {
        builder.put(options.getClass(), options);
      }
      return builder.build();
    }

    private static <K> boolean commonKeysHaveEqualValues(Map<K, ?> left, Map<K, ?> right) {
      Set<K> commonKeys = Sets.intersection(left.keySet(), right.keySet());
      for (K commonKey : commonKeys) {
        if (!Objects.equals(left.get(commonKey), right.get(commonKey))) {
          return false;
        }
      }
      return true;
    }

    private boolean hasExtraNativeFragmentsOrStarlarkOptionsNotIn(
        OptionsDiffForReconstruction that) {
      // extra fragments/options can be...
      // starlark options added by this diff, but not that one
      if (!that.extraSecondStarlarkOptions
          .keySet()
          .containsAll(this.extraSecondStarlarkOptions.keySet())) {
        return true;
      }
      // native fragments added by this diff, but not that one
      if (!that.getExtraSecondFragmentsByClass()
          .keySet()
          .containsAll(this.getExtraSecondFragmentsByClass().keySet())) {
        return true;
      }
      // starlark options removed by that diff, but not this one
      if (!this.extraFirstStarlarkOptions.containsAll(that.extraFirstStarlarkOptions)) {
        return true;
      }
      // native fragments removed by that diff, but not this one
      if (!this.extraFirstFragmentClasses.containsAll(that.extraFirstFragmentClasses)) {
        return true;
      }
      return false;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof OptionsDiffForReconstruction)) {
        return false;
      }
      OptionsDiffForReconstruction that = (OptionsDiffForReconstruction) o;
      return Arrays.equals(this.baseFingerprint, that.baseFingerprint)
          && this.checksum.equals(that.checksum);
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("differingOptions", differingOptions)
          .add("extraFirstFragmentClasses", extraFirstFragmentClasses)
          .add("extraSecondFragments", extraSecondFragments)
          .add("differingStarlarkOptions", differingStarlarkOptions)
          .add("extraFirstStarlarkOptions", extraFirstStarlarkOptions)
          .add("extraSecondStarlarkOptions", extraSecondStarlarkOptions)
          .toString();
    }

    @Override
    public int hashCode() {
      return 31 * Arrays.hashCode(baseFingerprint) + checksum.hashCode();
    }

    @SuppressWarnings("unused") // Used reflectively.
    private static class Codec implements ObjectCodec<OptionsDiffForReconstruction> {

      @Override
      public Class<OptionsDiffForReconstruction> getEncodedClass() {
        return OptionsDiffForReconstruction.class;
      }

      @Override
      public void serialize(
          SerializationContext context,
          OptionsDiffForReconstruction diff,
          CodedOutputStream codedOut)
          throws SerializationException, IOException {
        OptionsDiffCache cache = context.getDependency(OptionsDiffCache.class);
        ByteString bytes = cache.getBytesFromOptionsDiff(diff);
        if (bytes == null) {
          context = context.getNewNonMemoizingContext();
          ByteString.Output byteStringOut = ByteString.newOutput();
          CodedOutputStream bytesOut = CodedOutputStream.newInstance(byteStringOut);
          context.serialize(diff.differingOptions, bytesOut);
          context.serialize(diff.extraFirstFragmentClasses, bytesOut);
          context.serialize(diff.extraSecondFragments, bytesOut);
          bytesOut.writeByteArrayNoTag(diff.baseFingerprint);
          context.serialize(diff.checksum, bytesOut);
          context.serialize(diff.differingStarlarkOptions, bytesOut);
          context.serialize(diff.extraFirstStarlarkOptions, bytesOut);
          context.serialize(diff.extraSecondStarlarkOptions, bytesOut);
          bytesOut.flush();
          byteStringOut.flush();
          int optionsDiffSize = byteStringOut.size();
          bytes = byteStringOut.toByteString();
          cache.putBytesFromOptionsDiff(diff, bytes);
          logger.atFine().log(
              "Serialized OptionsDiffForReconstruction %s. Diff took %d bytes.",
              diff, optionsDiffSize);
        }
        codedOut.writeBytesNoTag(bytes);
      }

      @Override
      public OptionsDiffForReconstruction deserialize(
          DeserializationContext context, CodedInputStream codedIn)
          throws SerializationException, IOException {
        OptionsDiffCache cache = context.getDependency(OptionsDiffCache.class);
        ByteString bytes = codedIn.readBytes();
        OptionsDiffForReconstruction diff = cache.getOptionsDiffFromBytes(bytes);
        if (diff == null) {
          CodedInputStream codedInput = bytes.newCodedInput();
          context = context.getNewNonMemoizingContext();
          Map<Class<? extends FragmentOptions>, Map<String, Object>> differingOptions =
              context.deserialize(codedInput);
          ImmutableSet<Class<? extends FragmentOptions>> extraFirstFragmentClasses =
              context.deserialize(codedInput);
          ImmutableList<FragmentOptions> extraSecondFragments = context.deserialize(codedInput);
          byte[] baseFingerprint = codedInput.readByteArray();
          String checksum = context.deserialize(codedInput);
          Map<Label, Object> differingStarlarkOptions = context.deserialize(codedInput);
          List<Label> extraFirstStarlarkOptions = context.deserialize(codedInput);
          Map<Label, Object> extraSecondStarlarkOptions = context.deserialize(codedInput);
          diff =
              new OptionsDiffForReconstruction(
                  differingOptions,
                  extraFirstFragmentClasses,
                  extraSecondFragments,
                  baseFingerprint,
                  checksum,
                  differingStarlarkOptions,
                  extraFirstStarlarkOptions,
                  extraSecondStarlarkOptions,
                  /*original=*/ null);
          cache.putBytesFromOptionsDiff(diff, bytes);
        }
        return diff;
      }
    }
  }

  /**
   * Injected cache for {@code Codec}, so that we don't have to repeatedly serialize the same
   * object. We still incur the over-the-wire cost of the bytes, but we don't use CPU to repeatedly
   * compute it.
   *
   * <p>We provide the cache as an injected dependency so that different serializers' caches are
   * isolated.
   *
   * <p>Used when configured targets are serialized: the more compact {@link
   * FingerprintingKDiffToByteStringCache} cache below cannot be easily used because a configured
   * target may have an edge to a configured target in a different configuration, and with only the
   * checksum there would be no way to compute that configuration (although it is very likely in the
   * graph already).
   */
  public interface OptionsDiffCache {
    ByteString getBytesFromOptionsDiff(OptionsDiffForReconstruction diff);

    void putBytesFromOptionsDiff(OptionsDiffForReconstruction diff, ByteString bytes);

    OptionsDiffForReconstruction getOptionsDiffFromBytes(ByteString bytes);
  }

  /**
   * Implementation of the {@link OptionsDiffCache} that acts as a {@code BiMap} utilizing two
   * {@code ConcurrentHashMaps}.
   */
  public static final class DiffToByteCache implements OptionsDiffCache {
    // We expect there to be very few elements so keeping the reverse map as well as the forward
    // map should be very cheap.
    private final ConcurrentHashMap<OptionsDiffForReconstruction, ByteString> diffToByteStringMap =
        new ConcurrentHashMap<>();
    private final ConcurrentHashMap<ByteString, OptionsDiffForReconstruction> byteStringToDiffMap =
        new ConcurrentHashMap<>();

    @Override
    public ByteString getBytesFromOptionsDiff(OptionsDiffForReconstruction diff) {
      return diffToByteStringMap.get(diff);
    }

    @Override
    public void putBytesFromOptionsDiff(OptionsDiffForReconstruction diff, ByteString bytes) {
      // We need to insert data into map that will be used for deserialization first in case there
      // is a race between two threads. If we populated the diffToByteStringMap first, another
      // thread could get the result above and race ahead, but then get a cache miss on
      // deserialization.
      byteStringToDiffMap.put(bytes, diff);
      diffToByteStringMap.put(diff, bytes);
    }

    @Override
    public OptionsDiffForReconstruction getOptionsDiffFromBytes(ByteString bytes) {
      return byteStringToDiffMap.get(bytes);
    }
  }

  /**
   * {@link BuildOptions.OptionsDiffForReconstruction} serialization cache that relies on only
   * serializing the checksum string instead of the full OptionsDiffForReconstruction.
   *
   * <p>This requires that every {@link BuildOptions.OptionsDiffForReconstruction} instance
   * encountered is serialized <i>before</i> it is ever deserialized. When not serializing
   * configured targets, this holds because every configuration present in the build is looked up in
   * the graph using a {@code BuildConfigurationValue.Key}, which contains its {@link
   * BuildOptions.OptionsDiffForReconstruction}. This requires that {@code BuildConfigurationValue}
   * instances must always be serialized.
   */
  public static class FingerprintingKDiffToByteStringCache
      implements BuildOptions.OptionsDiffCache {
    private static final ConcurrentHashMap<OptionsDiffForReconstruction, ByteString>
        diffToByteStringCache = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<ByteString, OptionsDiffForReconstruction>
        byteStringToDiffMap = new ConcurrentHashMap<>();

    @Override
    public ByteString getBytesFromOptionsDiff(OptionsDiffForReconstruction diff) {
      ByteString checksumString = diffToByteStringCache.get(diff);
      if (checksumString != null) {
        // Fast path to avoid ByteString creation churn and unnecessary map insertions.
        return checksumString;
      }
      checksumString = ByteString.copyFromUtf8(diff.getChecksum());
      // We need to insert data into map that will be used for deserialization first in case there
      // is a race between two threads. If we populated the diffToByteStringCache first, another
      // thread could get checksumString above during serialization and race ahead, but then get a
      // cache miss on deserialization.
      byteStringToDiffMap.put(checksumString, diff);
      diffToByteStringCache.put(diff, checksumString);
      return checksumString;
    }

    @Override
    public void putBytesFromOptionsDiff(OptionsDiffForReconstruction diff, ByteString bytes) {
      throw new UnsupportedOperationException(
          "diff "
              + diff
              + " should have not been serialized: "
              + diffToByteStringCache
              + ", "
              + byteStringToDiffMap);
    }

    @Override
    public OptionsDiffForReconstruction getOptionsDiffFromBytes(ByteString bytes) {
      return Preconditions.checkNotNull(
          byteStringToDiffMap.get(bytes),
          "Missing bytes %s: %s %s",
          bytes,
          diffToByteStringCache,
          byteStringToDiffMap);
    }
  }
}
