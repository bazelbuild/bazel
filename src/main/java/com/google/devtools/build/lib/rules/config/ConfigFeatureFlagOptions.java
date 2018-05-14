// Copyright 2017 The Bazel Authors. All rights reserved.
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
// limitations under the License

package com.google.devtools.build.lib.rules.config;

import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/** The options fragment which defines {@link ConfigFeatureFlagConfiguration}. */
@AutoCodec(strategy = AutoCodec.Strategy.PUBLIC_FIELDS)
public final class ConfigFeatureFlagOptions extends FragmentOptions {
  /** A converter used by the flag options which always returns an empty map, ignoring input. */
  public static final class EmptyImmutableSortedMapConverter
      implements Converter<ImmutableSortedMap<Label, String>> {
    @Override
    public ImmutableSortedMap<Label, String> convert(String input) {
      return ImmutableSortedMap.<Label, String>of();
    }

    @Override
    public String getTypeDescription() {
      return "n/a (do not set this on the command line)";
    }
  }

  /** A converter used by the flag options which always returns an empty set, ignoring input. */
  public static final class EmptyImmutableSortedSetConverter
      implements Converter<ImmutableSortedSet<Label>> {
    @Override
    public ImmutableSortedSet<Label> convert(String input) {
      return ImmutableSortedSet.of();
    }

    @Override
    public String getTypeDescription() {
      return "n/a (do not set this on the command line)";
    }
  }

  /**
   * Whether to perform user-guided trimming of feature flags based on the tagging in the
   * transitive_configs attribute.
   *
   * <p>Currently a no-op.
   */
  @Option(
    name = "enforce_transitive_configs_for_config_feature_flag",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {
      OptionEffectTag.LOSES_INCREMENTAL_STATE,
      OptionEffectTag.AFFECTS_OUTPUTS,
      OptionEffectTag.BUILD_FILE_SEMANTICS,
      OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION,
      OptionEffectTag.LOADING_AND_ANALYSIS
    },
    metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
    defaultValue = "false"
  )
  public boolean enforceTransitiveConfigsForConfigFeatureFlag = false;

  /** The mapping from config_feature_flag rules to their values. */
  @Option(
    name = "config_feature_flag values (private)",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {
      OptionEffectTag.AFFECTS_OUTPUTS,
      OptionEffectTag.BUILD_FILE_SEMANTICS,
      OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION,
      OptionEffectTag.LOADING_AND_ANALYSIS
    },
    metadataTags = {OptionMetadataTag.INTERNAL},
    converter = EmptyImmutableSortedMapConverter.class,
    defaultValue = "{}"
  )
  public ImmutableSortedMap<Label, String> flagValues = ImmutableSortedMap.of();

  /**
   * The set of feature flags which are definitely set to their default values.
   *
   * <p>If the set is non-null, the current configuration is trimmed, and this set contains the
   * labels of feature flags whose values are known to be default in the current configuration.
   * In this case:
   *
   * <ul>
   *   <li>Keys present in flagValues are known to have non-default values. The value of such a
   *       feature flag is the value in flagValues.
   *   <li>Keys present in this set are known to have default values. The value of such a feature
   *       flag is its default value.
   *   <li>Keys missing from both flagValues and this set have unknown values - they may be unset
   *       and have their default value, or they may be set to a non-default value which has been
   *       trimmed out. Attempting to access the value of such a feature flag is an error.
   * </ul>
   *
   * <p>If the set is null, the current configuration is untrimmed, and flagValues contains the
   * mapping of ALL feature flags with non-default values. In this case:
   *
   * <ul>
   *   <li>Keys present in flagValues are known to have non-default values. The value of such a
   *       feature flag is the value in flagValues.
   *   <li>Keys missing from flagValues are known to have default values. The value of such a
   *       feature flag is its default value.
   * </ul>
   */
  @Option(
    name = "config_feature_flag known default values (private)",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {
      OptionEffectTag.AFFECTS_OUTPUTS,
      OptionEffectTag.BUILD_FILE_SEMANTICS,
      OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION,
      OptionEffectTag.LOADING_AND_ANALYSIS
    },
    metadataTags = {OptionMetadataTag.INTERNAL},
    converter = EmptyImmutableSortedSetConverter.class,
    defaultValue = "null"
  )
  public ImmutableSortedSet<Label> knownDefaultFlags = null;

  /**
   * The set of feature flags which were requested but whose values are not known. If this value
   * is ever set non-empty, the configuration loader fails.
   */
  @Option(
    name = "config_feature_flag unknown values (private)",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {
      OptionEffectTag.AFFECTS_OUTPUTS,
      OptionEffectTag.BUILD_FILE_SEMANTICS,
      OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION,
      OptionEffectTag.LOADING_AND_ANALYSIS
    },
    metadataTags = {OptionMetadataTag.INTERNAL},
    converter = EmptyImmutableSortedSetConverter.class,
    defaultValue = "{}"
  )
  public ImmutableSortedSet<Label> unknownFlags = ImmutableSortedSet.of();

  /**
   * Retrieves the map of flag-value pairs for flags which are definitely set to some non-default
   * value.
   */
  public ImmutableSortedMap<Label, String> getFlagValues() {
    return this.flagValues;
  }

  /**
   * Retrieves the set of flags which are definitely set to their default values.
   *
   * <p>The returned Optional will be empty if {@link isTrimmed} is false. In this case, all flags
   * not in {@link getFlagValues} should be considered set to their default values.
   */
  public Optional<ImmutableSortedSet<Label>> getKnownDefaultFlags() {
    return Optional.ofNullable(this.knownDefaultFlags);
  }

  /**
   * Returns whether this configuration has been trimmed, meaning that not all feature flags' values
   * are known.
   */
  public boolean isTrimmed() {
    return this.knownDefaultFlags != null;
  }

  /**
   * Retrieves the set of flags whose values were requested while trimming, but whose values are not
   * known.
   *
   * <p>If this set is non-empty, this configuration is in error; a target requested a flag which
   * was not requested by earlier trimmings.
   */
  public ImmutableSortedSet<Label> getUnknownFlags() {
    return this.unknownFlags;
  }

  /**
   * Replaces the set of flag-value pairs with the given mapping of flag-value pairs.
   *
   * <p>Flags not present in the new {@code flagValues} will return to being unset! To set flags
   * while still retaining the values already set, call {@link #getFlagValues()} and build a map
   * containing both the old values and the new ones. Note that when {@link #isTrimmed()} is true,
   * it's not possible to know the values of ALL flags.
   *
   * <p>Because this method replaces the entire set of flag values, all flag values for this
   * configuration are known, and thus knownValues is set to null, and unknownFlags is cleared.
   * After this method is called, isTrimmed will return false.
   */
  public void replaceFlagValues(Map<Label, String> flagValues) {
    this.flagValues = ImmutableSortedMap.copyOf(flagValues);
    this.knownDefaultFlags = null;
    this.unknownFlags = ImmutableSortedSet.of();
  }

  /**
   * Trims the set of known flag-value pairs to the given set.
   *
   * <p>Each target which participates in manual trimming will call this method (via
   * ConfigFeatureFlagTaggedTrimmingTransitionFactory) with its set of requested flags. This set
   * typically comes straight from the user via the transitive_configs attribute. For feature
   * flags themselves, this will be a singleton set containing the feature flag's own label.
   *
   * <p>At the top level, or when there is also a transition which calls replaceFlagValues (e.g.,
   * ConfigFeatureFlagValuesTransition, created by ConfigFeatureFlagTransitionFactory and used by
   * android_binary among others), the configuration will start off untrimmed (knownDefaultFlags is
   * null). In this case:
   *
   * <ul>
   *   <li>Any map entries from flagValues whose keys are in requiredFlags will be retained in
   *       flagValues; all other entries of flagValues will be discarded.</li>
   *   <li>All other elements of requiredFlags will be put into knownDefaultFlags.</li>
   *   <li>unknownFlags will always be set to the empty set; its old value will be discarded.</li>
   * </ul>
   *
   * <p>At any place other than the top level and the aforementioned replaceFlagValues transitions,
   * the source configuration is already trimmed (knownDefaultFlags is not null). In this case:
   *
   * <ul>
   *   <li>Any map entries from flagValues which have keys that are in requiredFlags will be
   *       retained in flagValues; all other entries of flagValues will be discarded.</li>
   *   <li>Any elements of knownDefaultFlags which are also in requiredFlags will be retained in
   *       knownDefaultFlags; all other elements of knownDefaultFlags will be discarded.</li>
   *   <li>unknownFlags will be set to contain all other elements of requiredFlags; its old value
   *       will be discarded.</li>
   * </ul>
   *
   * <p>If requiredFlags is empty, then flagValues, knownDefaultFlags, and unknownFlags will all be
   * set to empty values.
   *
   * <p>After this method is called, regardless of circumstances:
   *
   * <ul>
   *   <li>knownDefaultValues will be non-null, and thus isTrimmed will return true, indicating that
   *       the configuration is trimmed.</li>
   *   <li>If unknownFlags is set non-empty, this indicates that the target this configuration is
   *       for has been reached via a path which mistakenly trimmed out one or more of the flags it
   *       needs, and thus there isn't enough information to evaluate it.</li>
   * </ul>
   */
  public void trimFlagValues(Set<Label> requiredFlags) {
    ImmutableSortedMap.Builder<Label, String> flagValuesBuilder =
        ImmutableSortedMap.naturalOrder();
    ImmutableSortedSet.Builder<Label> knownDefaultFlagsBuilder = ImmutableSortedSet.naturalOrder();
    ImmutableSortedSet.Builder<Label> unknownFlagsBuilder = ImmutableSortedSet.naturalOrder();

    for (Label label : requiredFlags) {
      if (this.flagValues.containsKey(label)) {
        flagValuesBuilder.put(label, flagValues.get(label));
      } else if (!this.isTrimmed() || this.knownDefaultFlags.contains(label)) {
        knownDefaultFlagsBuilder.add(label);
      } else {
        unknownFlagsBuilder.add(label);
      }
    }

    this.flagValues = flagValuesBuilder.build();
    this.knownDefaultFlags = knownDefaultFlagsBuilder.build();
    this.unknownFlags = unknownFlagsBuilder.build();
  }
}
