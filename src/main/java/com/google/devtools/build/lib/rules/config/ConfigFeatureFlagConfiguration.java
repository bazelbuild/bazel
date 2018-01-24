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

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.hash.Hasher;
import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import java.util.Map;
import java.util.Optional;
import java.util.SortedMap;
import javax.annotation.Nullable;

/**
 * Configuration fragment for Android's config_feature_flag, flags which can be defined in BUILD
 * files.
 */
@AutoCodec
public final class ConfigFeatureFlagConfiguration extends BuildConfiguration.Fragment {
  public static final ObjectCodec<ConfigFeatureFlagConfiguration> CODEC =
      new ConfigFeatureFlagConfiguration_AutoCodec();

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

  /** The options fragment which defines {@link ConfigFeatureFlagConfiguration}. */
  @AutoCodec(strategy = AutoCodec.Strategy.PUBLIC_FIELDS)
  public static final class Options extends FragmentOptions {
    public static final ObjectCodec<Options> CODEC =
        new ConfigFeatureFlagConfiguration_Options_AutoCodec();

    /** The mapping from config_feature_flag rules to their values. */
    @Option(
      name = "config_feature_flag values (private)",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      metadataTags = {OptionMetadataTag.INTERNAL},
      converter = EmptyImmutableSortedMapConverter.class,
      defaultValue = "{}"
    )
    public ImmutableSortedMap<Label, String> flagValues = ImmutableSortedMap.of();

    /** Retrieves the set of flag-value pairs. */
    public ImmutableSortedMap<Label, String> getFlagValues() {
      return this.flagValues;
    }

    /**
     * Replaces the set of flag-value pairs with the given mapping of flag-value pairs.
     *
     * <p>Flags not present in the new {@code flagValues} will return to being unset! To set flags
     * while still retaining the values already set, call {@link #getFlagValues()} and build a map
     * containing both the old values and the new ones.
     */
    public void replaceFlagValues(Map<Label, String> flagValues) {
      this.flagValues = ImmutableSortedMap.copyOf(flagValues);
    }
  }

  /**
   * A configuration fragment loader able to create instances of {@link
   * ConfigFeatureFlagConfiguration} from {@link ConfigFeatureFlagConfiguration.Options}.
   */
  public static final class Loader implements ConfigurationFragmentFactory {
    @Override
    public BuildConfiguration.Fragment create(
        ConfigurationEnvironment env, BuildOptions buildOptions) {
      return new ConfigFeatureFlagConfiguration(buildOptions.get(Options.class));
    }

    @Override
    public Class<? extends BuildConfiguration.Fragment> creates() {
      return ConfigFeatureFlagConfiguration.class;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
      return ImmutableSet.<Class<? extends FragmentOptions>>of(Options.class);
    }
  }

  private final ImmutableSortedMap<Label, String> flagValues;
  @Nullable private final String flagHash;

  /** Creates a new configuration fragment from the given {@link Options} fragment. */
  public ConfigFeatureFlagConfiguration(Options options) {
    this(options.getFlagValues());
  }

  @AutoCodec.Constructor
  ConfigFeatureFlagConfiguration(ImmutableSortedMap<Label, String> flagValues) {
    this.flagValues = flagValues;
    this.flagHash = this.flagValues.isEmpty() ? null : hashFlags(this.flagValues);
  }

  /** Converts the given flag values into a string hash for use as an output directory fragment. */
  private static String hashFlags(SortedMap<Label, String> flagValues) {
    // This hash function is relatively fast and stable between JVM invocations.
    Hasher hasher = Hashing.murmur3_128().newHasher();

    for (Map.Entry<Label, String> flag : flagValues.entrySet()) {
      hasher.putUnencodedChars(flag.getKey().toString());
      hasher.putByte((byte) 0);
      hasher.putUnencodedChars(flag.getValue());
      hasher.putByte((byte) 0);
    }
    return hasher.hash().toString();
  }

  /**
   * Retrieves the value of a configuration flag.
   *
   * <p>If the flag is not set in the current configuration, then the returned value will be absent.
   *
   * <p>This method should only be used by the rule whose label is passed here. Other rules should
   * depend on that rule and read a provider exported by it. To encourage callers of this method to
   * do the right thing, this class takes {@link ArtifactOwner} instead of {@link Label}; to get the
   * ArtifactOwner for a rule, call {@code ruleContext.getOwner()}.
   */
  public Optional<String> getFeatureFlagValue(ArtifactOwner owner) {
    return Optional.ofNullable(flagValues.get(owner.getLabel()));
  }

  /**
   * Returns a fragment of the output directory name for this configuration, based on the set of
   * flags and their values. It will be {@code null} if no flags are set.
   */
  @Nullable
  @Override
  public String getOutputDirectoryName() {
    return flagHash;
  }
}
