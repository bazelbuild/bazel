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
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import java.util.Map;

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
