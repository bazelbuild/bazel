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

import static com.google.devtools.build.lib.packages.BuildType.LABEL_KEYED_STRING_DICT;

import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleTransitionFactory;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import java.util.Map;

/**
 * Transition factory which allows for setting the values of config_feature_flags below the rule
 * it is attached to based on one of that rule's attributes.
 *
 * <p>Currently, this is only intended for use by android_binary and other Android top-level rules.
 */
public class ConfigFeatureFlagTransitionFactory implements RuleTransitionFactory {

  /** Transition which resets the set of flag-value pairs to the map it was constructed with. */
  @AutoCodec
  @VisibleForSerialization
  static final class ConfigFeatureFlagValuesTransition implements PatchTransition {
    private final ImmutableSortedMap<Label, String> flagValues;
    private final int cachedHashCode;

    public ConfigFeatureFlagValuesTransition(Map<Label, String> flagValues) {
      this(ImmutableSortedMap.copyOf(flagValues), flagValues.hashCode());
    }

    @AutoCodec.Instantiator
    ConfigFeatureFlagValuesTransition(
        ImmutableSortedMap<Label, String> flagValues, int cachedHashCode) {
      this.flagValues = ImmutableSortedMap.copyOf(flagValues);
      this.cachedHashCode = cachedHashCode;
    }

    @Override
    public BuildOptions patch(BuildOptions options) {
      if (!options.contains(ConfigFeatureFlagOptions.class)) {
        return options;
      }
      return FeatureFlagValue.replaceFlagValues(options, flagValues);
    }

    @Override
    public boolean equals(Object other) {
      return other instanceof ConfigFeatureFlagValuesTransition
          && this.flagValues.equals(((ConfigFeatureFlagValuesTransition) other).flagValues);
    }

    @Override
    public int hashCode() {
      return cachedHashCode;
    }

    @Override
    public String toString() {
      return String.format("ConfigFeatureFlagValuesTransition{flagValues=%s}", flagValues);
    }
  }

  private final String attributeName;

  /**
   * Creates a transition factory which will generate a transition over a given rule which sets
   * exactly the flag values in the attribute with the given {@code attributeName} of that rule,
   * unsetting any flag values not listed there.
   *
   * <p>This attribute must be a nonconfigurable {@code LABEL_KEYED_STRING_DICT}.
   */
  public ConfigFeatureFlagTransitionFactory(String attributeName) {
    this.attributeName = attributeName;
  }

  @Override
  public PatchTransition buildTransitionFor(Rule rule) {
    NonconfigurableAttributeMapper attrs = NonconfigurableAttributeMapper.of(rule);
    if (attrs.isAttributeValueExplicitlySpecified(attributeName)) {
      return new ConfigFeatureFlagValuesTransition(
          attrs.get(attributeName, LABEL_KEYED_STRING_DICT));
    } else {
      return NoTransition.INSTANCE;
    }
  }

  /**
   * Returns the attribute examined by this transition factory.
   */
  public String getAttributeName() {
    return this.attributeName;
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof ConfigFeatureFlagTransitionFactory
        && this.attributeName.equals(((ConfigFeatureFlagTransitionFactory) other).attributeName);
  }

  @Override
  public int hashCode() {
    return attributeName.hashCode();
  }

  @Override
  public String toString() {
    return String.format("ConfigFeatureFlagTransitionFactory{attributeName=%s}", attributeName);
  }
}
