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
import com.google.devtools.build.lib.analysis.config.PatchTransition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleTransitionFactory;
import java.util.Map;

/**
 * Transition factory which allows for setting the values of config_feature_flags below the rule
 * it is attached to based on one of that rule's attributes.
 *
 * <p>Currently, this is only intended for use by android_binary and other Android top-level rules.
 */
public class ConfigFeatureFlagTransitionFactory implements RuleTransitionFactory {

  /** Transition which resets the set of flag-value pairs to the map it was constructed with. */
  private static final class ConfigFeatureFlagValuesTransition implements PatchTransition {
    private final ImmutableSortedMap<Label, String> flagValues;
    private final int cachedHashCode;

    public ConfigFeatureFlagValuesTransition(Map<Label, String> flagValues) {
      this.flagValues = ImmutableSortedMap.copyOf(flagValues);
      this.cachedHashCode = this.flagValues.hashCode();
    }

    @Override
    public BuildOptions apply(BuildOptions options) {
      if (!options.contains(ConfigFeatureFlagConfiguration.Options.class)) {
        return options;
      }
      BuildOptions result = options.clone();
      result.get(ConfigFeatureFlagConfiguration.Options.class).replaceFlagValues(flagValues);
      return result;
    }

    @Override
    public boolean defaultsToSelf() {
      throw new UnsupportedOperationException("supported in dynamic mode only");
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
    return new ConfigFeatureFlagValuesTransition(
        NonconfigurableAttributeMapper.of(rule).get(attributeName, LABEL_KEYED_STRING_DICT));
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
