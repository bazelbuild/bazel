// Copyright 2018 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** Marker interface for detecting feature flags in the Starlark setting map. */
interface FeatureFlagValue {
  /** Returns the value of this flag, or null if it's set default. */
  @Nullable
  String getValue();

  /** A feature flag value for a flag known to be set to a particular value. */
  @AutoValue
  abstract class SetValue implements FeatureFlagValue {
    static SetValue of(String value) {
      return new AutoValue_FeatureFlagValue_SetValue(value);
    }

    @Override
    public abstract String getValue();

    @Override
    public final String toString() {
      return String.format("FeatureFlagValue.SetValue{%s}", getValue());
    }
  }

  /** A feature flag value for a flag known to be set to its default value. */
  enum DefaultValue implements FeatureFlagValue {
    INSTANCE;

    @Override
    public String getValue() {
      return null;
    }

    @Override
    public String toString() {
      return "FeatureFlagValue.DefaultValue{}";
    }
  }

  /** A feature flag value for a flag which was requested but which value was already trimmed. */
  enum UnknownValue implements FeatureFlagValue {
    INSTANCE;

    @Override
    public String getValue() {
      throw new IllegalStateException();
    }

    @Override
    public String toString() {
      return "FeatureFlagValue.UnknownValue{}";
    }
  }

  /** Returns a new BuildOptions with a new map of feature flag values. */
  static BuildOptions replaceFlagValues(BuildOptions original, Map<Label, String> newValues) {
    BuildOptions.Builder result = original.toBuilder();
    for (Map.Entry<String, Object> entry : original.getStarlarkOptions().entrySet()) {
      if (entry.getValue() instanceof FeatureFlagValue) {
        result.removeStarlarkOption(entry.getKey());
      }
    }
    ImmutableMap.Builder<String, Object> newValueObjects = new ImmutableMap.Builder<>();
    for (Map.Entry<Label, String> entry : newValues.entrySet()) {
      newValueObjects.put(entry.getKey().toString(), SetValue.of(entry.getValue()));
    }
    result.addStarlarkOptions(newValueObjects.build());
    BuildOptions builtResult = result.build();
    if (builtResult.contains(ConfigFeatureFlagOptions.class)) {
      builtResult.get(ConfigFeatureFlagOptions.class).allFeatureFlagValuesArePresent = true;
    }
    return builtResult;
  }

  /** Returns a new BuildOptions with the feature flag values trimmed down to the given flags. */
  static BuildOptions trimFlagValues(BuildOptions original, Set<Label> availableFlags) {
    BuildOptions.Builder result = original.toBuilder();
    ImmutableSet.Builder<String> seenFlagsBuilder = new ImmutableSet.Builder<>();
    for (Map.Entry<String, Object> entry : original.getStarlarkOptions().entrySet()) {
      if (entry.getValue() instanceof FeatureFlagValue) {
        seenFlagsBuilder.add(entry.getKey());
      }
    }
    // TODO(juliexxia): remove this hack-around when the SBC map is Label, Object
    ImmutableSet<String> availableFlagsAsStrings =
        availableFlags.stream().map(Label::toString).collect(toImmutableSet());
    ImmutableSet<String> seenFlags = seenFlagsBuilder.build();
    for (String trimmedFlag : Sets.difference(seenFlags, availableFlagsAsStrings)) {
      result.removeStarlarkOption(trimmedFlag);
    }
    FeatureFlagValue unknownFlagValue =
        (original.contains(ConfigFeatureFlagOptions.class)
                && original.get(ConfigFeatureFlagOptions.class).allFeatureFlagValuesArePresent)
            ? DefaultValue.INSTANCE
            : UnknownValue.INSTANCE;
    for (String unknownFlag : Sets.difference(availableFlagsAsStrings, seenFlags)) {
      result.addStarlarkOption(unknownFlag, unknownFlagValue);
    }
    BuildOptions builtResult = result.build();
    if (builtResult.contains(ConfigFeatureFlagOptions.class)) {
      builtResult.get(ConfigFeatureFlagOptions.class).allFeatureFlagValuesArePresent = false;
    }
    return builtResult;
  }

  /**
   * Returns the map of known non-default flag values. Throws UnknownValueException when a flag is
   * set to UNKNOWN_VALUE (due to an earlier trimming gone wrong).
   */
  static ImmutableSortedMap<Label, String> getFlagValues(BuildOptions options)
      throws UnknownValueException {
    ImmutableSortedMap.Builder<Label, String> knownValues = ImmutableSortedMap.naturalOrder();
    ImmutableList.Builder<Label> unknownFlagsBuilder = new ImmutableList.Builder<>();
    for (Map.Entry<String, Object> entry : options.getStarlarkOptions().entrySet()) {
      if (entry.getValue().equals(UnknownValue.INSTANCE)) {
        // we can parseAbsoluteUnchecked here and below because no one else should be putting in
        // FeatureFlagValues, and this package only puts in stringized labels.
        // TODO(juliexxia): remove this parsing when the SBC map is Label, Object
        unknownFlagsBuilder.add(Label.parseAbsoluteUnchecked(entry.getKey()));
      } else if (entry.getValue() instanceof FeatureFlagValue) {
        String value = ((FeatureFlagValue) entry.getValue()).getValue();
        if (value != null) {
          knownValues.put(Label.parseAbsoluteUnchecked(entry.getKey()), value);
        }
      }
    }
    ImmutableList<Label> unknownFlags = unknownFlagsBuilder.build();
    if (!unknownFlags.isEmpty()) {
      throw new UnknownValueException(unknownFlags);
    }
    return knownValues.build();
  }

  /** Exception class for when getFlagValues runs into UNKNOWN_VALUE. */
  static final class UnknownValueException extends InvalidConfigurationException {
    private static final String ERROR_TEMPLATE =
        "Feature flag %1$s was accessed in a configuration it is not present in. All "
            + "targets which depend on %1$s directly or indirectly must name it in their "
            + "transitive_configs attribute.";
    private final ImmutableList<Label> unknownFlags;

    UnknownValueException(ImmutableList<Label> unknownFlags) {
      super(
          "Some feature flags were incorrectly specified:\n"
              + unknownFlags.stream()
                  .map((missingLabel) -> String.format(ERROR_TEMPLATE, missingLabel))
                  .collect(Collectors.joining("\n")));
      this.unknownFlags = unknownFlags;
    }

    ImmutableList<Label> getUnknownFlags() {
      return unknownFlags;
    }
  }
}
