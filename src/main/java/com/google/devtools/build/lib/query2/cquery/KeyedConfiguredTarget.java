// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.cquery;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import javax.annotation.Nullable;

/**
 * A representation of a ConfiguredTargetKey and the ConfiguredTarget that it points to, for use in
 * configured target queries.
 */
@AutoValue
public abstract class KeyedConfiguredTarget {

  /** Returns a new KeyedConfiguredTarget for the given data. */
  public static KeyedConfiguredTarget create(
      ConfiguredTargetKey configuredTargetKey, ConfiguredTarget configuredTarget) {
    return new AutoValue_KeyedConfiguredTarget(configuredTargetKey, configuredTarget);
  }

  /** Returns the key for this KeyedConfiguredTarget. */
  @Nullable
  public abstract ConfiguredTargetKey getConfiguredTargetKey();

  /** Returns the ConfiguredTarget for this KeyedConfiguredTarget. */
  public abstract ConfiguredTarget getConfiguredTarget();

  /** Returns the original (pre-alias) label for the underlying ConfiguredTarget. */
  public Label getLabel() {
    return getConfiguredTarget().getOriginalLabel();
  }

  /** Returns the configuration key used for this KeyedConfiguredTarget. */
  @Nullable
  public BuildConfigurationValue.Key getConfigurationKey() {
    return getConfiguredTarget().getConfigurationKey();
  }

  /** Returns the configuration checksum in use for this KeyedConfiguredTarget. */
  @Nullable
  public String getConfigurationChecksum() {
    return getConfigurationKey() == null
        ? null
        : getConfigurationKey().getOptionsDiff().getChecksum();
  }

  /**
   * The configuration conditions that trigger this configured target's configurable attributes. For
   * targets that do not support configurable attributes, this will be an empty map.
   */
  public ImmutableMap<Label, ConfigMatchingProvider> getConfigConditions() {
    return getConfiguredTarget().getConfigConditions();
  }

  /** Returns a KeyedConfiguredTarget instance that resolves aliases. */
  public KeyedConfiguredTarget getActual() {
    ConfiguredTarget actual = getConfiguredTarget().getActual();
    // Use old values for unchanged fields, like toolchain ctx, if possible.
    ConfiguredTargetKey.Builder oldKey =
        (this.getConfiguredTargetKey() == null
            ? ConfiguredTargetKey.builder()
            : this.getConfiguredTargetKey().toBuilder());

    ConfiguredTargetKey actualKey =
        oldKey
            .setLabel(actual.getLabel())
            .setConfigurationKey(actual.getConfigurationKey())
            .build();
    return KeyedConfiguredTarget.create(actualKey, actual);
  }
}
