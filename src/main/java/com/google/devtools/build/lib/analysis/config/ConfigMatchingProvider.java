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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.RequiredConfigFragmentsProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import java.util.Map;

/**
 * A "configuration target" that asserts whether or not it matches the configuration it's bound to.
 *
 * <p>This can be used, e.g., to declare a BUILD target that defines the conditions which trigger a
 * configurable attribute branch. In general, this can be used to trigger for any user-configurable
 * build behavior.
 */
@Immutable
@AutoValue
public abstract class ConfigMatchingProvider implements TransitiveInfoProvider {

  /**
   * @param label the build label corresponding to this matcher
   * @param settingsMap the condition settings that trigger this matcher
   * @param flagSettingsMap the label-keyed settings that trigger this matcher
   * @param requiredFragmentOptions {@link FragmentOptions} required to match the options this
   *     matcher checks. This provides comparable functionality to {@link
   *     com.google.devtools.build.lib.analysis.RequiredConfigFragmentsProvider}.
   * @param matches whether or not this matcher matches the configuration associated with its
   *     configured target
   */
  public static ConfigMatchingProvider create(
      Label label,
      ImmutableMultimap<String, String> settingsMap,
      ImmutableMap<Label, String> flagSettingsMap,
      RequiredConfigFragmentsProvider requiredFragmentOptions,
      ImmutableSet<Label> constraintValueSettings,
      boolean matches) {
    return new AutoValue_ConfigMatchingProvider(
        label,
        settingsMap,
        flagSettingsMap,
        requiredFragmentOptions,
        constraintValueSettings,
        matches);
  }

  /** The target's label. */
  public abstract Label label();

  abstract ImmutableMultimap<String, String> settingsMap();

  abstract ImmutableMap<Label, String> flagSettingsMap();

  public abstract RequiredConfigFragmentsProvider requiredFragmentOptions();

  abstract ImmutableSet<Label> constraintValuesSetting();

  /**
   * Whether or not the configuration criteria defined by this target match its actual
   * configuration.
   */
  public abstract boolean matches();

  /**
   * Returns true if this matcher's conditions are a proper superset of another matcher's
   * conditions, i.e. if this matcher is a specialization of the other one.
   */
  public boolean refines(ConfigMatchingProvider other) {
    ImmutableSet<Map.Entry<String, String>> settings = ImmutableSet.copyOf(settingsMap().entries());
    ImmutableSet<Map.Entry<String, String>> otherSettings =
        ImmutableSet.copyOf(other.settingsMap().entries());
    ImmutableSet<Map.Entry<Label, String>> flagSettings = flagSettingsMap().entrySet();
    ImmutableSet<Map.Entry<Label, String>> otherFlagSettings = other.flagSettingsMap().entrySet();

    ImmutableSet<Label> constraintValueSettings = constraintValuesSetting();
    ImmutableSet<Label> otherConstraintValueSettings = other.constraintValuesSetting();

    if (!settings.containsAll(otherSettings)
        || !flagSettings.containsAll(otherFlagSettings)
        || !constraintValueSettings.containsAll(otherConstraintValueSettings)) {
      return false; // Not a superset.
    }

    return settings.size() > otherSettings.size()
        || flagSettings.size() > otherFlagSettings.size()
        || constraintValueSettings.size() > otherConstraintValueSettings.size();
  }

  /** Format this provider as its label. */
  @Override
  public final String toString() {
    return label().toString();
  }
}
