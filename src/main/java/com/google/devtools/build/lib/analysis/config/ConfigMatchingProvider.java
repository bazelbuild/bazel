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
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.util.Map;

/**
 * A "configuration target" that asserts whether or not it matches the configuration it's bound to.
 *
 * <p>This can be used, e.g., to declare a BUILD target that defines the conditions which trigger a
 * configurable attribute branch. In general, this can be used to trigger for any user-configurable
 * build behavior.
 */
@Immutable
@AutoCodec
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
  @AutoCodec.Instantiator
  public static ConfigMatchingProvider create(
      Label label,
      ImmutableMultimap<String, String> settingsMap,
      ImmutableMap<Label, String> flagSettingsMap,
      ImmutableSet<String> requiredFragmentOptions,
      boolean matches) {
    return new AutoValue_ConfigMatchingProvider(
        label, settingsMap, flagSettingsMap, requiredFragmentOptions, matches);
  }

  /** The target's label. */
  public abstract Label label();

  abstract ImmutableMultimap<String, String> settingsMap();

  abstract ImmutableMap<Label, String> flagSettingsMap();

  public abstract ImmutableSet<String> requiredFragmentOptions();

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

    if (!settings.containsAll(otherSettings)) {
      // not a superset
      return false;
    }

    if (!flagSettings.containsAll(otherFlagSettings)) {
      // not a superset
      return false;
    }

    if (!(settings.size() > otherSettings.size()
        || flagSettings.size() > otherFlagSettings.size())) {
      // not a proper superset
      return false;
    }

    return true;
  }

  /** Format this provider as its label. */
  @Override
  public final String toString() {
    return label().toString();
  }
}
