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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.actions.ArtifactOwner;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.RequiresOptions;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.Optional;

/**
 * Configuration fragment for Android's config_feature_flag, flags which can be defined in BUILD
 * files.
 */
@RequiresOptions(options = ConfigFeatureFlagOptions.class, starlark = true)
public final class ConfigFeatureFlagConfiguration extends Fragment {
  private final ImmutableSortedMap<Label, String> flagValues;

  /** Creates a new configuration fragment from the given {@link ConfigFeatureFlagOptions}. */
  public ConfigFeatureFlagConfiguration(BuildOptions buildOptions)
      throws InvalidConfigurationException {
    this(FeatureFlagValue.getFlagValues(buildOptions));
  }

  @VisibleForTesting
  ConfigFeatureFlagConfiguration(ImmutableSortedMap<Label, String> flagValues) {
    this.flagValues = flagValues;
  }

  /**
   * Retrieves the value of a configuration flag.
   *
   * <p>If the flag is not set in the current configuration, then the returned value will be empty.
   *
   * <p>Because the configuration should fail to construct if a required flag is missing, and
   * because config_feature_flag (the only intended user of this method) automatically requires its
   * own label, this method should never incorrectly return the default value for a flag which was
   * set to a non-default value. If the flag value is available and non-default, it will be in
   * flagValues; if the flag value is available and default, it will not be in flagValues; if the
   * flag value is unavailable, this fragment's loader will fail and this method will never be
   * called.
   *
   * <p>This method should only be used by the rule whose label is passed here. Other rules should
   * depend on that rule and read a provider exported by it. To encourage callers of this method to
   * do the right thing, this class takes {@link ArtifactOwner} instead of {@link Label}; to get the
   * ArtifactOwner for a rule, call {@code ruleContext.getOwner()}.
   */
  public Optional<String> getFeatureFlagValue(ArtifactOwner owner) {
    if (flagValues.containsKey(owner.getLabel())) {
      return Optional.of(flagValues.get(owner.getLabel()));
    } else {
      return Optional.empty();
    }
  }
}
