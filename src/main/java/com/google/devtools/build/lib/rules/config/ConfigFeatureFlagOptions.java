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

import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;

/** The options fragment which defines options related to tagged trimming of feature flags. */
public final class ConfigFeatureFlagOptions extends FragmentOptions {
  /**
   * Whether to perform user-guided trimming of feature flags based on the tagging in the
   * transitive_configs attribute.
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
      defaultValue = "false")
  public boolean enforceTransitiveConfigsForConfigFeatureFlag = false;

  /**
   * If {@code true}, the configuration contains all non-default feature flag values, and any flags
   * which are not present are known to have their default value; if {@code false}, the
   * configuration only contains some feature flag values, and all others have been trimmed and so
   * nothing is known about their values.
   */
  @Option(
      name = "all feature flag values are present (internal)",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
        OptionEffectTag.AFFECTS_OUTPUTS,
        OptionEffectTag.BUILD_FILE_SEMANTICS,
        OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION,
        OptionEffectTag.LOADING_AND_ANALYSIS
      },
      metadataTags = {OptionMetadataTag.INTERNAL},
      defaultValue = "true")
  public boolean allFeatureFlagValuesArePresent = true;

  @Override
  public ConfigFeatureFlagOptions getHost() {
    ConfigFeatureFlagOptions host = (ConfigFeatureFlagOptions) super.getHost();
    host.enforceTransitiveConfigsForConfigFeatureFlag = false;
    host.allFeatureFlagValuesArePresent = this.allFeatureFlagValuesArePresent;
    return host;
  }
}
