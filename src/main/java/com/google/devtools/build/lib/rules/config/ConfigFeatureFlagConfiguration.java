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
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.RequiresOptions;

/**
 * Configuration fragment for Android's config_feature_flag, flags which can be defined in BUILD
 * files. This exists only so that ConfigFeatureFlagOptions.class is retained.
 */
@RequiresOptions(options = ConfigFeatureFlagOptions.class, starlark = true)
public final class ConfigFeatureFlagConfiguration extends Fragment {
  /** Creates a new configuration fragment from the given {@link ConfigFeatureFlagOptions}. */
  public ConfigFeatureFlagConfiguration(BuildOptions buildOptions) {}

  @VisibleForTesting
  ConfigFeatureFlagConfiguration() {}
}
