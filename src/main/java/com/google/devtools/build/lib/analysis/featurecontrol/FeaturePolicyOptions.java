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
// limitations under the License.

package com.google.devtools.build.lib.analysis.featurecontrol;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsParser.OptionUsageRestrictions;
import java.util.List;

/** The options fragment which defines {@link FeaturePolicyConfiguration}. */
public final class FeaturePolicyOptions extends FragmentOptions {
  /** The mapping from features to their associated package groups. */
  @Option(
    name = "feature_control_policy",
    help =
        "Policy used to limit the rollout or deprecation of features within the Bazel binary to "
            + "specific packages. Pass a mapping from a feature name to the package group used to "
            + "control access to that feature, in the form feature=//label/of:package_group (note "
            + "that unlike visibility,  packages cannot be directly specified a la "
            + "//package:__pkg__ or //visibility:public). Can be repeated to specify multiple "
            + "features, but each feature must be specified only once.",
    valueHelp = "a feature=label pair",
    optionUsageRestrictions = OptionUsageRestrictions.UNDOCUMENTED,
    converter = PolicyEntryConverter.class,
    defaultValue = "n/a (default ignored for allowMultiple)",
    allowMultiple = true
  )
  public List<PolicyEntry> policies = ImmutableList.<PolicyEntry>of();

  @Override
  public FeaturePolicyOptions getHost(boolean fallback) {
    // host options are the same as target options
    return (FeaturePolicyOptions) clone();
  }
}
