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

import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.featurecontrol.FeaturePolicyConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;

/**
 * A validator utility class which confirms that config_feature_flag-related features can be used by
 * the current rule.
 */
public class ConfigFeatureFlagFeatureVisibility {

  /** The name of the policy that is used to restrict access to the config_feature_flag rule. */
  public static final String POLICY_NAME = "config_feature_flag";

  private ConfigFeatureFlagFeatureVisibility() {}

  /**
   * Checks whether the rule in the given RuleContext has access to config_feature_flag and related
   * features, and reports an error if not.
   *
   * @param ruleContext The context in which this check is being executed.
   * @param feature The name of the config_feature_flag-related feature being used, to be printed in
   *     the error if the policy forbids access to this rule.
   */
  public static void checkAvailable(RuleContext ruleContext, String feature)
      throws RuleErrorException {
    FeaturePolicyConfiguration policy = ruleContext.getFragment(FeaturePolicyConfiguration.class);
    Label label = ruleContext.getLabel();
    if (!policy.isFeatureEnabledForRule(POLICY_NAME, label)) {
      ruleContext.ruleError(
          String.format(
              "%s is not available in package '%s' according to policy '%s'",
              feature, label.getPackageIdentifier(), policy.getPolicyForFeature(POLICY_NAME)));
      throw new RuleErrorException();
    }
  }
}
