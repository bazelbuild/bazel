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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.PackageSpecification;
import com.google.devtools.build.lib.util.Preconditions;

/**
 * A configuration fragment which controls access to features being rolled out or deprecated,
 * allowing such features to be limited to particular packages to avoid excessive spread when
 * rolling out or deprecating a feature. It's controlled by {@link FeaturePolicyOptions}.
 *
 * <p>A "feature", for this package's purposes, is any combination of related functionality which
 * should be limited to specified packages. Because this is part of the configuration, it can only
 * be accessed during the analysis phase; decisions which are made during the loading phase can't
 * use this information. Some examples of use cases for this fragment:
 * <ul>
 *     <li>A new rule class which could have a major impact on Blaze's memory usage is added. To
 *         limit this impact during the experimental phase, a feature policy is added which makes it
 *         an error for rules of that class to be created - or used by other rules - in any package
 *         other than those defined by the policy. The policy is populated with projects who are
 *         doing guided experimentation with the feature, and gradually expands as the feature rolls
 *         out. Then the feature's policy can be removed, making it generally available.
 *     <li>An attribute is being deprecated. To prevent rollback, a feature policy is added which
 *         makes it an error for rules in packages other than those defined by the policy to specify
 *         a value for that attribute. The policy is populated with existing users, and as those
 *         users are migrated, they are removed from the policy until it is completely empty. Then
 *         the attribute can be removed entirely.
 * </ul>
 *
 * <p>To use this package:
 * <ol>
 *     <li>Define a feature ID in the {@link FeaturePolicyLoader}'s constructor (in the
 *         RuleClassProvider). This is the string that will be used when checking for the feature in
 *         rule code, as well as the string used in the flag value for {@link FeaturePolicyOptions}.
 *     <li>In the RuleClass(es) which will change based on the feature's state, declare
 *         {@link FeaturePolicyConfiguration} as a required configuration fragment.
 *     <li>In the ConfiguredTargetFactory of those rules, get the FeaturePolicyConfiguration and
 *         check {@link #isFeatureEnabledForRule(String,Label)} with the feature ID created in
 *         step 1 and the label of the current rule. In the event that an error needs to be
 *         displayed, use {@link #getPolicyForFeature(String)} to show the user where the policy is.
 *     <li>Create a package_group containing the list of packages which should have access to this
 *         feature. It can be empty (no packages can access the feature) or contain //... (all
 *         packages can access the feature) to begin with.
 *     <li>After the a release containing the feature ID has been pushed, update the global RC file
 *         with a --feature_control_policy=(your_feature)=(your_package_group) flag. You can now
 *         alter access to your feature by changing the package_group.
 * </ol>
 *
 * <p>To stop using this package:
 * <ol>
 *     <li>Your policy should be at an end state - containing all packages (a rollout which has
 *         become generally available) or no packages (a deprecated feature which has been totally
 *         cleaned up).
 *     <li>Make the behavior the policy controlled permanent - remove a deprecated feature, or
 *         remove the check on a feature which is being rolled out.
 *     <li>After this new version is released, remove the flag from the global rc file, and remove
 *         the feature ID from the constructor for {@link FeaturePolicyLoader}.
 * </ol>
 *
 * @see FeaturePolicyLoader
 */
public final class FeaturePolicyConfiguration extends BuildConfiguration.Fragment {

  private final ImmutableSetMultimap<String, PackageSpecification> features;
  private final ImmutableMap<String, String> policies;

  /**
   * Creates a new FeaturePolicyConfiguration.
   *
   * @param features Map mapping from a feature ID to the packages which are able to access it. If a
   *     feature ID is not present in this mapping, it is not accessible from any package.
   * @param policies Map mapping from a feature ID to a string policy to be shown to the user. A
   *     string must be present as a key in this mapping to be considered a valid feature ID.
   */
  public FeaturePolicyConfiguration(
      ImmutableSetMultimap<String, PackageSpecification> features,
      ImmutableMap<String, String> policies) {
    this.features = features;
    this.policies = policies;
  }

  /**
   * Returns whether, according to the current policy, the given feature is enabled for the given
   * rule.
   */
  public boolean isFeatureEnabledForRule(String feature, Label rule) {
    Preconditions.checkArgument(policies.containsKey(feature), "No such feature: %s", feature);
    ImmutableSet<PackageSpecification> result = features.get(feature);
    for (PackageSpecification spec : result) {
      if (spec.containsPackage(rule.getPackageIdentifier())) {
        return true;
      }
    }
    return false;
  }

  /**
   * Returns a String suitable for presenting to the user which represents the policy used for the
   * given feature.
   */
  public String getPolicyForFeature(String feature) {
    String result = policies.get(feature);
    Preconditions.checkArgument(result != null, "No such feature: %s", feature);
    return result;
  }
}
