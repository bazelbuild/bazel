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

package com.google.devtools.build.lib.analysis;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static java.util.Objects.requireNonNull;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.ExecutionTransitionFactory;
import com.google.devtools.build.lib.analysis.configuredtargets.PackageGroupConfiguredTarget;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.PackageSpecification.PackageGroupContents;
import javax.annotation.Nullable;

/**
 * Class used for implementing allowlists using package groups.
 *
 * <p>To use add an attribute {@link getAttributeFromAllowlistName(String) to the rule class
 * which needs the allowlisting mechanism and use {@link isAvailable(RuleContext,String)} to check
 * during analysis if a rule is present
 */
public final class Allowlist {

  private Allowlist() {}

  /**
   * Returns an Attribute.Builder that can be used to add an implicit attribute to a rule containing
   * a package group allowlist.
   *
   * @param allowlistName The name of the allowlist. This has to comply with attribute naming
   *     standards and will be used as a suffix for the attribute name.
   */
  public static Attribute.Builder<Label> getAttributeFromAllowlistName(String allowlistName) {
    String attributeName = getAttributeNameFromAllowlistName(allowlistName).iterator().next();
    return attr(attributeName, LABEL)
        .cfg(ExecutionTransitionFactory.createFactory())
        .mandatoryProviders(PackageGroupConfiguredTarget.PROVIDER.id());
  }

  /**
   * Returns whether the rule in the given RuleContext *was defined* in a allowlist.
   *
   * @param ruleContext The context in which this check is being executed.
   * @param allowlistName The name of the allowlist being used.
   */
  public static boolean isAvailableBasedOnRuleLocation(
      RuleContext ruleContext, String allowlistName) {
    return isAvailableFor(
        ruleContext,
        allowlistName,
        ruleContext.getRule().getRuleClassObject().getRuleDefinitionEnvironmentLabel());
  }

  /**
   * Returns whether the rule in the given RuleContext *was instantiated* in a allowlist.
   *
   * @param ruleContext The context in which this check is being executed.
   * @param allowlistName The name of the allowlist being used.
   */
  public static boolean isAvailable(RuleContext ruleContext, String allowlistName) {
    return isAvailableFor(ruleContext, allowlistName, ruleContext.getLabel());
  }

  /**
   * @param ruleContext The context in which this check is being executed.
   * @param allowlistName The name of the allowlist being used.
   * @param relevantLabel The label to check for in the allowlist. This allows features that
   *     allowlist on rule definition location and features that allowlist on rule instantiation
   *     location to share logic.
   */
  public static boolean isAvailableFor(
      RuleContext ruleContext, String allowlistName, Label relevantLabel) {
    PackageSpecificationProvider packageSpecificationProvider =
        fetchPackageSpecificationProvider(ruleContext, allowlistName);
    return isAvailableFor(packageSpecificationProvider.getPackageSpecifications(), relevantLabel);
  }

  public static boolean isAvailableFor(
      NestedSet<PackageGroupContents> packageGroupContents, Label relevantLabel) {
    return packageGroupContents.toList().stream()
        .anyMatch(p -> p.containsPackage(relevantLabel.getPackageIdentifier()));
  }

  public static PackageSpecificationProvider fetchPackageSpecificationProvider(
      RuleContext ruleContext, String allowlistName) {
    return checkNotNull(
        fetchPackageSpecificationProviderOrNull(ruleContext, allowlistName),
        "Allowlist argument for %s not found",
        allowlistName);
  }

  @Nullable
  public static PackageSpecificationProvider fetchPackageSpecificationProviderOrNull(
      RuleContext ruleContext, String allowlistName) {
    for (String attributeName : getAttributeNameFromAllowlistName(allowlistName)) {
      if (!ruleContext.isAttrDefined(attributeName, LABEL)) {
        continue;
      }
      Preconditions.checkArgument(ruleContext.isAttrDefined(attributeName, LABEL), attributeName);
      TransitiveInfoCollection packageGroup = ruleContext.getPrerequisite(attributeName);
      PackageSpecificationProvider packageSpecificationProvider =
          packageGroup.get(PackageGroupConfiguredTarget.PROVIDER);
      return requireNonNull(packageSpecificationProvider, packageGroup.getLabel().toString());
    }
    return null;
  }

  /**
   * Returns whether the rule from the given rule context has a allowlist by the given name.
   *
   * @param ruleContext The rule context to check
   * @param allowlistName The name of the allowlist to check for.
   * @return True if the given rule context has the given allowlist.
   */
  public static boolean hasAllowlist(RuleContext ruleContext, String allowlistName) {
    for (String attributeName : getAttributeNameFromAllowlistName(allowlistName)) {
      if (ruleContext.isAttrDefined(attributeName, LABEL)) {
        return true;
      }
    }
    return false;
  }

  private static ImmutableList<String> getAttributeNameFromAllowlistName(String allowlistName) {
    return ImmutableList.of(
        String.format("$whitelist_%s", allowlistName),
        String.format("$allowlist_%s", allowlistName));
  }
}
