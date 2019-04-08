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

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static java.util.Objects.requireNonNull;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.PackageSpecification.PackageGroupContents;

/**
 * Class used for implementing whitelists using package groups.
 *
 * <p>To use add an attribute {@link getAttributeFromWhitelistName(String,Label) to the rule class
 * which needs the whitelisting mechanism and use {@link isAvailable(RuleContext,String)} to check
 * during analysis if a rule is present
 */
public final class Whitelist {

  private Whitelist() {}

  /**
   * Returns an Attribute.Builder that can be used to add an implicit attribute to a rule containing
   * a package group whitelist.
   *
   * @param whitelistName The name of the whitelist. This has to comply with attribute naming
   *     standards and will be used as a suffix for the attribute name.
   */
  public static Attribute.Builder<Label> getAttributeFromWhitelistName(String whitelistName) {
    String attributeName = getAttributeNameFromWhitelistName(whitelistName);
    return attr(attributeName, LABEL)
        .cfg(HostTransition.createFactory())
        .mandatoryNativeProviders(ImmutableList.of(PackageSpecificationProvider.class));
  }

  /**
   * Returns whether the rule in the given RuleContext *was defined* in a whitelist.
   *
   * @param ruleContext The context in which this check is being executed.
   * @param whitelistName The name of the whitelist being used.
   */
  public static boolean isAvailableBasedOnRuleLocation(
      RuleContext ruleContext, String whitelistName) {
    return isAvailableFor(
        ruleContext,
        whitelistName,
        ruleContext.getRule().getRuleClassObject().getRuleDefinitionEnvironmentLabel());
  }

  /**
   * Returns whether the rule in the given RuleContext *was instantiated* in a whitelist.
   *
   * @param ruleContext The context in which this check is being executed.
   * @param whitelistName The name of the whitelist being used.
   */
  public static boolean isAvailable(RuleContext ruleContext, String whitelistName) {
    return isAvailableFor(ruleContext, whitelistName, ruleContext.getLabel());
  }

  /**
   * @param relevantLabel the label to check for in the whitelist. This allows features that
   *     whitelist on rule definition location and features that whitelist on rule instantiation
   *     location to share logic.
   */
  private static boolean isAvailableFor(
      RuleContext ruleContext, String whitelistName, Label relevantLabel) {
    PackageSpecificationProvider packageSpecificationProvider =
        fetchPackageSpecificationProvider(ruleContext, whitelistName);
    return isAvailableFor(packageSpecificationProvider.getPackageSpecifications(), relevantLabel);
  }

  public static PackageSpecificationProvider fetchPackageSpecificationProvider(
      RuleContext ruleContext, String whitelistName) {
    String attributeName = getAttributeNameFromWhitelistName(whitelistName);
    Preconditions.checkArgument(ruleContext.isAttrDefined(attributeName, LABEL));
    TransitiveInfoCollection packageGroup = ruleContext.getPrerequisite(attributeName, Mode.HOST);
    PackageSpecificationProvider packageSpecificationProvider =
        packageGroup.getProvider(PackageSpecificationProvider.class);
    return requireNonNull(packageSpecificationProvider, packageGroup.getLabel().toString());
  }

  public static boolean isAvailableFor(
      Iterable<PackageGroupContents> packageGroupContents, Label relevantLabel) {
    return Streams.stream(packageGroupContents)
        .anyMatch(p -> p.containsPackage(relevantLabel.getPackageIdentifier()));
  }

  /**
   * Returns whether the rule from the given rule context has a whitelist by the given name.
   *
   * @param ruleContext The rule context to check
   * @param whitelistName The name of the whitelist to check for.
   * @return True if the given rule context has the given whitelist.
   */
  public static boolean hasWhitelist(RuleContext ruleContext, String whitelistName) {
    String attributeName = getAttributeNameFromWhitelistName(whitelistName);
    return ruleContext.isAttrDefined(attributeName, LABEL);
  }

  private static String getAttributeNameFromWhitelistName(String whitelistName) {
    return String.format("$whitelist_%s", whitelistName);
  }
}
