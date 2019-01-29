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

package com.google.devtools.build.lib.bazel.rules;

import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.AliasProvider.TargetMode;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.FunctionSplitTransitionWhitelist;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.RequiredProviders;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.syntax.Type;

/** Ensures that a target's prerequisites are visible to it and match its testonly status. */
public class BazelPrerequisiteValidator
    implements ConfiguredRuleClassProvider.PrerequisiteValidator {

  @Override
  public void validate(
      RuleContext.Builder context, ConfiguredTargetAndData prerequisite, Attribute attribute) {
    validateDirectPrerequisiteVisibility(context, prerequisite, attribute.getName());
    validateDirectPrerequisiteForTestOnly(context, prerequisite);
    ConfiguredRuleClassProvider.DeprecationValidator.validateDirectPrerequisiteForDeprecation(
        context, context.getRule(), prerequisite, context.forAspect());
  }

  private void validateDirectPrerequisiteVisibility(
      RuleContext.Builder context, ConfiguredTargetAndData prerequisite, String attrName) {
    Rule rule = context.getRule();
    Target prerequisiteTarget = prerequisite.getTarget();
    if (!context
            .getRule()
            .getLabel()
            .getPackageIdentifier()
            .equals(
                AliasProvider.getDependencyLabel(prerequisite.getConfiguredTarget())
                    .getPackageIdentifier())
        && !context.isVisible(prerequisite.getConfiguredTarget())) {
      String errorMessage;
      if (!context.getConfiguration().checkVisibility()) {
        errorMessage =
            String.format(
                "Target '%s' violates visibility of "
                    + "%s. Continuing because --nocheck_visibility is active",
                rule.getLabel(), AliasProvider.describeTargetWithAliases(prerequisite,
                    TargetMode.WITHOUT_KIND));
        context.ruleWarning(errorMessage);
      } else {
        // Oddly enough, we use reportError rather than ruleError here.
        errorMessage =
            String.format(
                "%s is not visible from target '%s'. Check "
                    + "the visibility declaration of the former target if you think "
                    + "the dependency is legitimate",
                AliasProvider.describeTargetWithAliases(prerequisite, TargetMode.WITHOUT_KIND),
                rule.getLabel());
        context.ruleError(errorMessage);
      }
      // We can always post the visibility error as, regardless of the value of keep going,
      // that target will not be built.
      context.post(
          new VisibilityErrorEvent(context.getConfiguration(), rule.getLabel(), errorMessage));
    }

    if (prerequisiteTarget instanceof PackageGroup) {
      RequiredProviders requiredProviders =
          RawAttributeMapper.of(rule).getAttributeDefinition(attrName).getRequiredProviders();
      boolean containsPackageSpecificationProvider =
          requiredProviders.getDescription().contains("PackageSpecificationProvider");
      // TODO(plf): Add the PackageSpecificationProvider to the 'visibility' attribute.
      if (!attrName.equals("visibility")
          && !attrName.equals(FunctionSplitTransitionWhitelist.WHITELIST_ATTRIBUTE_NAME)
          && !containsPackageSpecificationProvider) {
        context.attributeError(
            attrName,
            "in "
                + attrName
                + " attribute of "
                + rule.getRuleClass()
                + " rule "
                + rule.getLabel()
                + ": "
                + AliasProvider.describeTargetWithAliases(prerequisite, TargetMode.WITH_KIND)
                + " is misplaced here "
                + "(they are only allowed in the visibility attribute)");
      }
    }
  }

  private void validateDirectPrerequisiteForTestOnly(
      RuleContext.Builder context, ConfiguredTargetAndData prerequisite) {
    Rule rule = context.getRule();

    if (rule.getRuleClassObject().getAdvertisedProviders().canHaveAnyProvider()) {
      // testonly-ness will be checked directly between the depender and the target of the alias;
      // getTarget() called by the depender will not return the alias rule, but its actual target
      return;
    }

    Target prerequisiteTarget = prerequisite.getTarget();
    String thisPackage = rule.getLabel().getPackageName();

    if (isTestOnlyRule(prerequisiteTarget) && !isTestOnlyRule(rule)) {
      String message =
          "non-test target '"
              + rule.getLabel()
              + "' depends on testonly "
              + AliasProvider.describeTargetWithAliases(prerequisite, TargetMode.WITHOUT_KIND)
              + " and doesn't have testonly attribute set";
      if (thisPackage.startsWith("experimental/")) {
        context.ruleWarning(message);
      } else {
        context.ruleError(message);
      }
    }
  }

  private static boolean isTestOnlyRule(Target target) {
    return (target instanceof Rule)
        && NonconfigurableAttributeMapper.of((Rule) target).has("testonly", Type.BOOLEAN)
        && NonconfigurableAttributeMapper.of((Rule) target).get("testonly", Type.BOOLEAN);
  }
}
