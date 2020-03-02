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
package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.analysis.AliasProvider.TargetMode;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.PrerequisiteValidator;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.FunctionSplitTransitionWhitelist;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;

/**
 * A base implementation of {@link PrerequisiteValidator} that performs common checks based on
 * definitions of what is considered the same logical package and what is considered "experimental"
 * code, which has may have relaxed checks for visibility and deprecation.
 */
public abstract class CommonPrerequisiteValidator implements PrerequisiteValidator {
  @Override
  public void validate(
      RuleContext.Builder contextBuilder,
      ConfiguredTargetAndData prerequisite,
      Attribute attribute) {
    validateDirectPrerequisiteLocation(contextBuilder, prerequisite);
    validateDirectPrerequisiteVisibility(contextBuilder, prerequisite, attribute);
    validateDirectPrerequisiteForTestOnly(contextBuilder, prerequisite);
    validateDirectPrerequisiteForDeprecation(
        contextBuilder, contextBuilder.getRule(), prerequisite, contextBuilder.forAspect());
  }

  /**
   * Returns whether two packages are considered the same for purposes of deprecation warnings.
   * Dependencies within the same package do not print deprecation warnings; a package in the
   * javatests directory may also depend on its corresponding java package without a warning.
   */
  public abstract boolean isSameLogicalPackage(
      PackageIdentifier thisPackage, PackageIdentifier prerequisitePackage);

  /**
   * Returns whether a package is considered experimental. Packages outside of experimental may not
   * depend on packages that are experimental.
   */
  protected abstract boolean packageUnderExperimental(PackageIdentifier packageIdentifier);

  protected abstract boolean checkVisibilityForExperimental(RuleContext.Builder context);

  protected abstract boolean allowExperimentalDeps(RuleContext.Builder context);

  private void validateDirectPrerequisiteVisibility(
      RuleContext.Builder context, ConfiguredTargetAndData prerequisite, Attribute attribute) {
    String attrName = attribute.getName();
    Rule rule = context.getRule();
    Target prerequisiteTarget = prerequisite.getTarget();

    // We don't check the visibility of late-bound attributes, because it would break some
    // features.
    if (!isSameLogicalPackage(
            rule.getLabel().getPackageIdentifier(),
            AliasProvider.getDependencyLabel(prerequisite.getConfiguredTarget())
                .getPackageIdentifier())
        && !Attribute.isLateBound(attrName)) {

      // Determine if we should use the new visibility rules for tools.
      boolean toolCheckAtDefinition = false;
      try {
        toolCheckAtDefinition =
            context.getStarlarkSemantics().incompatibleVisibilityPrivateAttributesAtDefinition();
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }

      if (!toolCheckAtDefinition
          || !attribute.isImplicit()
          || rule.getRuleClassObject().getRuleDefinitionEnvironmentLabel() == null) {
        // Default check: The attribute must be visible from the target.
        if (!context.isVisible(prerequisite.getConfiguredTarget())) {
          handleVisibilityConflict(context, prerequisite, rule.getLabel());
        }
      } else {
        // For implicit attributes, check if the prerequisite is visible from the location of the
        // rule definition
        Label implicitDefinition = rule.getRuleClassObject().getRuleDefinitionEnvironmentLabel();
        if (!RuleContext.isVisible(implicitDefinition, prerequisite.getConfiguredTarget())) {
          handleVisibilityConflict(context, prerequisite, implicitDefinition);
        }
      }
    }

    if (prerequisiteTarget instanceof PackageGroup) {
      boolean containsPackageSpecificationProvider =
          RawAttributeMapper.of(rule)
              .getAttributeDefinition(attrName)
              .getRequiredProviders()
              .getDescription()
              .contains("PackageSpecificationProvider");
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
                + " is misplaced here (they are only allowed in the visibility attribute)");
      }
    }
  }

  private void handleVisibilityConflict(
      RuleContext.Builder context, ConfiguredTargetAndData prerequisite, Label rule) {
    if (packageUnderExperimental(rule.getPackageIdentifier())
        && !checkVisibilityForExperimental(context)) {
      return;
    }

    if (!context.getConfiguration().checkVisibility()) {
      String errorMessage =
          String.format(
              "Target '%s' violates visibility of "
                  + "%s. Continuing because --nocheck_visibility is active",
              rule, AliasProvider.describeTargetWithAliases(prerequisite, TargetMode.WITHOUT_KIND));
      context.ruleWarning(errorMessage);
    } else {
      String errorMessage =
          String.format(
              "%s is not visible from target '%s'. Check "
                  + "the visibility declaration of the former target if you think "
                  + "the dependency is legitimate",
              AliasProvider.describeTargetWithAliases(prerequisite, TargetMode.WITHOUT_KIND), rule);

      if (prerequisite.getTarget().getTargetKind().equals(InputFile.targetKind())) {
        errorMessage +=
            ". To set the visibility of that source file target, use the exports_files() function";
      }
      context.ruleError(errorMessage);
    }
  }

  private void validateDirectPrerequisiteLocation(
      RuleContext.Builder context, ConfiguredTargetAndData prerequisite) {
    Rule rule = context.getRule();
    Target prerequisiteTarget = prerequisite.getTarget();
    Label prerequisiteLabel = prerequisiteTarget.getLabel();

    if (packageUnderExperimental(prerequisiteLabel.getPackageIdentifier())
        && !packageUnderExperimental(rule.getLabel().getPackageIdentifier())) {
      String message =
          "non-experimental target '"
              + rule.getLabel()
              + "' depends on experimental target '"
              + prerequisiteLabel
              + "'";
      if (allowExperimentalDeps(context)) {
        context.ruleWarning(
            message + " (ignored due to --experimental_deps_ok;" + " do not submit)");
      } else {
        context.ruleError(
            message
                + " (you may not check in such a dependency,"
                + " though you can test "
                + "against it by passing --experimental_deps_ok)");
      }
    }
  }

  /** Returns whether a deprecation warning should be printed for the prerequisite described. */
  private boolean shouldEmitDeprecationWarningFor(
      String thisDeprecation,
      PackageIdentifier thisPackage,
      String prerequisiteDeprecation,
      PackageIdentifier prerequisitePackage,
      boolean forAspect) {
    // Don't report deprecation edges from javatests to java or within a package;
    // otherwise tests of deprecated code generate nuisance warnings.
    // Don't report deprecation if the current target is also deprecated,
    // or if the current context is evaluating an aspect,
    // as the base target would have already printed the deprecation warnings.
    return (!forAspect
        && prerequisiteDeprecation != null
        && !isSameLogicalPackage(thisPackage, prerequisitePackage)
        && thisDeprecation == null);
  }

  /** Checks if the given prerequisite is deprecated and prints a warning if so. */
  private void validateDirectPrerequisiteForDeprecation(
      RuleErrorConsumer errors,
      Rule rule,
      ConfiguredTargetAndData prerequisite,
      boolean forAspect) {
    Target prerequisiteTarget = prerequisite.getTarget();
    Label prerequisiteLabel = prerequisiteTarget.getLabel();
    PackageIdentifier thatPackage = prerequisiteLabel.getPackageIdentifier();
    PackageIdentifier thisPackage = rule.getLabel().getPackageIdentifier();

    if (prerequisiteTarget instanceof Rule) {
      Rule prerequisiteRule = (Rule) prerequisiteTarget;
      String thisDeprecation =
          NonconfigurableAttributeMapper.of(rule).has("deprecation", Type.STRING)
              ? NonconfigurableAttributeMapper.of(rule).get("deprecation", Type.STRING)
              : null;
      String thatDeprecation =
          NonconfigurableAttributeMapper.of(prerequisiteRule).has("deprecation", Type.STRING)
              ? NonconfigurableAttributeMapper.of(prerequisiteRule).get("deprecation", Type.STRING)
              : null;
      if (shouldEmitDeprecationWarningFor(
          thisDeprecation, thisPackage, thatDeprecation, thatPackage, forAspect)) {
        errors.ruleWarning(
            "target '"
                + rule.getLabel()
                + "' depends on deprecated target '"
                + prerequisiteLabel
                + "': "
                + thatDeprecation);
      }
    }

    if (prerequisiteTarget instanceof OutputFile) {
      Rule generatingRule = ((OutputFile) prerequisiteTarget).getGeneratingRule();
      String thisDeprecation =
          NonconfigurableAttributeMapper.of(rule).get("deprecation", Type.STRING);
      String thatDeprecation =
          NonconfigurableAttributeMapper.of(generatingRule).get("deprecation", Type.STRING);
      if (shouldEmitDeprecationWarningFor(
          thisDeprecation, thisPackage, thatDeprecation, thatPackage, forAspect)) {
        errors.ruleWarning(
            "target '"
                + rule.getLabel()
                + "' depends on the output file "
                + prerequisiteLabel
                + " of a deprecated rule "
                + generatingRule.getLabel()
                + "': "
                + thatDeprecation);
      }
    }
  }

  /** Check that the dependency is not test-only, or the current rule is test-only. */
  private void validateDirectPrerequisiteForTestOnly(
      RuleContext.Builder context, ConfiguredTargetAndData prerequisite) {
    Rule rule = context.getRule();

    if (rule.getRuleClassObject().getAdvertisedProviders().canHaveAnyProvider()) {
      // testonly-ness will be checked directly between the depender and the target of the alias;
      // getTarget() called by the depender will not return the alias rule, but its actual target
      return;
    }

    Target prerequisiteTarget = prerequisite.getTarget();
    PackageIdentifier thisPackage = rule.getLabel().getPackageIdentifier();

    if (isTestOnlyRule(prerequisiteTarget) && !isTestOnlyRule(rule)) {
      String message =
          "non-test target '"
              + rule.getLabel()
              + "' depends on testonly "
              + AliasProvider.describeTargetWithAliases(prerequisite, TargetMode.WITHOUT_KIND)
              + " and doesn't have testonly attribute set";
      if (packageUnderExperimental(thisPackage)) {
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
