// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.rules.genrule.GenRuleRule;
import com.google.devtools.build.lib.bazel.rules.sh.ShBinaryRule;
import com.google.devtools.build.lib.bazel.rules.sh.ShLibraryRule;
import com.google.devtools.build.lib.bazel.rules.sh.ShRuleClasses;
import com.google.devtools.build.lib.bazel.rules.sh.ShTestRule;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.rules.SkylarkRuleImplementationFunctions;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.view.BaseRuleClasses;
import com.google.devtools.build.lib.view.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.view.ConfiguredRuleClassProvider.PrerequisiteValidator;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.lib.view.config.FragmentOptions;
import com.google.devtools.build.lib.view.extra.ActionListenerRule;
import com.google.devtools.build.lib.view.extra.ExtraActionRule;
import com.google.devtools.build.lib.view.filegroup.FilegroupRule;
import com.google.devtools.build.lib.view.test.TestSuiteRule;

/**
 * A rule class provider implementing the rules Bazel knows.
 */
public class BazelRuleClassProvider {
  private static class BazelPrerequisiteValidator implements PrerequisiteValidator {
    @Override
    public void validate(RuleContext.Builder context,
        ConfiguredTarget prerequisite, Attribute attribute) {
      validateDirectPrerequisiteVisibility(context, prerequisite, attribute.getName());
    }

    private void validateDirectPrerequisiteVisibility(
        RuleContext.Builder context, ConfiguredTarget prerequisite, String attrName) {
      Rule rule = context.getRule();
      Target prerequisiteTarget = prerequisite.getTarget();
      Label prerequisiteLabel = prerequisiteTarget.getLabel();
      // We don't check the visibility of late-bound attributes, because it would break some
      // features.
      if (!context.getRule().getLabel().getPackageName().equals(
              prerequisite.getTarget().getLabel().getPackageName())
          && !context.isVisible(prerequisite)) {
        if (!context.getConfiguration().checkVisibility()) {
          context.ruleWarning(String.format("Target '%s' violates visibility of target "
              + "'%s'. Continuing because --nocheck_visibility is active",
              rule.getLabel(), prerequisiteLabel));
        } else {
          // Oddly enough, we use reportError rather than ruleError here.
          context.reportError(rule.getLocation(),
              String.format("Target '%s' is not visible from target '%s'. Check "
                  + "the visibility declaration of the former target if you think "
                  + "the dependency is legitimate",
                  prerequisiteLabel, rule.getLabel()));
        }
      }

      if (prerequisiteTarget instanceof PackageGroup) {
        if (!attrName.equals("visibility")) {
          context.reportError(rule.getAttributeLocation(attrName),
              "in " + attrName + " attribute of " + rule.getRuleClass()
              + " rule " + rule.getLabel() +  ": package group '"
              + prerequisiteLabel + "' is misplaced here "
              + "(they are only allowed in the visibility attribute)");
        }
      }
    }
  }

  public static ConfiguredRuleClassProvider create() {
    ConfiguredRuleClassProvider.Builder builder =
        new ConfiguredRuleClassProvider.Builder();
    setup(builder);
    return builder.build();
  }

  /**
   * List of all build option classes in Blaze.
   */
  // TODO(bazel-team): make this private, remove from tests, then BuildOptions.of can be merged
  // into RuleClassProvider.
  @VisibleForTesting
  @SuppressWarnings("unchecked")
  public static final ImmutableList<Class<? extends FragmentOptions>> BUILD_OPTIONS =
      ImmutableList.<Class<? extends FragmentOptions>>of(
          BuildConfiguration.Options.class
      );

  /**
   * Java objects accessible from Skylark rule implementations using this module.
   */
  private static final ImmutableMap<String, SkylarkType> skylarkBuiltinJavaObects =
      ImmutableMap.<String, SkylarkType>of();

  public static void setup(ConfiguredRuleClassProvider.Builder builder) {
    builder
        .setConfigurationCollectionFactory(new BazelConfigurationCollection())
        .setPrerequisiteValidator(new BazelPrerequisiteValidator())
        .setSkylarkValidationEnvironment(
            SkylarkRuleImplementationFunctions.getValidationEnvironment(skylarkBuiltinJavaObects))
        .setSkylarkAccessibleJavaClasses(skylarkBuiltinJavaObects);

    for (Class<? extends FragmentOptions> fragmentOptions : BUILD_OPTIONS) {
      builder.addConfigurationOptions(fragmentOptions);
    }

    builder.addRuleDefinition(BaseRuleClasses.BaseRule.class);
    builder.addRuleDefinition(BaseRuleClasses.RuleBase.class);
    builder.addRuleDefinition(BazelBaseRuleClasses.BinaryBaseRule.class);
    builder.addRuleDefinition(BaseRuleClasses.TestBaseRule.class);
    builder.addRuleDefinition(BazelBaseRuleClasses.BaselineCoverageRule.class);
    builder.addRuleDefinition(BazelBaseRuleClasses.ErrorRule.class);
    builder.addRuleDefinition(FilegroupRule.class);
    builder.addRuleDefinition(TestSuiteRule.class);
    builder.addRuleDefinition(GenRuleRule.class);
    builder.addRuleDefinition(ShRuleClasses.ShRule.class);
    builder.addRuleDefinition(ShLibraryRule.class);
    builder.addRuleDefinition(ShBinaryRule.class);
    builder.addRuleDefinition(ShTestRule.class);
    builder.addRuleDefinition(ExtraActionRule.class);
    builder.addRuleDefinition(ActionListenerRule.class);

    builder.addConfigurationFragment(new BazelConfiguration.Loader());
  }
}
