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
import com.google.common.base.Functions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.PrerequisiteValidator;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigRuleClasses;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.constraints.EnvironmentRule;
import com.google.devtools.build.lib.bazel.rules.common.BazelActionListenerRule;
import com.google.devtools.build.lib.bazel.rules.common.BazelExtraActionRule;
import com.google.devtools.build.lib.bazel.rules.common.BazelFilegroupRule;
import com.google.devtools.build.lib.bazel.rules.common.BazelTestSuiteRule;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCppRuleClasses;
import com.google.devtools.build.lib.bazel.rules.genrule.BazelGenRuleRule;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaBinaryRule;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaBuildInfoFactory;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaImportRule;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaLibraryRule;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaPluginRule;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaRuleClasses;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaTestRule;
import com.google.devtools.build.lib.bazel.rules.objc.BazelIosTestRule;
import com.google.devtools.build.lib.bazel.rules.sh.BazelShBinaryRule;
import com.google.devtools.build.lib.bazel.rules.sh.BazelShLibraryRule;
import com.google.devtools.build.lib.bazel.rules.sh.BazelShRuleClasses;
import com.google.devtools.build.lib.bazel.rules.sh.BazelShTestRule;
import com.google.devtools.build.lib.bazel.rules.workspace.HttpArchiveRule;
import com.google.devtools.build.lib.bazel.rules.workspace.HttpJarRule;
import com.google.devtools.build.lib.bazel.rules.workspace.LocalRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.workspace.MavenJarRule;
import com.google.devtools.build.lib.bazel.rules.workspace.NewLocalRepositoryRule;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.rules.cpp.CcToolchainRule;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfigurationLoader;
import com.google.devtools.build.lib.rules.cpp.CppOptions;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaConfigurationLoader;
import com.google.devtools.build.lib.rules.java.JavaCpuSupplier;
import com.google.devtools.build.lib.rules.java.JavaImportBaseRule;
import com.google.devtools.build.lib.rules.java.JavaOptions;
import com.google.devtools.build.lib.rules.java.JavaToolchainRule;
import com.google.devtools.build.lib.rules.java.Jvm;
import com.google.devtools.build.lib.rules.java.JvmConfigurationLoader;
import com.google.devtools.build.lib.rules.objc.IosApplicationRule;
import com.google.devtools.build.lib.rules.objc.IosDeviceRule;
import com.google.devtools.build.lib.rules.objc.IosExtensionBinaryRule;
import com.google.devtools.build.lib.rules.objc.ObjcBinaryRule;
import com.google.devtools.build.lib.rules.objc.ObjcBundleLibraryRule;
import com.google.devtools.build.lib.rules.objc.ObjcBundleRule;
import com.google.devtools.build.lib.rules.objc.ObjcCommandLineOptions;
import com.google.devtools.build.lib.rules.objc.ObjcConfigurationLoader;
import com.google.devtools.build.lib.rules.objc.ObjcFrameworkRule;
import com.google.devtools.build.lib.rules.objc.ObjcImportRule;
import com.google.devtools.build.lib.rules.objc.ObjcLibraryRule;
import com.google.devtools.build.lib.rules.objc.ObjcOptionsRule;
import com.google.devtools.build.lib.rules.objc.ObjcProtoLibraryRule;
import com.google.devtools.build.lib.rules.objc.ObjcRuleClasses;
import com.google.devtools.build.lib.rules.objc.ObjcXcodeprojRule;
import com.google.devtools.build.lib.rules.workspace.BindRule;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.SkylarkType;

/**
 * A rule class provider implementing the rules Bazel knows.
 */
public class BazelRuleClassProvider {

  /**
   * Used by the build encyclopedia generator.
   */
  public static ConfiguredRuleClassProvider create() {
    ConfiguredRuleClassProvider.Builder builder =
        new ConfiguredRuleClassProvider.Builder();
    setup(builder);
    return builder.build();
  }

  public static final JavaCpuSupplier JAVA_CPU_SUPPLIER = new JavaCpuSupplier() {
    @Override
    public String getJavaCpu(BuildOptions buildOptions, ConfigurationEnvironment env)
        throws InvalidConfigurationException {
      JavaOptions javaOptions = buildOptions.get(JavaOptions.class);
      return javaOptions.javaCpu == null ? "default" : javaOptions.javaCpu;
    }
  };

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

  /**
   * List of all build option classes in Blaze.
   */
  // TODO(bazel-team): make this private, remove from tests, then BuildOptions.of can be merged
  // into RuleClassProvider.
  @VisibleForTesting
  @SuppressWarnings("unchecked")
  public static final ImmutableList<Class<? extends FragmentOptions>> BUILD_OPTIONS =
      ImmutableList.of(
          BuildConfiguration.Options.class,
          CppOptions.class,
          JavaOptions.class,
          ObjcCommandLineOptions.class
      );

  /**
   * Java objects accessible from Skylark rule implementations using this module.
   */
  private static final ImmutableMap<String, SkylarkType> skylarkBuiltinJavaObects =
      ImmutableMap.of(
          "jvm", SkylarkType.of(Jvm.class),
          "java_configuration", SkylarkType.of(JavaConfiguration.class),
          "cpp", SkylarkType.of(CppConfiguration.class));

  public static void setup(ConfiguredRuleClassProvider.Builder builder) {
    builder
        .addBuildInfoFactory(new BazelJavaBuildInfoFactory())
        .setConfigurationCollectionFactory(new BazelConfigurationCollection())
        .setPrerequisiteValidator(new BazelPrerequisiteValidator())
        .setSkylarkAccessibleJavaClasses(skylarkBuiltinJavaObects);

    for (Class<? extends FragmentOptions> fragmentOptions : BUILD_OPTIONS) {
      builder.addConfigurationOptions(fragmentOptions);
    }

    builder.addRuleDefinition(BaseRuleClasses.BaseRule.class);
    builder.addRuleDefinition(BaseRuleClasses.RuleBase.class);
    builder.addRuleDefinition(BazelBaseRuleClasses.BinaryBaseRule.class);
    builder.addRuleDefinition(BaseRuleClasses.TestBaseRule.class);
    builder.addRuleDefinition(BazelBaseRuleClasses.ErrorRule.class);

    builder.addRuleDefinition(EnvironmentRule.class);

    builder.addRuleDefinition(ConfigRuleClasses.ConfigBaseRule.class);
    builder.addRuleDefinition(ConfigRuleClasses.ConfigSettingRule.class);

    builder.addRuleDefinition(BazelFilegroupRule.class);
    builder.addRuleDefinition(BazelTestSuiteRule.class);
    builder.addRuleDefinition(BazelGenRuleRule.class);

    builder.addRuleDefinition(BazelShRuleClasses.ShRule.class);
    builder.addRuleDefinition(BazelShLibraryRule.class);
    builder.addRuleDefinition(BazelShBinaryRule.class);
    builder.addRuleDefinition(BazelShTestRule.class);

    builder.addRuleDefinition(CcToolchainRule.class);
    builder.addRuleDefinition(BazelCppRuleClasses.CcLinkingRule.class);
    builder.addRuleDefinition(BazelCppRuleClasses.CcDeclRule.class);
    builder.addRuleDefinition(BazelCppRuleClasses.CcBaseRule.class);
    builder.addRuleDefinition(BazelCppRuleClasses.CcRule.class);
    builder.addRuleDefinition(BazelCppRuleClasses.CcBinaryBaseRule.class);
    builder.addRuleDefinition(BazelCppRuleClasses.CcBinaryRule.class);
    builder.addRuleDefinition(BazelCppRuleClasses.CcTestRule.class);

    builder.addRuleDefinition(BazelCppRuleClasses.CcLibraryBaseRule.class);
    builder.addRuleDefinition(BazelCppRuleClasses.CcLibraryRule.class);

    builder.addWorkspaceFile(BazelJavaRuleClasses.getDefaultWorkspace());
    builder.addRuleDefinition(BazelJavaRuleClasses.BaseJavaBinaryRule.class);
    builder.addRuleDefinition(BazelJavaRuleClasses.IjarBaseRule.class);
    builder.addRuleDefinition(BazelJavaRuleClasses.JavaBaseRule.class);
    builder.addRuleDefinition(JavaImportBaseRule.class);
    builder.addRuleDefinition(BazelJavaRuleClasses.JavaRule.class);
    builder.addRuleDefinition(BazelJavaBinaryRule.class);
    builder.addRuleDefinition(BazelJavaLibraryRule.class);
    builder.addRuleDefinition(BazelJavaImportRule.class);
    builder.addRuleDefinition(BazelJavaTestRule.class);
    builder.addRuleDefinition(BazelJavaPluginRule.class);
    builder.addRuleDefinition(JavaToolchainRule.class);

    builder.addRuleDefinition(BazelIosTestRule.class);
    builder.addRuleDefinition(IosDeviceRule.class);
    builder.addRuleDefinition(ObjcBinaryRule.class);
    builder.addRuleDefinition(ObjcBundleRule.class);
    builder.addRuleDefinition(ObjcBundleLibraryRule.class);
    builder.addRuleDefinition(ObjcFrameworkRule.class);
    builder.addRuleDefinition(ObjcImportRule.class);
    builder.addRuleDefinition(ObjcLibraryRule.class);
    builder.addRuleDefinition(ObjcOptionsRule.class);
    builder.addRuleDefinition(ObjcProtoLibraryRule.class);
    builder.addRuleDefinition(ObjcXcodeprojRule.class);
    builder.addRuleDefinition(ObjcRuleClasses.IosTestBaseRule.class);
    builder.addRuleDefinition(ObjcRuleClasses.BundlingRule.class);
    builder.addRuleDefinition(ObjcRuleClasses.ReleaseBundlingRule.class);
    builder.addRuleDefinition(ObjcRuleClasses.CompilingRule.class);
    builder.addRuleDefinition(ObjcRuleClasses.LinkingRule.class);
    builder.addRuleDefinition(ObjcRuleClasses.ResourcesRule.class);
    builder.addRuleDefinition(ObjcRuleClasses.XcodegenRule.class);
    builder.addRuleDefinition(ObjcRuleClasses.AlwaysLinkRule.class);
    builder.addRuleDefinition(ObjcRuleClasses.OptionsRule.class);
    builder.addRuleDefinition(ObjcRuleClasses.SdkFrameworksDependerRule.class);
    builder.addRuleDefinition(ObjcRuleClasses.CompileDependencyRule.class);
    builder.addRuleDefinition(ObjcRuleClasses.ResourceToolsRule.class);
    builder.addRuleDefinition(IosApplicationRule.class);
    builder.addRuleDefinition(IosExtensionBinaryRule.class);

    builder.addRuleDefinition(BazelExtraActionRule.class);
    builder.addRuleDefinition(BazelActionListenerRule.class);

    builder.addRuleDefinition(BindRule.class);
    builder.addRuleDefinition(HttpArchiveRule.class);
    builder.addRuleDefinition(HttpJarRule.class);
    builder.addRuleDefinition(LocalRepositoryRule.class);
    builder.addRuleDefinition(MavenJarRule.class);
    builder.addRuleDefinition(NewLocalRepositoryRule.class);

    builder.addConfigurationFragment(new BazelConfiguration.Loader());
    builder.addConfigurationFragment(new CppConfigurationLoader(
        Functions.<String>identity()));
    builder.addConfigurationFragment(new JvmConfigurationLoader(JAVA_CPU_SUPPLIER));
    builder.addConfigurationFragment(new JavaConfigurationLoader(JAVA_CPU_SUPPLIER));
    builder.addConfigurationFragment(new ObjcConfigurationLoader());
  }
}
