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
import com.google.devtools.build.lib.bazel.rules.android.AndroidNdkRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.android.AndroidSdkRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.android.BazelAndroidBinaryRule;
import com.google.devtools.build.lib.bazel.rules.android.BazelAndroidLibraryRule;
import com.google.devtools.build.lib.bazel.rules.android.BazelAndroidRuleClasses.BazelAndroidToolsDefaultsJarRule;
import com.google.devtools.build.lib.bazel.rules.android.BazelAndroidSemantics;
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
import com.google.devtools.build.lib.bazel.rules.objc.BazelJ2ObjcLibraryRule;
import com.google.devtools.build.lib.bazel.rules.python.BazelPyBinaryRule;
import com.google.devtools.build.lib.bazel.rules.python.BazelPyLibraryRule;
import com.google.devtools.build.lib.bazel.rules.python.BazelPyRuleClasses;
import com.google.devtools.build.lib.bazel.rules.python.BazelPyTestRule;
import com.google.devtools.build.lib.bazel.rules.python.BazelPythonConfiguration;
import com.google.devtools.build.lib.bazel.rules.sh.BazelShBinaryRule;
import com.google.devtools.build.lib.bazel.rules.sh.BazelShLibraryRule;
import com.google.devtools.build.lib.bazel.rules.sh.BazelShRuleClasses;
import com.google.devtools.build.lib.bazel.rules.sh.BazelShTestRule;
import com.google.devtools.build.lib.bazel.rules.workspace.GitRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.workspace.HttpArchiveRule;
import com.google.devtools.build.lib.bazel.rules.workspace.HttpFileRule;
import com.google.devtools.build.lib.bazel.rules.workspace.HttpJarRule;
import com.google.devtools.build.lib.bazel.rules.workspace.MavenJarRule;
import com.google.devtools.build.lib.bazel.rules.workspace.MavenServerRule;
import com.google.devtools.build.lib.bazel.rules.workspace.NewGitRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.workspace.NewHttpArchiveRule;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.ideinfo.AndroidStudioInfoAspect;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.rules.android.AndroidBinaryOnlyRule;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration;
import com.google.devtools.build.lib.rules.android.AndroidLibraryBaseRule;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses;
import com.google.devtools.build.lib.rules.android.AndroidSkylarkCommon;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.apple.XcodeConfigRule;
import com.google.devtools.build.lib.rules.apple.XcodeVersionRule;
import com.google.devtools.build.lib.rules.cpp.CcToolchainRule;
import com.google.devtools.build.lib.rules.cpp.CcToolchainSuiteRule;
import com.google.devtools.build.lib.rules.cpp.CppBuildInfo;
import com.google.devtools.build.lib.rules.cpp.CppConfigurationLoader;
import com.google.devtools.build.lib.rules.cpp.CppOptions;
import com.google.devtools.build.lib.rules.genquery.GenQueryRule;
import com.google.devtools.build.lib.rules.java.JavaConfigurationLoader;
import com.google.devtools.build.lib.rules.java.JavaCpuSupplier;
import com.google.devtools.build.lib.rules.java.JavaImportBaseRule;
import com.google.devtools.build.lib.rules.java.JavaOptions;
import com.google.devtools.build.lib.rules.java.JavaToolchainRule;
import com.google.devtools.build.lib.rules.java.JvmConfigurationLoader;
import com.google.devtools.build.lib.rules.java.ProguardLibraryRule;
import com.google.devtools.build.lib.rules.objc.IosApplicationRule;
import com.google.devtools.build.lib.rules.objc.IosDeviceRule;
import com.google.devtools.build.lib.rules.objc.IosExtensionBinaryRule;
import com.google.devtools.build.lib.rules.objc.IosExtensionRule;
import com.google.devtools.build.lib.rules.objc.IosFrameworkBinaryRule;
import com.google.devtools.build.lib.rules.objc.IosFrameworkRule;
import com.google.devtools.build.lib.rules.objc.IosTestRule;
import com.google.devtools.build.lib.rules.objc.J2ObjcCommandLineOptions;
import com.google.devtools.build.lib.rules.objc.J2ObjcConfiguration;
import com.google.devtools.build.lib.rules.objc.J2ObjcLibraryBaseRule;
import com.google.devtools.build.lib.rules.objc.ObjcBinaryRule;
import com.google.devtools.build.lib.rules.objc.ObjcBuildInfoFactory;
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
import com.google.devtools.build.lib.rules.proto.BazelProtoLibraryRule;
import com.google.devtools.build.lib.rules.python.PythonConfigurationLoader;
import com.google.devtools.build.lib.rules.python.PythonOptions;
import com.google.devtools.build.lib.rules.repository.BindRule;
import com.google.devtools.build.lib.rules.repository.LocalRepositoryRule;
import com.google.devtools.build.lib.rules.repository.NewLocalRepositoryRule;
import com.google.devtools.build.lib.rules.repository.WorkspaceBaseRule;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.util.ResourceFileLoader;

import java.io.IOException;

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
      return "default";
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
      if (rule.getLabel().getPackageFragment().equals(Label.EXTERNAL_PACKAGE_NAME)) {
        // //external: labels are special. They have access to everything and visibility is checked
        // at the edge that points to the //external: label.
        return;
      }
      Target prerequisiteTarget = prerequisite.getTarget();
      Label prerequisiteLabel = prerequisiteTarget.getLabel();
      // We don't check the visibility of late-bound attributes, because it would break some
      // features.
      if (!context.getRule().getLabel().getPackageIdentifier().equals(
              prerequisite.getTarget().getLabel().getPackageIdentifier())
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

      if (prerequisiteTarget instanceof PackageGroup && !attrName.equals("visibility")) {
        context.reportError(
            rule.getAttributeLocation(attrName),
            "in "
                + attrName
                + " attribute of "
                + rule.getRuleClass()
                + " rule "
                + rule.getLabel()
                + ": package group '"
                + prerequisiteLabel
                + "' is misplaced here "
                + "(they are only allowed in the visibility attribute)");
      }
    }
  }

  /**
   * List of all build option classes in Bazel.
   */
  // TODO(bazel-team): merge BuildOptions.of into RuleClassProvider.
  @VisibleForTesting
  @SuppressWarnings("unchecked")
  private static final ImmutableList<Class<? extends FragmentOptions>> BUILD_OPTIONS =
      ImmutableList.of(
          BuildConfiguration.Options.class,
          CppOptions.class,
          JavaOptions.class,
          PythonOptions.class,
          BazelPythonConfiguration.Options.class,
          ObjcCommandLineOptions.class,
          AppleCommandLineOptions.class,
          J2ObjcCommandLineOptions.class,
          AndroidConfiguration.Options.class
      );

  /**
   * Java objects accessible from Skylark rule implementations using this module.
   */
  public static final ImmutableMap<String, SkylarkType> skylarkBuiltinJavaObects =
      ImmutableMap.of("android_common", SkylarkType.of(AndroidSkylarkCommon.class));


  public static void setup(ConfiguredRuleClassProvider.Builder builder) {
    builder
        .addBuildInfoFactory(new BazelJavaBuildInfoFactory())
        .addBuildInfoFactory(new CppBuildInfo())
        .addBuildInfoFactory(new ObjcBuildInfoFactory())
        .setConfigurationCollectionFactory(new BazelConfigurationCollection())
        .setPrelude("//tools/build_rules:prelude_bazel")
        .setRunfilesPrefix("")
        .setToolsRepository("@bazel_tools")
        .setPrerequisiteValidator(new BazelPrerequisiteValidator())
        .setSkylarkAccessibleJavaClasses(skylarkBuiltinJavaObects);

    builder.addBuildOptions(BUILD_OPTIONS);

    for (Class<? extends FragmentOptions> fragmentOptions : BUILD_OPTIONS) {
      builder.addConfigurationOptions(fragmentOptions);
    }

    builder.addRuleDefinition(new WorkspaceBaseRule());

    builder.addRuleDefinition(new BaseRuleClasses.BaseRule());
    builder.addRuleDefinition(new BaseRuleClasses.RuleBase());
    builder.addRuleDefinition(new BazelBaseRuleClasses.BinaryBaseRule());
    builder.addRuleDefinition(new BaseRuleClasses.TestBaseRule());
    builder.addRuleDefinition(new BazelBaseRuleClasses.ErrorRule());

    builder.addRuleDefinition(new EnvironmentRule());

    builder.addRuleDefinition(new ConfigRuleClasses.ConfigBaseRule());
    builder.addRuleDefinition(new ConfigRuleClasses.ConfigSettingRule());

    builder.addRuleDefinition(new BazelFilegroupRule());
    builder.addRuleDefinition(new BazelTestSuiteRule());
    builder.addRuleDefinition(new BazelGenRuleRule());
    builder.addRuleDefinition(new GenQueryRule());

    builder.addRuleDefinition(new BazelShRuleClasses.ShRule());
    builder.addRuleDefinition(new BazelShLibraryRule());
    builder.addRuleDefinition(new BazelShBinaryRule());
    builder.addRuleDefinition(new BazelShTestRule());
    builder.addRuleDefinition(new BazelProtoLibraryRule());

    builder.addRuleDefinition(new CcToolchainRule());
    builder.addRuleDefinition(new CcToolchainSuiteRule());
    builder.addRuleDefinition(new BazelCppRuleClasses.CcLinkingRule());
    builder.addRuleDefinition(new BazelCppRuleClasses.CcDeclRule());
    builder.addRuleDefinition(new BazelCppRuleClasses.CcBaseRule());
    builder.addRuleDefinition(new BazelCppRuleClasses.CcRule());
    builder.addRuleDefinition(new BazelCppRuleClasses.CcBinaryBaseRule());
    builder.addRuleDefinition(new BazelCppRuleClasses.CcBinaryRule());
    builder.addRuleDefinition(new BazelCppRuleClasses.CcTestRule());
    builder.addRuleDefinition(new BazelCppRuleClasses.CcLibraryBaseRule());
    builder.addRuleDefinition(new BazelCppRuleClasses.CcLibraryRule());

    builder.addRuleDefinition(new BazelPyRuleClasses.PyBaseRule());
    builder.addRuleDefinition(new BazelPyRuleClasses.PyBinaryBaseRule());
    builder.addRuleDefinition(new BazelPyLibraryRule());
    builder.addRuleDefinition(new BazelPyBinaryRule());
    builder.addRuleDefinition(new BazelPyTestRule());

    try {
      builder.addWorkspaceFile(
          ResourceFileLoader.loadResource(BazelRuleClassProvider.class, "tools.WORKSPACE"));
      builder.addWorkspaceFile(
          ResourceFileLoader.loadResource(BazelJavaRuleClasses.class, "jdk.WORKSPACE"));
      builder.addWorkspaceFile(
          ResourceFileLoader.loadResource(BazelAndroidSemantics.class, "android.WORKSPACE"));
      builder.addWorkspaceFile(
          ResourceFileLoader.loadResource(BazelJ2ObjcLibraryRule.class, "j2objc.WORKSPACE"));
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }

    builder.addRuleDefinition(new BazelJavaRuleClasses.BaseJavaBinaryRule());
    builder.addRuleDefinition(new BazelJavaRuleClasses.IjarBaseRule());
    builder.addRuleDefinition(new BazelJavaRuleClasses.JavaBaseRule());
    builder.addRuleDefinition(new ProguardLibraryRule());
    builder.addRuleDefinition(new JavaImportBaseRule());
    builder.addRuleDefinition(new BazelJavaRuleClasses.JavaRule());
    builder.addRuleDefinition(new BazelJavaBinaryRule());
    builder.addRuleDefinition(new BazelJavaLibraryRule());
    builder.addRuleDefinition(new BazelJavaImportRule());
    builder.addRuleDefinition(new BazelJavaTestRule());
    builder.addRuleDefinition(new BazelJavaPluginRule());
    builder.addRuleDefinition(new JavaToolchainRule());

    builder.addRuleDefinition(new AndroidRuleClasses.AndroidSdkRule());
    builder.addRuleDefinition(new BazelAndroidToolsDefaultsJarRule());
    builder.addRuleDefinition(new AndroidRuleClasses.AndroidBaseRule());
    builder.addRuleDefinition(new AndroidRuleClasses.AndroidAaptBaseRule());
    builder.addRuleDefinition(new AndroidRuleClasses.AndroidResourceSupportRule());
    builder.addRuleDefinition(new AndroidRuleClasses.AndroidBinaryBaseRule());
    builder.addRuleDefinition(new AndroidBinaryOnlyRule());
    builder.addRuleDefinition(new AndroidLibraryBaseRule());
    builder.addRuleDefinition(new BazelAndroidLibraryRule());
    builder.addRuleDefinition(new BazelAndroidBinaryRule());

    builder.addRuleDefinition(new IosTestRule());
    builder.addRuleDefinition(new IosDeviceRule());
    builder.addRuleDefinition(new ObjcBinaryRule());
    builder.addRuleDefinition(new ObjcBundleRule());
    builder.addRuleDefinition(new ObjcBundleLibraryRule());
    builder.addRuleDefinition(new ObjcFrameworkRule());
    builder.addRuleDefinition(new ObjcImportRule());
    builder.addRuleDefinition(new ObjcLibraryRule());
    builder.addRuleDefinition(new ObjcOptionsRule());
    builder.addRuleDefinition(new ObjcProtoLibraryRule());
    builder.addRuleDefinition(new ObjcXcodeprojRule());
    builder.addRuleDefinition(new ObjcRuleClasses.CoptsRule());
    builder.addRuleDefinition(new ObjcRuleClasses.BundlingRule());
    builder.addRuleDefinition(new ObjcRuleClasses.ReleaseBundlingRule());
    builder.addRuleDefinition(new ObjcRuleClasses.SimulatorRule());
    builder.addRuleDefinition(new ObjcRuleClasses.CompilingRule());
    builder.addRuleDefinition(new ObjcRuleClasses.LinkingRule());
    builder.addRuleDefinition(new ObjcRuleClasses.ResourcesRule());
    builder.addRuleDefinition(new ObjcRuleClasses.XcodegenRule());
    builder.addRuleDefinition(new ObjcRuleClasses.AlwaysLinkRule());
    builder.addRuleDefinition(new ObjcRuleClasses.OptionsRule());
    builder.addRuleDefinition(new ObjcRuleClasses.SdkFrameworksDependerRule());
    builder.addRuleDefinition(new ObjcRuleClasses.CompileDependencyRule());
    builder.addRuleDefinition(new ObjcRuleClasses.ResourceToolsRule());
    builder.addRuleDefinition(new ObjcRuleClasses.XcrunRule());
    builder.addRuleDefinition(new ObjcRuleClasses.IpaRule());
    builder.addRuleDefinition(new AppleToolchain.RequiresXcodeConfigRule());
    builder.addRuleDefinition(new IosApplicationRule());
    builder.addRuleDefinition(new IosExtensionBinaryRule());
    builder.addRuleDefinition(new IosExtensionRule());
    builder.addRuleDefinition(new IosFrameworkBinaryRule());
    builder.addRuleDefinition(new IosFrameworkRule());
    builder.addRuleDefinition(new XcodeVersionRule());
    builder.addRuleDefinition(new XcodeConfigRule());
    builder.addRuleDefinition(new J2ObjcLibraryBaseRule());
    builder.addRuleDefinition(new BazelJ2ObjcLibraryRule());

    builder.addRuleDefinition(new BazelExtraActionRule());
    builder.addRuleDefinition(new BazelActionListenerRule());

    builder.addRuleDefinition(new BindRule());
    builder.addRuleDefinition(new GitRepositoryRule());
    builder.addRuleDefinition(new HttpArchiveRule());
    builder.addRuleDefinition(new HttpJarRule());
    builder.addRuleDefinition(new HttpFileRule());
    builder.addRuleDefinition(new LocalRepositoryRule());
    builder.addRuleDefinition(new MavenJarRule());
    builder.addRuleDefinition(new MavenServerRule());
    builder.addRuleDefinition(new NewHttpArchiveRule());
    builder.addRuleDefinition(new NewGitRepositoryRule());
    builder.addRuleDefinition(new NewLocalRepositoryRule());
    builder.addRuleDefinition(new AndroidSdkRepositoryRule());
    builder.addRuleDefinition(new AndroidNdkRepositoryRule());

    builder.addAspectFactory(AndroidStudioInfoAspect.NAME, AndroidStudioInfoAspect.class);

    builder.addConfigurationFragment(new BazelConfiguration.Loader());
    builder.addConfigurationFragment(new CppConfigurationLoader(
        Functions.<String>identity()));
    builder.addConfigurationFragment(new PythonConfigurationLoader(Functions.<String>identity()));
    builder.addConfigurationFragment(new BazelPythonConfiguration.Loader());
    builder.addConfigurationFragment(new JvmConfigurationLoader(false, JAVA_CPU_SUPPLIER));
    builder.addConfigurationFragment(new JavaConfigurationLoader());
    builder.addConfigurationFragment(new ObjcConfigurationLoader());
    builder.addConfigurationFragment(new AppleConfiguration.Loader());
    builder.addConfigurationFragment(new J2ObjcConfiguration.Loader());
    builder.addConfigurationFragment(new AndroidConfiguration.Loader());

    builder.setUniversalConfigurationFragment(BazelConfiguration.class);
  }
}
