// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ConfigurationMakeVariableContext;
import com.google.devtools.build.lib.analysis.MakeVariableSupplier;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.stringtemplate.ExpansionException;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.util.Crosstool.CcToolchainConfig;
import com.google.devtools.build.lib.rules.cpp.CcCommon.Language;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.testutil.TestConstants;
import java.util.List;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for calculating the sysroot that require building configured targets. */
@RunWith(JUnit4.class)
public final class CppSysrootTest extends BuildViewTestCase {

  @Before
  public void writeDummyLibrary() throws Exception {
    scratch.file("dummy/BUILD", "cc_library(name='library')");
  }

  /**
   * Supply CC_FLAGS Make variable value computed from FeatureConfiguration. Appends them to
   * original CC_FLAGS, so FeatureConfiguration can override legacy values.
   */
  public static class CcFlagsSupplier implements MakeVariableSupplier {

    private final RuleContext ruleContext;

    public CcFlagsSupplier(RuleContext ruleContext) {
      this.ruleContext = Preconditions.checkNotNull(ruleContext);
    }

    @Override
    @Nullable
    public String getMakeVariable(String variableName) throws ExpansionException {
      if (!variableName.equals(CppConfiguration.CC_FLAGS_MAKE_VARIABLE_NAME)) {
        return null;
      }

      try {
        CcToolchainProvider toolchain = CppHelper.getToolchain(ruleContext);
        return computeCcFlags(ruleContext, toolchain);
      } catch (RuleErrorException | EvalException e) {
        throw new ExpansionException(e.getMessage());
      }
    }

    @Override
    public ImmutableMap<String, String> getAllMakeVariables() throws ExpansionException {
      return ImmutableMap.of(
          CppConfiguration.CC_FLAGS_MAKE_VARIABLE_NAME,
          getMakeVariable(CppConfiguration.CC_FLAGS_MAKE_VARIABLE_NAME));
    }

    /**
     * Computes the appropriate value of the {@code $(CC_FLAGS)} Make variable based on the given
     * toolchain.
     */
    public static String computeCcFlags(
        RuleContext ruleContext, CcToolchainProvider toolchainProvider)
        throws RuleErrorException, EvalException {

      // Determine the original value of CC_FLAGS.
      String originalCcFlags = toolchainProvider.getLegacyCcFlagsMakeVariable();
      String sysrootCcFlags = "";
      if (toolchainProvider.getSysrootPathFragment() != null) {
        sysrootCcFlags = SYSROOT_FLAG + toolchainProvider.getSysrootPathFragment();
      }

      // Fetch additional flags from the FeatureConfiguration.
      List<String> featureConfigCcFlags =
          computeCcFlagsFromFeatureConfig(ruleContext, toolchainProvider);

      // Combine the different flag sources.
      ImmutableList.Builder<String> ccFlags = new ImmutableList.Builder<>();
      ccFlags.add(originalCcFlags);

      // Only add the sysroot flag if nothing else adds sysroot, _but_ it must appear before
      // the feature config flags.
      if (!containsSysroot(originalCcFlags, featureConfigCcFlags)) {
        ccFlags.add(sysrootCcFlags);
      }

      ccFlags.addAll(featureConfigCcFlags);
      return Joiner.on(" ").join(ccFlags.build());
    }

    private static boolean containsSysroot(String ccFlags, List<String> moreCcFlags) {
      return Stream.concat(Stream.of(ccFlags), moreCcFlags.stream())
          .anyMatch(str -> str.contains(SYSROOT_FLAG));
    }

    private static List<String> computeCcFlagsFromFeatureConfig(
        RuleContext ruleContext, CcToolchainProvider toolchainProvider) throws RuleErrorException {
      FeatureConfiguration featureConfiguration = null;
      CppConfiguration cppConfiguration;
      cppConfiguration = ruleContext.getFragment(CppConfiguration.class);
      try {
        featureConfiguration =
            CcCommon.configureFeaturesOrThrowEvalException(
                ruleContext.getFeatures(),
                ruleContext.getDisabledFeatures(),
                Language.CPP,
                toolchainProvider,
                cppConfiguration);
      } catch (EvalException e) {
        ruleContext.ruleError(e.getMessage());
      }
      if (featureConfiguration.actionIsConfigured(CppActionNames.CC_FLAGS_MAKE_VARIABLE)) {
        try {
          CcToolchainVariables buildVariables = toolchainProvider.getBuildVars();
          return ImmutableList.copyOf(
              featureConfiguration.getCommandLine(
                  CppActionNames.CC_FLAGS_MAKE_VARIABLE, buildVariables));
        } catch (EvalException e) {
          throw new RuleErrorException(e.getMessage());
        }
      }
      return ImmutableList.of();
    }
  }

  private static final String SYSROOT_FLAG = "--sysroot=";

  void testCCFlagsContainsSysroot(
      BuildConfigurationValue config, String sysroot, boolean shouldContain) throws Exception {

    RuleContext ruleContext =
        getRuleContext(getConfiguredTarget(Label.parseCanonical("//dummy:library"), config));
    ConfigurationMakeVariableContext context =
        new ConfigurationMakeVariableContext(
            ruleContext.getTarget().getPackageDeclarations(),
            config,
            ruleContext.getDefaultTemplateVariableProviders(),
            ImmutableList.of(new CcFlagsSupplier(ruleContext)));
    if (shouldContain) {
      assertThat(context.lookupVariable("CC_FLAGS")).contains("--sysroot=" + sysroot);
    } else {
      assertThat(context.lookupVariable("CC_FLAGS")).doesNotContain("--sysroot=" + sysroot);
    }
  }

  CcToolchainProvider getCcToolchainProvider(BuildConfigurationValue configuration)
      throws Exception {
    // use dummy library to get C++ toolchain from toolchain resolution
    RuleContext ruleContext =
        getRuleContext(getConfiguredTarget(Label.parseCanonical("//dummy:library"), configuration));
    return Preconditions.checkNotNull(CppHelper.getToolchain(ruleContext));
  }

  @Test
  public void testHostGrteTop() throws Exception {
    scratch.file(
        "a/grte/top/BUILD",
        """
        filegroup(name = "everything")

        cc_library(name = "library")
        """);
    useConfiguration("--host_grte_top=//a/grte/top");
    BuildConfigurationValue target = getTargetConfiguration();
    CcToolchainProvider targetCcProvider = getCcToolchainProvider(target);
    BuildConfigurationValue exec = getExecConfiguration();
    CcToolchainProvider hostCcProvider = getCcToolchainProvider(exec);

    testCCFlagsContainsSysroot(exec, "a/grte/top", true);
    assertThat(hostCcProvider.getSysroot().equals(targetCcProvider.getSysroot())).isFalse();
  }

  @Test
  public void testOverrideHostGrteTop() throws Exception {
    scratch.file("a/grte/top/BUILD", "filegroup(name='everything')");
    scratch.file("b/grte/top/BUILD", "filegroup(name='everything')");
    useConfiguration("--grte_top=//a/grte/top", "--host_grte_top=//b/grte/top");
    BuildConfigurationValue target = getTargetConfiguration();
    CcToolchainProvider targetCcProvider = getCcToolchainProvider(target);
    BuildConfigurationValue exec = getExecConfiguration();
    CcToolchainProvider hostCcProvider = getCcToolchainProvider(exec);

    assertThat(targetCcProvider.getSysroot()).isEqualTo("a/grte/top");
    assertThat(hostCcProvider.getSysroot()).isEqualTo("b/grte/top");

    testCCFlagsContainsSysroot(target, "a/grte/top", true);
    testCCFlagsContainsSysroot(target, "b/grte/top", false);
    testCCFlagsContainsSysroot(exec, "b/grte/top", true);
    testCCFlagsContainsSysroot(exec, "a/grte/top", false);
  }

  @Test
  public void testGrteTopAlias() throws Exception {
    scratch.file("a/grte/top/BUILD", "filegroup(name='everything')");
    scratch.file("b/grte/top/BUILD", "alias(name='everything', actual='//a/grte/top:everything')");
    useConfiguration("--grte_top=//b/grte/top");
    BuildConfigurationValue target = getTargetConfiguration();
    CcToolchainProvider targetCcProvider = getCcToolchainProvider(target);

    assertThat(targetCcProvider.getSysroot()).isEqualTo("a/grte/top");

    testCCFlagsContainsSysroot(target, "a/grte/top", true);
    testCCFlagsContainsSysroot(target, "b/grte/top", false);
  }

  @Test
  public void testSysroot() throws Exception {
    // BuildConfigurationValue shouldn't provide a sysroot option by default.
    useConfiguration("--platforms=" + TestConstants.PLATFORM_LABEL);
    BuildConfigurationValue config = getTargetConfiguration();
    testCCFlagsContainsSysroot(config, "/usr/grte/v1", true);

    scratch.file("a/grte/top/BUILD", "filegroup(name='everything')");
    // BuildConfigurationValue should work with label grte_top options.
    useConfiguration(
        "--platforms=" + TestConstants.PLATFORM_LABEL, "--grte_top=//a/grte/top:everything");
    config = getTargetConfiguration();
    testCCFlagsContainsSysroot(config, "a/grte/top", true);
  }

  @Test
  public void testSysrootInFeatureConfigBlocksLegacySysroot() throws Exception {
    AnalysisMock.get()
        .ccSupport()
        .setupCcToolchainConfig(
            mockToolsConfig,
            CcToolchainConfig.builder().withActionConfigs("sysroot_in_action_config"));
    scratch.overwriteFile("a/grte/top/BUILD", "filegroup(name='everything')");
    useConfiguration("--grte_top=//a/grte/top:everything");
    RuleContext ruleContext =
        getRuleContext(getConfiguredTarget(Label.parseCanonical("//dummy:library"), targetConfig));
    ConfigurationMakeVariableContext context =
        new ConfigurationMakeVariableContext(
            ruleContext.getTarget().getPackageDeclarations(),
            targetConfig,
            ruleContext.getDefaultTemplateVariableProviders(),
            ImmutableList.of(new CcFlagsSupplier(ruleContext)));
    assertThat(context.lookupVariable("CC_FLAGS"))
        .contains("fc-start --sysroot=a/grte/top-from-feature fc-end");
    assertThat(context.lookupVariable("CC_FLAGS")).doesNotContain("--sysroot=a/grte/top fc");
  }

  @Test
  public void testSysrootWithExecConfig() throws Exception {
    // The exec BuildConfigurationValue shouldn't provide a sysroot option by default.
    for (String platform :
        new String[] {TestConstants.PLATFORM_LABEL, TestConstants.PIII_PLATFORM_LABEL}) {
      useConfiguration("--platforms=" + platform);
      BuildConfigurationValue config = getExecConfiguration();
      testCCFlagsContainsSysroot(config, "/usr/grte/v1", true);
    }
    // The exec BuildConfigurationValue should work with label grte_top options.
    scratch.file("a/grte/top/BUILD", "filegroup(name='everything')");
    for (String platform :
        new String[] {TestConstants.PLATFORM_LABEL, TestConstants.PIII_PLATFORM_LABEL}) {
      useConfiguration("--platforms=" + platform, "--host_grte_top=//a/grte/top");
      BuildConfigurationValue config = getExecConfiguration();
      testCCFlagsContainsSysroot(config, "a/grte/top", true);

      // "--grte_top" does *not* set the exec grte_top,
      // so we don't get "a/grte/top" here, but instead the default "/usr/grte/v1"
      useConfiguration("--platforms=" + platform, "--grte_top=//a/grte/top");
      config = getExecConfiguration();
      testCCFlagsContainsSysroot(config, "/usr/grte/v1", true);
    }
  }

  @Test
  public void testConfigurableSysroot() throws Exception {
    scratch.file(
        "test/config_setting/BUILD",
        "config_setting(name='defines', values={'define': 'override_grte_top=1'})");
    scratch.file("a/grte/top/BUILD", "filegroup(name='everything')");
    scratch.file("b/grte/top/BUILD", "filegroup(name='everything')");
    scratch.file(
        "c/grte/top/BUILD",
        """
        alias(
            name = "everything",
            actual = select(
                {
                    "//test/config_setting:defines": "//a/grte/top:everything",
                    "//conditions:default": "//b/grte/top:everything",
                },
            ),
        )
        """);
    useConfiguration("--grte_top=//c/grte/top:everything");
    CcToolchainProvider ccProvider = getCcToolchainProvider(getTargetConfiguration());
    assertThat(ccProvider.getSysroot()).isEqualTo("b/grte/top");

    useConfiguration("--grte_top=//c/grte/top:everything", "--define=override_grte_top=1");
    ccProvider = getCcToolchainProvider(getTargetConfiguration());
    assertThat(ccProvider.getSysroot()).isEqualTo("a/grte/top");
  }
}
