// Copyright 2015 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.rules.cpp.CrosstoolConfigurationLoader.toReleaseConfiguration;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.LicensesProvider;
import com.google.devtools.build.lib.analysis.MiddlemanProvider;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TemplateVariableInfo;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.rules.cpp.CcSkyframeSupportFunction.CcSkyframeSupportException;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CrosstoolRelease;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Map;

/**
 * Implementation of the {@code cc_toolchain_suite} rule.
 *
 * <p>This is currently a no-op because the logic that transforms this rule into something that can
 * be understood by the {@code cc_*} rules is in
 * {@link com.google.devtools.build.lib.rules.cpp.CppConfigurationLoader}.
 */
public class CcToolchainSuite implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    CppConfiguration cppConfiguration = ruleContext.getFragment(CppConfiguration.class);
    if (!cppConfiguration.provideCcToolchainInfoFromCcToolchainSuite()) {
      NestedSetBuilder<Artifact> filesToBuild = NestedSetBuilder.stableOrder();
      for (TransitiveInfoCollection dep : ruleContext.getPrerequisiteMap("toolchains").values()) {
        CcToolchainProvider provider = (CcToolchainProvider) dep.get(ToolchainInfo.PROVIDER);
        if (provider != null) {
          filesToBuild.addTransitive(provider.getCrosstool());
        }
      }

      return new RuleConfiguredTargetBuilder(ruleContext)
          .setFilesToBuild(filesToBuild.build())
          .add(RunfilesProvider.class, RunfilesProvider.EMPTY)
          .build();
    }

    String transformedCpu = cppConfiguration.getTransformedCpuFromOptions();
    String compiler = cppConfiguration.getCompilerFromOptions();
    String key = transformedCpu + (compiler == null ? "" : ("|" + compiler));
    Map<String, Label> toolchains =
        ruleContext.attributes().get("toolchains", BuildType.LABEL_DICT_UNARY);
    Label selectedCcToolchain = toolchains.get(key);
    if (selectedCcToolchain == null && !cppConfiguration.disableCcToolchainFromCrosstool()) {
      try {
        CrosstoolRelease crosstoolRelease = getCrosstoolRelease(ruleContext);
        if (crosstoolRelease == null) {
          // Skyframe restart
          return null;
        }
        selectedCcToolchain =
            CppConfigurationLoader.selectCcToolchainLabelUsingCrosstool(
                transformedCpu,
                compiler,
                ruleContext.getLabel(),
                toolchains,
                crosstoolRelease,
                /* usingCrosstoolForToolchainSelectionDisabled= */ false);
      } catch (InvalidConfigurationException e) {
        ruleContext.throwWithRuleError(e.getMessage());
      }
    }

    CcToolchainProvider ccToolchainProvider;
    PlatformConfiguration platformConfig =
        Preconditions.checkNotNull(ruleContext.getFragment(PlatformConfiguration.class));
    if (platformConfig.isToolchainTypeEnabled(
        CppHelper.getToolchainTypeFromRuleClass(ruleContext))) {
      // This is a platforms build (and the user requested to build this suite explicitly).
      // Cc_toolchains provide CcToolchainInfo already. Let's select the CcToolchainProvider from
      // toolchains and provide it here as well.
      ccToolchainProvider =
          selectCcToolchain(
              CcToolchainProvider.class,
              ruleContext,
              transformedCpu,
              compiler,
              selectedCcToolchain);
    } else {
      // This is not a platforms build, and cc_toolchain_suite is the one responsible for creating
      // and providing CcToolchainInfo.
      CcToolchainAttributesProvider selectedAttributes =
          selectCcToolchain(
              CcToolchainAttributesProvider.class,
              ruleContext,
              transformedCpu,
              compiler,
              selectedCcToolchain);
      ccToolchainProvider =
          CcToolchainProviderHelper.getCcToolchainProvider(ruleContext, selectedAttributes);

      if (ccToolchainProvider == null) {
        // Skyframe restart
        return null;
      }
    }

    TemplateVariableInfo templateVariableInfo =
        CcToolchain.createMakeVariableProvider(
            ccToolchainProvider.getCppConfiguration(),
            ccToolchainProvider,
            ccToolchainProvider.getSysrootPathFragment(),
            ruleContext.getRule().getLocation());

    RuleConfiguredTargetBuilder builder =
        new RuleConfiguredTargetBuilder(ruleContext)
            .addNativeDeclaredProvider(ccToolchainProvider)
            .addNativeDeclaredProvider(templateVariableInfo)
            .setFilesToBuild(ccToolchainProvider.getCrosstool())
            .addProvider(RunfilesProvider.simple(Runfiles.EMPTY))
            .addProvider(new MiddlemanProvider(ccToolchainProvider.getCrosstoolMiddleman()));

    if (ccToolchainProvider.getLicensesProvider() != null) {
      builder.add(LicensesProvider.class, ccToolchainProvider.getLicensesProvider());
    }

    return builder.build();
  }

  private <T extends HasCcToolchainLabel> T selectCcToolchain(
      Class<T> clazz,
      RuleContext ruleContext,
      String cpu,
      String compiler,
      Label selectedCcToolchain)
      throws RuleErrorException {
    T selectedAttributes = null;
    for (TransitiveInfoCollection dep : ruleContext.getPrerequisiteMap("toolchains").values()) {
      T attributes = (T) dep.get(ToolchainInfo.PROVIDER);
      if (attributes != null && attributes.getCcToolchainLabel().equals(selectedCcToolchain)) {
        selectedAttributes = attributes;
        break;
      }
    }
    if (selectedAttributes != null) {
      return clazz.cast(selectedAttributes);
    }

    String errorMessage =
        String.format(
            "cc_toolchain_suite '%s' does not contain a toolchain for CPU '%s'",
            ruleContext.getLabel(), cpu);
    if (compiler != null) {
      errorMessage = errorMessage + " and compiler " + compiler;
    }
    ruleContext.throwWithRuleError(errorMessage);
    throw new IllegalStateException("Should not be reached");
  }

  private CrosstoolRelease getCrosstoolRelease(RuleContext ruleContext)
      throws InvalidConfigurationException, InterruptedException, RuleErrorException {
    String protoAttribute = ruleContext.attributes().get("proto", Type.STRING);
    if (StringUtil.emptyToNull(protoAttribute) != null) {
      return toReleaseConfiguration(
          "cc_toolchain_suite rule " + ruleContext.getLabel(),
          () -> protoAttribute,
          /* digestOrNull= */ null);
    }

    SkyKey ccSupportKey =
        CcSkyframeSupportValue.key(
            /* fdoZipPath= */ null, ruleContext.getLabel().getPackageIdentifier());

    SkyFunction.Environment skyframeEnv = ruleContext.getAnalysisEnvironment().getSkyframeEnv();
    CcSkyframeSupportValue ccSkyframeSupportValue;
    try {
      ccSkyframeSupportValue =
          (CcSkyframeSupportValue)
              skyframeEnv.getValueOrThrow(ccSupportKey, CcSkyframeSupportException.class);
    } catch (CcSkyframeSupportException e) {
      ruleContext.throwWithRuleError(e.getMessage());
      throw new IllegalStateException("Should not be reached");
    }
    if (skyframeEnv.valuesMissing()) {
      return null;
    }
    return ccSkyframeSupportValue.getCrosstoolRelease();
  }
}
