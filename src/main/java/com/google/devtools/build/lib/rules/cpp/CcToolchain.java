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
package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.LicensesProvider;
import com.google.devtools.build.lib.analysis.MiddlemanProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TemplateVariableInfo;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import java.io.Serializable;
import java.util.HashMap;
import net.starlark.java.syntax.Location;

/**
 * Implementation for the cc_toolchain rule.
 */
public class CcToolchain implements RuleConfiguredTargetFactory {

  /** Default attribute name where rules store the reference to cc_toolchain */
  public static final String CC_TOOLCHAIN_DEFAULT_ATTRIBUTE_NAME = ":cc_toolchain";

  public static final String CC_TOOLCHAIN_DEFAULT_ATTRIBUTE_NAME_FOR_STARLARK = "$cc_toolchain";

  /** Default attribute name for the c++ toolchain type */
  public static final String CC_TOOLCHAIN_TYPE_ATTRIBUTE_NAME = "$cc_toolchain_type";

  public static final String ALLOWED_LAYERING_CHECK_FEATURES_ALLOWLIST =
      "disabling_parse_headers_and_layering_check_allowed";
  public static final String ALLOWED_LAYERING_CHECK_FEATURES_TARGET =
      "@bazel_tools//tools/build_defs/cc/whitelists/parse_headers_and_layering_check:"
          + ALLOWED_LAYERING_CHECK_FEATURES_ALLOWLIST;
  public static final Label ALLOWED_LAYERING_CHECK_FEATURES_LABEL =
      Label.parseAbsoluteUnchecked(ALLOWED_LAYERING_CHECK_FEATURES_TARGET);

  public static final String LOOSE_HEADER_CHECK_ALLOWLIST =
      "loose_header_check_allowed_in_toolchain";
  public static final String LOOSE_HEADER_CHECK_TARGET =
      "@bazel_tools//tools/build_defs/cc/whitelists/starlark_hdrs_check:" + LOOSE_HEADER_CHECK_ALLOWLIST;
  public static final Label LOOSE_HEADER_CHECK_LABEL =
      Label.parseAbsoluteUnchecked(LOOSE_HEADER_CHECK_TARGET);

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    if (!isAppleToolchain()) {
      CcCommon.checkRuleLoadedThroughMacro(ruleContext);
    }
    validateToolchain(ruleContext);
    CcToolchainAttributesProvider attributes =
        new CcToolchainAttributesProvider(
            ruleContext, isAppleToolchain(), getAdditionalBuildVariablesComputer(ruleContext));

    RuleConfiguredTargetBuilder ruleConfiguredTargetBuilder =
        new RuleConfiguredTargetBuilder(ruleContext)
            .addNativeDeclaredProvider(attributes)
            .addProvider(RunfilesProvider.simple(Runfiles.EMPTY));

    if (attributes.getLicensesProvider() != null) {
      ruleConfiguredTargetBuilder.add(LicensesProvider.class, attributes.getLicensesProvider());
    }

    if (!CppHelper.useToolchainResolution(ruleContext)) {
      // This is not a platforms-backed build, let's provide CcToolchainAttributesProvider
      // and have cc_toolchain_suite select one of its toolchains and create CcToolchainProvider
      // from its attributes. We also need to provide a do-nothing ToolchainInfo.
      return ruleConfiguredTargetBuilder
          .addNativeDeclaredProvider(new ToolchainInfo(ImmutableMap.of("cc", "dummy cc toolchain")))
          .build();
    }

    // This is a platforms-backed build, we will not analyze cc_toolchain_suite at all, and we are
    // sure current cc_toolchain is the one selected. We can create CcToolchainProvider here.
    CcToolchainProvider ccToolchainProvider =
        CcToolchainProviderHelper.getCcToolchainProvider(ruleContext, attributes);

    if (ccToolchainProvider == null) {
      // Skyframe restart
      return null;
    }

    TemplateVariableInfo templateVariableInfo =
        createMakeVariableProvider(
            ccToolchainProvider,
            ruleContext.getRule().getLocation());

    ruleConfiguredTargetBuilder
        .addNativeDeclaredProvider(ccToolchainProvider)
        .addNativeDeclaredProvider(templateVariableInfo)
        .setFilesToBuild(ccToolchainProvider.getAllFiles())
        .addProvider(new MiddlemanProvider(ccToolchainProvider.getAllFilesMiddleman()));
    return ruleConfiguredTargetBuilder.build();
  }

  static TemplateVariableInfo createMakeVariableProvider(
      CcToolchainProvider toolchainProvider,
      Location location) {

    HashMap<String, String> makeVariables =
        new HashMap<>(toolchainProvider.getAdditionalMakeVariables());

    // Add make variables from the toolchainProvider, also.
    ImmutableMap.Builder<String, String> ccProviderMakeVariables = new ImmutableMap.Builder<>();
    toolchainProvider.addGlobalMakeVariables(ccProviderMakeVariables);
    makeVariables.putAll(ccProviderMakeVariables.build());

    return new TemplateVariableInfo(ImmutableMap.copyOf(makeVariables), location);
  }

  /**
   * This method marks that the toolchain at hand is actually apple_cc_toolchain. Good job me for
   * object design and encapsulation.
   */
  protected boolean isAppleToolchain() {
    // To be overridden in subclass.
    return false;
  }

  /** Functional interface for a function that accepts cpu and {@link BuildOptions}. */
  protected interface AdditionalBuildVariablesComputer {
    CcToolchainVariables apply(BuildOptions buildOptions);
  }

  /** Returns a function that will be called to retrieve root {@link CcToolchainVariables}. */
  protected AdditionalBuildVariablesComputer getAdditionalBuildVariablesComputer(
      RuleContext ruleContextPossiblyInHostConfiguration) {
    return (AdditionalBuildVariablesComputer & Serializable)
        (options) -> CcToolchainVariables.EMPTY;
  }

  /** Will be called during analysis to ensure target attributes are set correctly. */
  protected void validateToolchain(RuleContext ruleContext) throws RuleErrorException {
    // To be overridden in subclass.
  }
}
