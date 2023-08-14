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

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TemplateVariableInfo;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.syntax.Location;

/**
 * Implementation of the {@code cc_toolchain_suite} rule.
 *
 * <p>This is currently a no-op because the logic that transforms this rule into something that can
 * be understood by the {@code cc_*} rules is in {@link
 * com.google.devtools.build.lib.rules.cpp.CppConfiguration}.
 */
public class CcToolchainSuite implements RuleConfiguredTargetFactory {

  private static TemplateVariableInfo createMakeVariableProvider(
      CcToolchainProvider toolchainProvider, Location location) {

    HashMap<String, String> makeVariables =
        new HashMap<>(toolchainProvider.getAdditionalMakeVariables());

    // Add make variables from the toolchainProvider, also.
    ImmutableMap.Builder<String, String> ccProviderMakeVariables = new ImmutableMap.Builder<>();
    toolchainProvider.addGlobalMakeVariables(ccProviderMakeVariables);
    makeVariables.putAll(ccProviderMakeVariables.buildOrThrow());

    return new TemplateVariableInfo(ImmutableMap.copyOf(makeVariables), location);
  }

  @Override
  @Nullable
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    CppConfiguration cppConfiguration = ruleContext.getFragment(CppConfiguration.class);

    String transformedCpu = ruleContext.getConfiguration().getCpu();
    String compiler = cppConfiguration.getCompilerFromOptions();
    String key = transformedCpu + (Strings.isNullOrEmpty(compiler) ? "" : ("|" + compiler));
    Map<String, Label> toolchains =
        ruleContext.attributes().get("toolchains", BuildType.LABEL_DICT_UNARY);
    Label selectedCcToolchain = toolchains.get(key);
    CcToolchainProvider ccToolchainProvider;

    if (CppHelper.useToolchainResolution(ruleContext)) {
      // This is a platforms build (and the user requested to build this suite explicitly).
      // Cc_toolchains provide CcToolchainInfo already. Let's select the CcToolchainProvider from
      // toolchains and provide it here as well.
      ccToolchainProvider =
          selectCcToolchain(
              CcToolchainProvider.PROVIDER,
              ruleContext,
              transformedCpu,
              compiler,
              selectedCcToolchain);
    } else {
      // This is not a platforms build, and cc_toolchain_suite is the one responsible for creating
      // and providing CcToolchainInfo.
      CcToolchainAttributesProvider selectedAttributes =
          selectCcToolchain(
              CcToolchainAttributesProvider.PROVIDER,
              ruleContext,
              transformedCpu,
              compiler,
              selectedCcToolchain);
      StarlarkFunction getCcToolchainProvider =
          (StarlarkFunction) ruleContext.getStarlarkDefinedBuiltin("get_cc_toolchain_provider");
      ruleContext.initStarlarkRuleContext();
      Object starlarkCcToolchainProvider =
          ruleContext.callStarlarkOrThrowRuleError(
              getCcToolchainProvider,
              ImmutableList.of(
                  /* ctx */ ruleContext.getStarlarkRuleContext(),
                  /* attributes */ selectedAttributes,
                  /* has_apple_fragment */ true),
              ImmutableMap.of());
      ccToolchainProvider =
          starlarkCcToolchainProvider != Starlark.NONE
              ? (CcToolchainProvider) starlarkCcToolchainProvider
              : null;

      if (ccToolchainProvider == null) {
        // Skyframe restart
        return null;
      }
    }

    CcCommon.reportInvalidOptions(ruleContext, cppConfiguration, ccToolchainProvider);

    TemplateVariableInfo templateVariableInfo =
        createMakeVariableProvider(ccToolchainProvider, ruleContext.getRule().getLocation());

    RuleConfiguredTargetBuilder builder =
        new RuleConfiguredTargetBuilder(ruleContext)
            .addNativeDeclaredProvider(ccToolchainProvider)
            .addNativeDeclaredProvider(templateVariableInfo)
            .setFilesToBuild(ccToolchainProvider.getAllFilesIncludingLibc())
            .addProvider(RunfilesProvider.simple(Runfiles.EMPTY));

    if (ccToolchainProvider.getLicensesProvider() != null) {
      builder.addNativeDeclaredProvider(ccToolchainProvider.getLicensesProvider());
    }

    return builder.build();
  }

  private <T extends HasCcToolchainLabel> T selectCcToolchain(
      BuiltinProvider<T> providerType,
      RuleContext ruleContext,
      String cpu,
      String compiler,
      Label selectedCcToolchain)
      throws RuleErrorException {
    T selectedAttributes = null;
    for (TransitiveInfoCollection dep : ruleContext.getPrerequisiteMap("toolchains").values()) {
      T attributes = dep.get(providerType);
      if (attributes != null && attributes.getCcToolchainLabel().equals(selectedCcToolchain)) {
        selectedAttributes = attributes;
        break;
      }
    }
    if (selectedAttributes != null) {
      return selectedAttributes;
    }

    String errorMessage =
        String.format(
            "cc_toolchain_suite '%s' does not contain a toolchain for cpu '%s'",
            ruleContext.getLabel(), cpu);
    if (compiler != null) {
      errorMessage = errorMessage + " and compiler '" + compiler + "'.";
    }
    ruleContext.throwWithRuleError(errorMessage);
    throw new IllegalStateException("Should not be reached");
  }
}
