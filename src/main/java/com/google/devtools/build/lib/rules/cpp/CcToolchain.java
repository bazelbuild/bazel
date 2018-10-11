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
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.HashMap;

/**
 * Implementation for the cc_toolchain rule.
 */
public class CcToolchain implements RuleConfiguredTargetFactory {

  /** Default attribute name where rules store the reference to cc_toolchain */
  public static final String CC_TOOLCHAIN_DEFAULT_ATTRIBUTE_NAME = ":cc_toolchain";

  /** Default attribute name for the c++ toolchain type */
  public static final String CC_TOOLCHAIN_TYPE_ATTRIBUTE_NAME = "$cc_toolchain_type";

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    CcToolchainAttributesProvider attributes =
        new CcToolchainAttributesProvider(
            ruleContext, isAppleToolchain(), getAdditionalBuildVariables(ruleContext));

    CcToolchainProvider ccToolchainProvider =
        CcToolchainProviderHelper.getCcToolchainProvider(ruleContext, attributes);

    if (ccToolchainProvider == null) {
      return null;
    }

    TemplateVariableInfo templateVariableInfo =
        createMakeVariableProvider(
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

    if (attributes.getLicensesProvider() != null) {
      builder.add(LicensesProvider.class, attributes.getLicensesProvider());
    }

    return builder.build();
  }

  private static TemplateVariableInfo createMakeVariableProvider(
      CppConfiguration cppConfiguration,
      CcToolchainProvider toolchainProvider,
      PathFragment sysroot,
      Location location) {

    HashMap<String, String> makeVariables =
        new HashMap<>(cppConfiguration.getAdditionalMakeVariables());

    // Add make variables from the toolchainProvider, also.
    ImmutableMap.Builder<String, String> ccProviderMakeVariables = new ImmutableMap.Builder<>();
    toolchainProvider.addGlobalMakeVariables(ccProviderMakeVariables);
    makeVariables.putAll(ccProviderMakeVariables.build());

    // Overwrite the CC_FLAGS variable to include sysroot, if it's available.
    if (sysroot != null) {
      String sysrootFlag = "--sysroot=" + sysroot;
      String ccFlags = makeVariables.get(CppConfiguration.CC_FLAGS_MAKE_VARIABLE_NAME);
      ccFlags = ccFlags.isEmpty() ? sysrootFlag : ccFlags + " " + sysrootFlag;
      makeVariables.put(CppConfiguration.CC_FLAGS_MAKE_VARIABLE_NAME, ccFlags);
    }
    return new TemplateVariableInfo(ImmutableMap.copyOf(makeVariables), location);
  }

  /**
   * Add local build variables from subclasses into {@link CcToolchainVariables} returned from
   * {@link CcToolchainProviderHelper#getBuildVariables(RuleContext, CcToolchainAttributesProvider,
   * PathFragment, CcToolchainVariables)}.
   *
   * <p>This method is meant to be overridden by subclasses of CcToolchain.
   */
  protected boolean isAppleToolchain() {
    // To be overridden in subclass.
    return false;
  }

  protected CcToolchainVariables getAdditionalBuildVariables(RuleContext ruleContext)
      throws RuleErrorException {
    // To be overridden in subclass.
    return CcToolchainVariables.EMPTY;
  }

}
