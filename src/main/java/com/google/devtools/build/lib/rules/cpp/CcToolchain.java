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
import com.google.devtools.build.lib.analysis.LicensesProvider.TargetLicense;
import com.google.devtools.build.lib.analysis.LicensesProviderImpl;
import com.google.devtools.build.lib.analysis.MiddlemanProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TemplateVariableInfo;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.License;
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
    CcToolchainProvider ccToolchainProvider =
        CcToolchainProviderHelper.getCcToolchainProvider(
            ruleContext, isAppleToolchain(), getAdditionalBuildVariables(ruleContext));

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

    // If output_license is specified on the cc_toolchain rule, override the transitive licenses
    // with that one. This is necessary because cc_toolchain is used in the target configuration,
    // but it is sort-of-kind-of a tool, but various parts of it are linked into the output...
    // ...so we trust the judgment of the author of the cc_toolchain rule to figure out what
    // licenses should be propagated to C++ targets.
    // TODO(elenairina): Remove this and use Attribute.Builder.useOutputLicenses() on the
    // :cc_toolchain attribute instead.
    final License outputLicense =
        ruleContext.getRule().getToolOutputLicense(ruleContext.attributes());
    if (outputLicense != null && !outputLicense.equals(License.NO_LICENSE)) {
      final NestedSet<TargetLicense> license = NestedSetBuilder.create(Order.STABLE_ORDER,
          new TargetLicense(ruleContext.getLabel(), outputLicense));
      LicensesProvider licensesProvider =
          new LicensesProviderImpl(
              license, new TargetLicense(ruleContext.getLabel(), outputLicense));
      builder.add(LicensesProvider.class, licensesProvider);
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
   * {@link CcToolchainProviderHelper#getBuildVariables(RuleContext, PathFragment,
   * CcToolchainVariables)}.
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
