// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.RuleSet;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCcBinaryRule;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCcImportRule;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCcLibraryRule;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCcModule;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCcTestRule;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCppRuleClasses;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCppRuleClasses.CcToolchainRequiringRule;
import com.google.devtools.build.lib.rules.core.CoreRules;
import com.google.devtools.build.lib.rules.cpp.CcHostToolchainAliasRule;
import com.google.devtools.build.lib.rules.cpp.CcImportRule;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CcLibcTopAlias;
import com.google.devtools.build.lib.rules.cpp.CcToolchainAliasRule;
import com.google.devtools.build.lib.rules.cpp.CcToolchainConfigInfo;
import com.google.devtools.build.lib.rules.cpp.CcToolchainRule;
import com.google.devtools.build.lib.rules.cpp.CcToolchainSuiteRule;
import com.google.devtools.build.lib.rules.cpp.CppBuildInfo;
import com.google.devtools.build.lib.rules.cpp.CppConfigurationLoader;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses.CcIncludeScanningRule;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses.CcLinkingRule;
import com.google.devtools.build.lib.rules.cpp.CpuTransformer;
import com.google.devtools.build.lib.rules.cpp.FdoPrefetchHintsRule;
import com.google.devtools.build.lib.rules.cpp.FdoProfileRule;
import com.google.devtools.build.lib.rules.cpp.GoogleLegacyStubs;
import com.google.devtools.build.lib.rules.cpp.GraphNodeAspect;
import com.google.devtools.build.lib.rules.cpp.PropellerOptimizeRule;
import com.google.devtools.build.lib.rules.platform.PlatformRules;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcBootstrap;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import java.io.IOException;

/**
 * Rules for C++ support in Bazel.
 */
public class CcRules implements RuleSet {
  public static final CcRules INSTANCE = new CcRules();

  private CcRules() {
    // Use the static INSTANCE field instead.
  }

  @Override
  public void init(ConfiguredRuleClassProvider.Builder builder) {
    GraphNodeAspect graphNodeAspect = new GraphNodeAspect();
    builder.addConfigurationFragment(new CppConfigurationLoader(CpuTransformer.IDENTITY));
    builder.addBuildInfoFactory(new CppBuildInfo());

    builder.addNativeAspectClass(graphNodeAspect);
    builder.addRuleDefinition(new CcToolchainRule());
    builder.addRuleDefinition(new CcToolchainSuiteRule());
    builder.addRuleDefinition(new CcToolchainAliasRule());
    builder.addRuleDefinition(new CcHostToolchainAliasRule());
    builder.addRuleDefinition(new CcLibcTopAlias());
    builder.addRuleDefinition(new CcImportRule());
    builder.addRuleDefinition(new CcToolchainRequiringRule());
    builder.addRuleDefinition(new BazelCppRuleClasses.CcDeclRule());
    builder.addRuleDefinition(new BazelCppRuleClasses.CcBaseRule());
    builder.addRuleDefinition(new BazelCppRuleClasses.CcRule());
    builder.addRuleDefinition(new BazelCppRuleClasses.CcBinaryBaseRule(graphNodeAspect));
    builder.addRuleDefinition(new BazelCcBinaryRule());
    builder.addRuleDefinition(new BazelCcTestRule());
    builder.addRuleDefinition(new BazelCppRuleClasses.CcLibraryBaseRule());
    builder.addRuleDefinition(new BazelCcLibraryRule());
    builder.addRuleDefinition(new BazelCcImportRule());
    builder.addRuleDefinition(new CcIncludeScanningRule());
    builder.addRuleDefinition(new FdoProfileRule());
    builder.addRuleDefinition(new FdoPrefetchHintsRule());
    builder.addRuleDefinition(new CcLinkingRule());
    builder.addRuleDefinition(new PropellerOptimizeRule());
    builder.addStarlarkBootstrap(
        new CcBootstrap(
            new BazelCcModule(),
            CcInfo.PROVIDER,
            CcToolchainConfigInfo.PROVIDER,
            new GoogleLegacyStubs.PyWrapCcHelper(),
            new GoogleLegacyStubs.GoWrapCcHelper(),
            new GoogleLegacyStubs.PyWrapCcInfoProvider(),
            new GoogleLegacyStubs.PyCcLinkParamsProvider()));

    try {
      builder.addWorkspaceFileSuffix(
          ResourceFileLoader.loadResource(JavaRules.class, "coverage.WORKSPACE"));
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }

  @Override
  public ImmutableList<RuleSet> requires() {
    return ImmutableList.of(CoreRules.INSTANCE, PlatformRules.INSTANCE);
  }
}
