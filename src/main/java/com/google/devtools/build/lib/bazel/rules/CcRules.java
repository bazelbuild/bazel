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
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.BaseRuleClasses.EmptyRule;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.RuleSet;
import com.google.devtools.build.lib.analysis.StaticallyLinkedMarkerProvider;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCcModule;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCppRuleClasses.CcToolchainRequiringRule;
import com.google.devtools.build.lib.rules.core.CoreRules;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CcLibcTopAlias;
import com.google.devtools.build.lib.rules.cpp.CcNativeLibraryInfo;
import com.google.devtools.build.lib.rules.cpp.CcToolchainAliasRule;
import com.google.devtools.build.lib.rules.cpp.CcToolchainConfigInfo;
import com.google.devtools.build.lib.rules.cpp.CcToolchainSuiteRule;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses.CcIncludeScanningRule;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses.CcLinkingRule;
import com.google.devtools.build.lib.rules.cpp.DebugPackageProvider;
import com.google.devtools.build.lib.rules.platform.PlatformRules;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcBootstrap;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import java.io.IOException;
import net.starlark.java.eval.Starlark;

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
    BazelCcModule bazelCcModule = new BazelCcModule();
    builder.addConfigurationFragment(CppConfiguration.class);
    builder.addBzlToplevel("CcSharedLibraryInfo", Starlark.NONE);
    builder.addBzlToplevel("CcSharedLibraryHintInfo", Starlark.NONE);
    builder.addBzlToplevel("cc_proto_aspect", Starlark.NONE);
    builder.addRuleDefinition(new EmptyRule("cc_proto_library") {});

    builder.addRuleDefinition(new EmptyRule("cc_toolchain") {});
    builder.addRuleDefinition(new CcToolchainSuiteRule());
    builder.addRuleDefinition(new CcToolchainAliasRule());
    builder.addRuleDefinition(new CcLibcTopAlias());
    builder.addRuleDefinition(new CcToolchainRequiringRule());
    builder.addRuleDefinition(new BaseRuleClasses.EmptyRule("cc_binary") {});
    builder.addRuleDefinition(new EmptyRule("cc_shared_library") {});
    builder.addRuleDefinition(new EmptyRule("cc_static_library") {});
    builder.addRuleDefinition(new BaseRuleClasses.EmptyRule("cc_test") {});
    builder.addRuleDefinition(new BaseRuleClasses.EmptyRule("cc_library") {});
    builder.addRuleDefinition(new EmptyRule("cc_import") {});
    builder.addRuleDefinition(new CcIncludeScanningRule(/* addGrepIncludes= */ false));
    builder.addRuleDefinition(new EmptyRule("fdo_profile") {});
    builder.addRuleDefinition(new EmptyRule("fdo_prefetch_hints") {});
    builder.addRuleDefinition(new CcLinkingRule());
    builder.addRuleDefinition(new EmptyRule("memprof_profile") {});
    builder.addRuleDefinition(new EmptyRule("propeller_optimize") {});
    builder.addStarlarkBuiltinsInternal(
        "StaticallyLinkedMarkerProvider", StaticallyLinkedMarkerProvider.PROVIDER);
    builder.addStarlarkBuiltinsInternal("CcNativeLibraryInfo", CcNativeLibraryInfo.PROVIDER);
    builder.addStarlarkBuiltinsInternal("cc_common", bazelCcModule);
    builder.addStarlarkBootstrap(
        new CcBootstrap(
            bazelCcModule,
            CcInfo.PROVIDER,
            DebugPackageProvider.PROVIDER,
            CcToolchainConfigInfo.PROVIDER));

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
