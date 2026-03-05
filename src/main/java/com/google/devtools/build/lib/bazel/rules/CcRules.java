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

import static com.google.devtools.build.lib.analysis.BaseRuleClasses.createEmptySymbol;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BaseRuleClasses.EmptyRule;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.RuleSet;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCcModule;
import com.google.devtools.build.lib.rules.core.CoreRules;
import com.google.devtools.build.lib.rules.cpp.CcLibcTopAlias;
import com.google.devtools.build.lib.rules.cpp.CcToolchainAliasRule;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.platform.PlatformRules;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcBootstrap;

/** Rules for C++ support in Bazel. */
public class CcRules implements RuleSet {
  public static final CcRules INSTANCE = new CcRules();

  private CcRules() {
    // Use the static INSTANCE field instead.
  }

  @Override
  public void init(ConfiguredRuleClassProvider.Builder builder) {
    BazelCcModule bazelCcModule = new BazelCcModule();
    builder.addConfigurationFragment(CppConfiguration.class);
    builder.addBzlToplevel(
        "CcInfo", createEmptySymbol("CcInfo", "@rules_cc//cc/common:cc_info.bzl"));
    builder.addBzlToplevel(
        "DebugPackageInfo",
        createEmptySymbol("DebugPackageInfo", "@rules_cc//cc/common:debug_package_info.bzl"));
    builder.addBzlToplevel(
        "CcSharedLibraryInfo",
        createEmptySymbol(
            "CcSharedLibraryInfo", "@rules_cc//cc/common:cc_shared_library_info.bzl"));
    builder.addBzlToplevel(
        "CcSharedLibraryHintInfo",
        createEmptySymbol(
            "CcSharedLibraryHintInfo", "@rules_cc//cc/common:cc_shared_library_hint_info.bzl"));

    builder.addRuleDefinition(new EmptyRule("cc_toolchain") {});
    builder.addRuleDefinition(new EmptyRule("cc_toolchain_suite") {});
    builder.addRuleDefinition(new CcToolchainAliasRule());
    builder.addRuleDefinition(new CcLibcTopAlias());
    builder.addRuleDefinition(new EmptyRule("cc_binary") {});
    builder.addRuleDefinition(new EmptyRule("cc_shared_library") {});
    builder.addRuleDefinition(new EmptyRule("cc_static_library") {});
    builder.addRuleDefinition(new EmptyRule("cc_test") {});
    builder.addRuleDefinition(new EmptyRule("cc_library") {});
    builder.addRuleDefinition(new EmptyRule("cc_import") {});
    builder.addRuleDefinition(new EmptyRule("fdo_profile") {});
    builder.addRuleDefinition(new EmptyRule("fdo_prefetch_hints") {});
    builder.addRuleDefinition(new EmptyRule("memprof_profile") {});
    builder.addRuleDefinition(new EmptyRule("propeller_optimize") {});
    builder.addStarlarkBuiltinsInternal("cc_common", bazelCcModule);
    builder.addStarlarkBootstrap(new CcBootstrap(bazelCcModule));
  }

  @Override
  public ImmutableList<RuleSet> requires() {
    return ImmutableList.of(CoreRules.INSTANCE, PlatformRules.INSTANCE);
  }
}
