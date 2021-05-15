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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.RuleSet;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCppSemantics;
import com.google.devtools.build.lib.bazel.rules.objc.BazelAppleBinaryRule;
import com.google.devtools.build.lib.bazel.rules.objc.BazelAppleStaticLibraryRule;
import com.google.devtools.build.lib.bazel.rules.objc.BazelObjcImportRule;
import com.google.devtools.build.lib.bazel.rules.objc.BazelObjcLibraryRule;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.apple.AvailableXcodesRule;
import com.google.devtools.build.lib.rules.apple.XcodeConfigAlias.XcodeConfigAliasRule;
import com.google.devtools.build.lib.rules.apple.XcodeConfigRule;
import com.google.devtools.build.lib.rules.apple.XcodeVersionRule;
import com.google.devtools.build.lib.rules.apple.cpp.AppleCcToolchainRule;
import com.google.devtools.build.lib.rules.apple.swift.SwiftConfiguration;
import com.google.devtools.build.lib.rules.core.CoreRules;
import com.google.devtools.build.lib.rules.objc.AppleBinaryBaseRule;
import com.google.devtools.build.lib.rules.objc.AppleStarlarkCommon;
import com.google.devtools.build.lib.rules.objc.AppleStaticLibraryBaseRule;
import com.google.devtools.build.lib.rules.objc.J2ObjcConfiguration;
import com.google.devtools.build.lib.rules.objc.ObjcBuildInfoFactory;
import com.google.devtools.build.lib.rules.objc.ObjcConfiguration;
import com.google.devtools.build.lib.rules.objc.ObjcImportBaseRule;
import com.google.devtools.build.lib.rules.objc.ObjcLibraryBaseRule;
import com.google.devtools.build.lib.rules.objc.ObjcRuleClasses;
import com.google.devtools.build.lib.starlarkbuildapi.apple.AppleBootstrap;

/** Rules for Objective-C support in Bazel. */
public class ObjcRules implements RuleSet {
  public static final ObjcRules INSTANCE = new ObjcRules();

  private ObjcRules() {
    // Use the static INSTANCE field instead.
  }

  @Override
  public void init(ConfiguredRuleClassProvider.Builder builder) {
    String toolsRepository = checkNotNull(builder.getToolsRepository());

    builder.addBuildInfoFactory(new ObjcBuildInfoFactory());

    builder.addConfigurationFragment(ObjcConfiguration.class);
    builder.addConfigurationFragment(AppleConfiguration.class);
    builder.addConfigurationFragment(SwiftConfiguration.class);
    // j2objc shouldn't be here!
    builder.addConfigurationFragment(J2ObjcConfiguration.class);

    builder.addRuleDefinition(new AppleBinaryBaseRule());
    builder.addRuleDefinition(new AppleStaticLibraryBaseRule());

    builder.addRuleDefinition(new AppleCcToolchainRule());
    builder.addRuleDefinition(new AppleToolchain.RequiresXcodeConfigRule(toolsRepository));
    builder.addRuleDefinition(new BazelAppleBinaryRule());
    builder.addRuleDefinition(new BazelAppleStaticLibraryRule());
    builder.addRuleDefinition(new BazelObjcImportRule());
    builder.addRuleDefinition(new BazelObjcLibraryRule());
    builder.addRuleDefinition(new ObjcImportBaseRule());
    builder.addRuleDefinition(new ObjcLibraryBaseRule());
    builder.addRuleDefinition(new ObjcRuleClasses.CoptsRule());
    builder.addRuleDefinition(new ObjcRuleClasses.DylibDependingRule());
    builder.addRuleDefinition(new ObjcRuleClasses.CompilingRule());
    builder.addRuleDefinition(new ObjcRuleClasses.LinkingRule());
    builder.addRuleDefinition(new ObjcRuleClasses.PlatformRule());
    builder.addRuleDefinition(new ObjcRuleClasses.MultiArchPlatformRule());
    builder.addRuleDefinition(new ObjcRuleClasses.AlwaysLinkRule());
    builder.addRuleDefinition(new ObjcRuleClasses.SdkFrameworksDependerRule());
    builder.addRuleDefinition(new ObjcRuleClasses.CompileDependencyRule());
    builder.addRuleDefinition(new ObjcRuleClasses.XcrunRule());
    builder.addRuleDefinition(new ObjcRuleClasses.CrosstoolRule());
    builder.addRuleDefinition(new XcodeConfigRule());
    builder.addRuleDefinition(new XcodeConfigAliasRule());
    builder.addRuleDefinition(new AvailableXcodesRule());
    builder.addRuleDefinition(new XcodeVersionRule());

    builder.addStarlarkBootstrap(
        new AppleBootstrap(new AppleStarlarkCommon(BazelCppSemantics.OBJC)));
  }

  @Override
  public ImmutableList<RuleSet> requires() {
    return ImmutableList.of(CoreRules.INSTANCE, CcRules.INSTANCE);
  }
}
