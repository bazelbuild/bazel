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
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleToolchain;
import com.google.devtools.build.lib.rules.apple.XcodeConfigAlias.XcodeConfigAliasRule;
import com.google.devtools.build.lib.rules.apple.XcodeConfigRule;
import com.google.devtools.build.lib.rules.apple.XcodeVersionRule;
import com.google.devtools.build.lib.rules.apple.cpp.AppleCcToolchainRule;
import com.google.devtools.build.lib.rules.apple.swift.SwiftCommandLineOptions;
import com.google.devtools.build.lib.rules.apple.swift.SwiftConfiguration;
import com.google.devtools.build.lib.rules.core.CoreRules;
import com.google.devtools.build.lib.rules.objc.AppleBinaryRule;
import com.google.devtools.build.lib.rules.objc.AppleSkylarkCommon;
import com.google.devtools.build.lib.rules.objc.AppleStaticLibraryRule;
import com.google.devtools.build.lib.rules.objc.AppleStubBinaryRule;
import com.google.devtools.build.lib.rules.objc.J2ObjcCommandLineOptions;
import com.google.devtools.build.lib.rules.objc.J2ObjcConfiguration;
import com.google.devtools.build.lib.rules.objc.ObjcBuildInfoFactory;
import com.google.devtools.build.lib.rules.objc.ObjcBundleLibraryRule;
import com.google.devtools.build.lib.rules.objc.ObjcCommandLineOptions;
import com.google.devtools.build.lib.rules.objc.ObjcConfigurationLoader;
import com.google.devtools.build.lib.rules.objc.ObjcFrameworkRule;
import com.google.devtools.build.lib.rules.objc.ObjcImportRule;
import com.google.devtools.build.lib.rules.objc.ObjcLibraryRule;
import com.google.devtools.build.lib.rules.objc.ObjcProtoAspect;
import com.google.devtools.build.lib.rules.objc.ObjcProtoLibraryRule;
import com.google.devtools.build.lib.rules.objc.ObjcRuleClasses;
import com.google.devtools.build.lib.skylarkbuildapi.apple.AppleBootstrap;

/**
 * Rules for Objective-C support in Bazel.
 */
public class ObjcRules implements RuleSet {
  public static final ObjcRules INSTANCE = new ObjcRules();

  protected ObjcRules() {
    // Use the static INSTANCE field instead.
  }

  @Override
  public void init(ConfiguredRuleClassProvider.Builder builder) {
    String toolsRepository = checkNotNull(builder.getToolsRepository());

    // objc_proto_library should go into a separate RuleSet!
    // TODO(ulfjack): Depending on objcProtoAspect from here is a layering violation.
    ObjcProtoAspect objcProtoAspect = new ObjcProtoAspect();

    builder.addBuildInfoFactory(new ObjcBuildInfoFactory());

    builder.addConfig(ObjcCommandLineOptions.class, new ObjcConfigurationLoader());
    builder.addConfig(AppleCommandLineOptions.class, new AppleConfiguration.Loader());
    builder.addConfig(SwiftCommandLineOptions.class, new SwiftConfiguration.Loader());
    // j2objc shouldn't be here!
    builder.addConfig(J2ObjcCommandLineOptions.class, new J2ObjcConfiguration.Loader());

    builder.addNativeAspectClass(objcProtoAspect);
    builder.addRuleDefinition(new AppleBinaryRule(objcProtoAspect));
    builder.addRuleDefinition(new AppleStaticLibraryRule(objcProtoAspect));
    builder.addRuleDefinition(new AppleStubBinaryRule());
    builder.addRuleDefinition(new ObjcProtoLibraryRule(objcProtoAspect));

    builder.addRuleDefinition(new AppleCcToolchainRule());
    builder.addRuleDefinition(new AppleToolchain.RequiresXcodeConfigRule(toolsRepository));
    builder.addRuleDefinition(new ObjcBundleLibraryRule());
    builder.addRuleDefinition(new ObjcFrameworkRule());
    builder.addRuleDefinition(new ObjcImportRule());
    builder.addRuleDefinition(new ObjcLibraryRule());
    builder.addRuleDefinition(new ObjcRuleClasses.CoptsRule());
    builder.addRuleDefinition(new ObjcRuleClasses.BundlingRule());
    builder.addRuleDefinition(new ObjcRuleClasses.DylibDependingRule(objcProtoAspect));
    builder.addRuleDefinition(new ObjcRuleClasses.CompilingRule());
    builder.addRuleDefinition(new ObjcRuleClasses.LinkingRule(objcProtoAspect));
    builder.addRuleDefinition(new ObjcRuleClasses.PlatformRule());
    builder.addRuleDefinition(new ObjcRuleClasses.MultiArchPlatformRule(objcProtoAspect));
    builder.addRuleDefinition(new ObjcRuleClasses.ResourcesRule());
    builder.addRuleDefinition(new ObjcRuleClasses.AlwaysLinkRule());
    builder.addRuleDefinition(new ObjcRuleClasses.SdkFrameworksDependerRule());
    builder.addRuleDefinition(new ObjcRuleClasses.CompileDependencyRule());
    builder.addRuleDefinition(new ObjcRuleClasses.ResourceToolsRule());
    builder.addRuleDefinition(new ObjcRuleClasses.XcrunRule());
    builder.addRuleDefinition(new ObjcRuleClasses.LibtoolRule());
    builder.addRuleDefinition(new ObjcRuleClasses.CrosstoolRule());
    builder.addRuleDefinition(new XcodeConfigRule());
    builder.addRuleDefinition(new XcodeConfigAliasRule());
    builder.addRuleDefinition(new XcodeVersionRule());

    builder.addSkylarkBootstrap(new AppleBootstrap(new AppleSkylarkCommon(objcProtoAspect)));
  }

  @Override
  public ImmutableList<RuleSet> requires() {
    return ImmutableList.of(CoreRules.INSTANCE, CcRules.INSTANCE);
  }
}
