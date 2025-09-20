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
import com.google.devtools.build.lib.analysis.BaseRuleClasses.EmptyRule;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.RuleSet;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.core.CoreRules;
import com.google.devtools.build.lib.rules.objc.AppleStarlarkCommon;
import com.google.devtools.build.lib.rules.objc.J2ObjcConfiguration;
import com.google.devtools.build.lib.rules.objc.ObjcConfiguration;
import com.google.devtools.build.lib.starlarkbuildapi.objc.AppleBootstrap;

/** Rules for Objective-C support in Bazel. */
public class ObjcRules implements RuleSet {
  public static final ObjcRules INSTANCE = new ObjcRules();

  private ObjcRules() {
    // Use the static INSTANCE field instead.
  }

  @Override
  public void init(ConfiguredRuleClassProvider.Builder builder) {
    builder.addConfigurationFragment(ObjcConfiguration.class);
    builder.addConfigurationFragment(AppleConfiguration.class);
    // j2objc shouldn't be here!
    builder.addConfigurationFragment(J2ObjcConfiguration.class);

    builder.addRuleDefinition(new EmptyRule("available_xcodes") {});
    builder.addRuleDefinition(new EmptyRule("xcode_config") {});
    builder.addRuleDefinition(new EmptyRule("xcode_config_alias") {});
    builder.addRuleDefinition(new EmptyRule("xcode_version") {});

    builder.addStarlarkBuiltinsInternal("apple_common", new AppleStarlarkCommon());
    builder.addStarlarkBootstrap(new AppleBootstrap());
  }

  @Override
  public ImmutableList<RuleSet> requires() {
    return ImmutableList.of(CoreRules.INSTANCE, CcRules.INSTANCE);
  }
}
