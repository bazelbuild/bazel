// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.syntax.Type.STRING;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.rules.ToolchainType;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;

/** A toolchain type for the c++ toolchain. */
public class CcToolchainType extends ToolchainType {

  /** Definition for {@code cc_toolchain_type}. */
  public static class CcToolchainTypeRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          .requiresConfigurationFragments(CppConfiguration.class, PlatformConfiguration.class)
          .addRequiredToolchains(CppRuleClasses.ccToolchainTypeAttribute(environment))
          .add(attr("$tools_repo", STRING).value(environment.getToolsRepository()))
          .removeAttribute("licenses")
          .removeAttribute("distribs")
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return Metadata.builder()
          .name("cc_toolchain_type")
          .factoryClass(CcToolchainType.class)
          .ancestors(BaseRuleClasses.BaseRule.class)
          .build();
    }
  }

  private static ImmutableMap<Label, Class<? extends BuildConfiguration.Fragment>>
      createFragmentMap(RuleContext ruleContext) {
    return ImmutableMap.of(
        Label.parseAbsoluteUnchecked(
            ruleContext.attributes().get("$tools_repo", STRING) + "//tools/cpp:toolchain_type"),
        CppConfiguration.class);
  }

  public CcToolchainType() {
    // Call constructor with a function that can infer fragment map from a RuleContext.
    super(CcToolchainType::createFragmentMap, ImmutableMap.of());
  }
}
