// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.rules.ToolchainType;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;

/**
 * Implementation of {@code toolchain_type}.
 */
public class BazelToolchainType extends ToolchainType {

  /**
   * Definition for {@code toolchain_type}.
   */
  public static class BazelToolchainTypeRule implements RuleDefinition {

    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          // This means that *every* toolchain_type rule depends on every configuration fragment
          // that contributes Make variables, regardless of which one it is.
          .requiresConfigurationFragments(CppConfiguration.class, JavaConfiguration.class)
          .removeAttribute("licenses")
          .removeAttribute("distribs")
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return Metadata.builder()
          .name("toolchain_type")
          .factoryClass(BazelToolchainType.class)
          .ancestors(BaseRuleClasses.BaseRule.class)
          .build();
    }
  }

  public BazelToolchainType() {
    super(
        ImmutableMap.<Label, Class<? extends BuildConfiguration.Fragment>>builder()
            .put(Label.parseAbsoluteUnchecked("@bazel_tools//tools/cpp:toolchain_type"),
                CppConfiguration.class)
            .put(Label.parseAbsoluteUnchecked("@bazel_tools//tools/jdk:toolchain_type"),
                JavaConfiguration.class)
            .build(),
        ImmutableMap.<Label, ImmutableMap<String, String>>of());
  }
}
