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
import com.google.devtools.build.lib.rules.ToolchainLookup;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.java.Jvm;

/**
 * Implementation of {@code toolchain_lookup}.
 */
public class BazelToolchainLookup extends ToolchainLookup {

  /**
   * Definition for {@code toolchain_lookup}.
   */
  public static class BazelToolchainLookupRule implements RuleDefinition {

    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          // This means that *every* toolchain_lookup rule depends on every configuration fragment
          // that contributes Make variables, regardless of which one it is.
          .requiresConfigurationFragments(
              CppConfiguration.class, Jvm.class, AndroidConfiguration.class)
          .removeAttribute("licenses")
          .removeAttribute("distribs")
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return Metadata.builder()
          .name("toolchain_lookup")
          .factoryClass(BazelToolchainLookup.class)
          .ancestors(BaseRuleClasses.BaseRule.class)
          .build();
    }
  }

  public BazelToolchainLookup() {
    super(
        ImmutableMap.<Label, Class<? extends BuildConfiguration.Fragment>>builder()
            .put(Label.parseAbsoluteUnchecked("@bazel_tools//tools/cpp:lookup"),
                CppConfiguration.class)
            .put(Label.parseAbsoluteUnchecked("@bazel_tools//tools/jdk:lookup"),
                Jvm.class)
            .put(Label.parseAbsoluteUnchecked("@bazel_tools//tools/android:lookup"),
                AndroidConfiguration.class)
            .build(),
        ImmutableMap.<Label, ImmutableMap<String, String>>of());
  }
}
