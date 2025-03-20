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

package com.google.devtools.build.lib.rules.cpp;


import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;

/**
 * Dummy rule definition for cc_toolchain_alias rule. To make sure that the rule can be overridden
 * from Starlark builtins.
 */
public class CcToolchainAliasRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return Metadata.builder()
        .name("cc_toolchain_alias")
        .factoryClass(BaseRuleClasses.EmptyRuleConfiguredTargetFactory.class)
        .ancestors(BaseRuleClasses.NativeBuildRule.class)
        .build();
  }
}
