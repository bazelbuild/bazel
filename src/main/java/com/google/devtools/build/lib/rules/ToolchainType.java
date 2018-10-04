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

package com.google.devtools.build.lib.rules;

import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.platform.ToolchainTypeInfo;
import com.google.devtools.build.lib.packages.RuleClass;

/**
 * Implementation of {@code toolchain_type}.
 */
public class ToolchainType implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws ActionConflictException {

    ToolchainTypeInfo toolchainTypeInfo = ToolchainTypeInfo.create(ruleContext.getLabel());

    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProvider(RunfilesProvider.simple(Runfiles.EMPTY))
        .addNativeDeclaredProvider(toolchainTypeInfo)
        .build();
  }

  /** Definition for {@code toolchain_type}. */
  public static class ToolchainTypeRule implements RuleDefinition {

    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          .supportsPlatforms(false)
          .removeAttribute("licenses")
          .removeAttribute("distribs")
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return Metadata.builder()
          .name("toolchain_type")
          .factoryClass(ToolchainType.class)
          .ancestors(BaseRuleClasses.BaseRule.class)
          .build();
    }
  }
}
