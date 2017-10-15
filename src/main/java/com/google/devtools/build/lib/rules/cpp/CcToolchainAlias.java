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

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.VisibilityProvider;
import com.google.devtools.build.lib.analysis.VisibilityProviderImpl;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.rules.AliasConfiguredTarget;

/**
 * Implementation of the {@code cc_toolchain_alias} rule.
 */
public class CcToolchainAlias implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    ConfiguredTarget actual =
        (ConfiguredTarget) ruleContext.getPrerequisite(":cc_toolchain", Mode.TARGET);
    return new AliasConfiguredTarget(
        ruleContext,
        actual,
        ImmutableMap.of(
            AliasProvider.class,
            AliasProvider.fromAliasRule(ruleContext.getLabel(), actual),
            VisibilityProvider.class,
            new VisibilityProviderImpl(ruleContext.getVisibility())));
  }

  /**
   * Rule definition for the {@code cc_runtime_alias} rule.
   */
  public static class CcToolchainAliasRule implements RuleDefinition {

    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          .requiresConfigurationFragments(CppConfiguration.class)
          .removeAttribute("licenses")
          .removeAttribute("distribs")
          .add(attr(":cc_toolchain", LABEL).value(CppRuleClasses.ccToolchainAttribute(environment)))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return Metadata.builder()
          .name("cc_toolchain_alias")
          .ancestors(BaseRuleClasses.BaseRule.class)
          .factoryClass(CcToolchainAlias.class)
          .build();
    }
  }
}
