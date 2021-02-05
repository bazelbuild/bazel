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

package com.google.devtools.build.lib.rules.apple;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.VisibilityProvider;
import com.google.devtools.build.lib.analysis.VisibilityProviderImpl;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.rules.AliasConfiguredTarget;

/**
 * Implementation of the {@code xcode_config_alias} rule.
 *
 * <p>This rule is an alias to the {@code xcode_config} rule currently in use, which is in turn
 * depends on the current configuration, in particular, the value of the {@code
 * --xcode_version_config} flag.
 */
public class XcodeConfigAlias implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    ConfiguredTarget actual =
        (ConfiguredTarget) ruleContext.getPrerequisite(XcodeConfigRule.XCODE_CONFIG_ATTR_NAME);
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
   * Rule definition for the {@code xcode_config_alias} rule.
   *
   * <p>This rule is an alias to the {@code xcode_config} rule currently in use, which is in turn
   * depends on the current configuration, in particular, the value of the {@code
   * --xcode_version_config} flag.
   *
   * <p>This is intentionally undocumented for users; the workspace is expected to contain exactly
   * one instance of this rule under {@code @bazel_tools//tools/osx} and people who want to get data
   * this rule provides should depend on that one.
   */
  public static class XcodeConfigAliasRule implements RuleDefinition {

    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          .requiresConfigurationFragments(AppleConfiguration.class)
          .removeAttribute("licenses")
          .removeAttribute("distribs")
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return Metadata.builder()
          .name("xcode_config_alias")
          .ancestors(
              BaseRuleClasses.NativeBuildRule.class, AppleToolchain.RequiresXcodeConfigRule.class)
          .factoryClass(XcodeConfigAlias.class)
          .build();
    }
  }
}
