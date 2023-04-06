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

package com.google.devtools.build.lib.rules.platform;

import static com.google.devtools.build.lib.packages.Attribute.attr;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.ToolchainResolutionMode;
import com.google.devtools.build.lib.packages.Type;

/**
 * Describes the common settings for all platform-related rules.
 */
public class PlatformBaseRule implements RuleDefinition{

  private static final String RULE_NAME = "$platform_base_rule";

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(PlatformConfiguration.class)
        .override(
            attr("tags", Type.STRING_LIST)
                // No need to show up in ":all", etc. target patterns.
                .value(ImmutableList.of("manual"))
                .nonconfigurable("low-level attribute, used in platform configuration"))
        .override(
            // A platform is essentially a constant which is never linked into a target.
            // This will, in a very hacky way, suppress picking up default_applicable_licenses
            attr("applicable_licenses", BuildType.LABEL_LIST)
                .value(ImmutableList.of())
                .allowedFileTypes()
                .nonconfigurable("fundamental constant, used in platform configuration"))
        .exemptFromConstraintChecking("this rule helps *define* a constraint")
        .useToolchainResolution(ToolchainResolutionMode.DISABLED)
        .removeAttribute(":action_listener")
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name(RULE_NAME)
        .type(RuleClass.Builder.RuleClassType.ABSTRACT)
        .ancestors(BaseRuleClasses.NativeBuildRule.class)
        .build();
  }
}
