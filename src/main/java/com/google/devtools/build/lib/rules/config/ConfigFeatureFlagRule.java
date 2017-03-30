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
// limitations under the License

package com.google.devtools.build.lib.rules.config;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.syntax.Type.STRING;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;

/** Rule definition for Android's config_feature_flag rule. */
public final class ConfigFeatureFlagRule implements RuleDefinition {

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .setUndocumented(/* It's unusable as yet, as there are no ways to interact with it. */)
        .requiresConfigurationFragments(ConfigFeatureFlagConfiguration.class)
        .add(
            attr("allowed_values", STRING_LIST)
                .mandatory()
                .nonEmpty()
                .orderIndependent()
                .nonconfigurable("policy decision; this is defining an element of configuration"))
        .add(
            attr("default_value", STRING)
                .mandatory()
                .nonconfigurable("policy decision; this is defining an element of configuration"))
        .build();
  }

  @Override
  public RuleDefinition.Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("config_feature_flag")
        .ancestors(BaseRuleClasses.BaseRule.class)
        .factoryClass(ConfigFeatureFlag.class)
        .build();
  }
}
