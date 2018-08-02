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

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass;

/** Rule definition for {@link ConstraintSetting}. */
public class ConstraintSettingRule implements RuleDefinition {
  public static final String RULE_NAME = "constraint_setting";
  public static final String DEFAULT_CONSTRAINT_VALUE_ATTR = "default_constraint_value";

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        /* <!-- #BLAZE_RULE(constraint_setting).ATTRIBUTE(default_constraint_value) -->
        The label of the default value for the setting, if none is set.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr(DEFAULT_CONSTRAINT_VALUE_ATTR, BuildType.NODEP_LABEL))
        .build();
  }

  @Override
  public RuleDefinition.Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name(RULE_NAME)
        .ancestors(PlatformBaseRule.class)
        .factoryClass(ConstraintSetting.class)
        .build();
  }
}
/*<!-- #BLAZE_RULE (NAME = constraint_setting, TYPE = OTHER, FAMILY = Platform)[GENERIC_RULE] -->

<p>This rule defines a type of constraint that can be used to define an execution platform.</p>

<!-- #END_BLAZE_RULE -->*/
