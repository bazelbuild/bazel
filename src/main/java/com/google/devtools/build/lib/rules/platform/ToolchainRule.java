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
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.syntax.Type;

/** Rule definition for {@link Toolchain}. */
public class ToolchainRule implements RuleDefinition {
  public static final String RULE_NAME = "toolchain";
  public static final String TOOLCHAIN_TYPE_ATTR = "toolchain_type";
  public static final String EXEC_COMPATIBLE_WITH_ATTR = "exec_compatible_with";
  public static final String TARGET_COMPATIBLE_WITH_ATTR = "target_compatible_with";
  public static final String TOOLCHAIN_ATTR = "toolchain";

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .override(
            attr("tags", Type.STRING_LIST)
                // No need to show up in ":all", etc. target patterns.
                .value(ImmutableList.of("manual"))
                .nonconfigurable("low-level attribute, used in platform configuration"))
        .removeAttribute("deps")
        .removeAttribute("data")
        .exemptFromConstraintChecking("this rule *defines* a constraint")

        /* <!-- #BLAZE_RULE(toolchain).ATTRIBUTE(toolchain_type) -->
        The type of the toolchain, which should be the label of a toolchain_type() target.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr(TOOLCHAIN_TYPE_ATTR, BuildType.NODEP_LABEL)
                .mandatory()
                .nonconfigurable("part of toolchain configuration"))
        /* <!-- #BLAZE_RULE(toolchain).ATTRIBUTE(exec_compatible_with) -->
        The constraints, as defined by constraint_value() and constraint_setting(), that are
        required on the execution platform for this toolchain to be selected.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr(EXEC_COMPATIBLE_WITH_ATTR, BuildType.LABEL_LIST)
                .mandatoryProviders(ConstraintValueInfo.SKYLARK_CONSTRUCTOR.id())
                .allowedFileTypes()
                .nonconfigurable("part of toolchain configuration"))
        /* <!-- #BLAZE_RULE(toolchain).ATTRIBUTE(target_compatible_with) -->
        The constraints, as defined by constraint_value() and constraint_setting(), that are
        required on the target platform for this toolchain to be selected.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr(TARGET_COMPATIBLE_WITH_ATTR, BuildType.LABEL_LIST)
                .mandatoryProviders(ConstraintValueInfo.SKYLARK_CONSTRUCTOR.id())
                .allowedFileTypes()
                .nonconfigurable("part of toolchain configuration"))
        /* <!-- #BLAZE_RULE(toolchain).ATTRIBUTE(toolchain) -->
        The label of the actual toolchain data to be used when this toolchain is selected.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        // This needs to not introduce a dependency so that we can load the toolchain only if it is
        // needed.
        .add(
            attr(TOOLCHAIN_ATTR, BuildType.NODEP_LABEL)
                .mandatory()
                .nonconfigurable("part of toolchain configuration"))
        .build();
  }

  @Override
  public RuleDefinition.Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name(RULE_NAME)
        .ancestors(BaseRuleClasses.RuleBase.class)
        .factoryClass(Toolchain.class)
        .build();
  }
}
/*<!-- #BLAZE_RULE (NAME = toolchain, TYPE = OTHER, FAMILY = Platform)[GENERIC_RULE] -->

<p>This rule declares a specific toolchain's type and constraints so that it can be selected
during toolchain resolution.</p>

<!-- #END_BLAZE_RULE -->*/
