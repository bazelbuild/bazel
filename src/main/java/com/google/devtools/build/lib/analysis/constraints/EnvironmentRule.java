// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.constraints;

import static com.google.devtools.build.lib.packages.Attribute.attr;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.util.FileTypeSet;

/**
 * Rule definition for environment rules (for Bazel's constraint enforcement system).
 */
public class EnvironmentRule implements RuleDefinition {

  public static final String FULFILLS_ATTRIBUTE = "fulfills";

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .cfg(HostTransition.createFactory())
        .override(
            attr("tags", Type.STRING_LIST)
                // No need to show up in ":all", etc. target patterns.
                .value(ImmutableList.of("manual"))
                .nonconfigurable("low-level attribute, used in TargetUtils without configurations"))
        /* <!-- #BLAZE_RULE(environment).ATTRIBUTE(fulfills) -->
        The set of environments this one is considered a valid "standin" for.
        <p>
          If rule A depends on rule B, A declares compatibility with environment <code>:foo</code>,
          and B declares compatibility with environment <code>:bar</code>, this is normally not
          considered a valid dependency. But if <code>:bar</code> fulfills <code>:foo</code>, the
          dependency is considered valid. B's own dependencies are subsequently expected to support
          <code>:bar</code> (the original expectation for <code>:foo</code> is dropped).
        </p>
        <p>
          Environments may only fulfill other environments in the same environment group.
        </p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr(FULFILLS_ATTRIBUTE, BuildType.LABEL_LIST)
                .allowedRuleClasses(ConstraintConstants.ENVIRONMENT_RULE)
                .allowedFileTypes(FileTypeSet.NO_FILE)
                .nonconfigurable(
                    "used for defining constraint models - this shouldn't be configured"))
        .exemptFromConstraintChecking("this rule *defines* a constraint")
        .setUndocumented()
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name(ConstraintConstants.ENVIRONMENT_RULE)
        .ancestors(BaseRuleClasses.NativeBuildRule.class)
        .factoryClass(Environment.class)
        .build();
  }
}
