// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.ninja.actions;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_KEYED_STRING_DICT;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.util.FileTypeSet;

/**
 * The rule that serves as interface between ninja_graph and Bazel targets, consuming artifacts,
 * created by {@link NinjaGraphRule}.
 *
 * Having a separate rule for artifact creation allows to consume several artifacts from one Ninja
 * graph in separate Bazel targets.
 */
public class NinjaBuildRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        .setOutputToGenfiles()
        .add(attr("ninja_graph", LABEL)
            .allowedRuleClasses("ninja_graph")
            .allowedFileTypes(FileTypeSet.NO_FILE))
        .add(attr("targets", Type.STRING_LIST)
            .setDoc("Names of Ninja targets from ninja_graph to be built."))
        .add(attr("output_groups", Type.STRING_LIST_DICT)
            .value(ImmutableMap.of())
            .setDoc("Mapping of output groups to be created, with the list of resulting paths"
                + " under output_root to be added to the groups."))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("ninja_build")
        .type(RuleClassType.NORMAL)
        .ancestors(BaseRuleClasses.BaseRule.class)
        .factoryClass(NinjaBuild.class)
        .build();
  }
}
