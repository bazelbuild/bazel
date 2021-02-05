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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.util.FileTypeSet;

/**
 * The rule creates the action subgraph from graph of {@link
 * com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget}, parsed by {@link
 * NinjaGraphRule} and passed in the form of {@link NinjaGraphProvider}.
 *
 * <p>The subgraph is computed as all actions needed to build targets from 'output_groups' (phony
 * targets can also be passed there). Bazel-built inputs should be passed with 'deps_mapping'
 * attribute. Currently, if there are two ninja_build targets which refer to intersecting subgraphs
 * in ninja_graph, all the actions will be created by each of ninja_build targets, i.e. duplicates.
 * Bazel will determine that those are duplicates and only execute each action once. Future
 * improvements are planned to avoid creation of duplicate actions, probably with the help of some
 * coordinating registry structure.
 *
 * <p>Currently all input files of the Ninja graph must be in a subdirectory of the package the
 * {@code ninja_build} rule is in. It's currently okay if they are in a subpackage, although that
 * may change later. For best results, put this rule in the top-level BUILD file.
 */
public class NinjaBuildRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        .add(
            attr("ninja_graph", LABEL)
                .allowedFileTypes(FileTypeSet.ANY_FILE)
                .allowedRuleClasses("ninja_graph")
                .setDoc("ninja_graph that parses all Ninja files that compose a graph of actions."))
        .add(
            attr("deps_mapping", BuildType.LABEL_DICT_UNARY)
                .allowedFileTypes(FileTypeSet.ANY_FILE)
                .setDoc(
                    "Mapping of paths in the Ninja file to the Bazel-built dependencies. Main"
                        + " output of each dependency will be used as an input to the Ninja"
                        + " action which refers to the corresponding path.")
                .value(ImmutableMap.of()))
        .add(
            attr("output_groups", Type.STRING_LIST_DICT)
                .setDoc(
                    "Mapping of output groups to the list of output paths in the Ninja file. "
                        + "Only the output paths mentioned in this attribute will be built."
                        + " Phony target names may be specified as the output paths."))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("ninja_build")
        .type(RuleClassType.NORMAL)
        .ancestors(BaseRuleClasses.NativeBuildRule.class)
        .factoryClass(NinjaBuild.class)
        .build();
  }
}
