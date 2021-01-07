// Copyright 2019 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.util.FileTypeSet;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkThread;

/**
 * The rule that parses the Ninja graph and symlinks inputs into output_root.
 *
 * <p>The rule exposes {@link NinjaGraphProvider} with maps of both non-phony and phony {@link
 * com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget} for {@link NinjaBuildRule} to
 * use for action creation.
 *
 * <p>The rules establishes Skyframe dependency on input Ninja files, as each time they change, the
 * action graph changes.
 *
 * <p>Important aspect is relation to non-symlinked-under-execroot-directories: {@link
 * com.google.devtools.build.lib.starlarkbuildapi.WorkspaceGlobalsApi#dontSymlinkDirectoriesInExecroot(Sequence,
 * StarlarkThread)} All the outputs of Ninja actions are expected to be under the directory,
 * specified in output_root of this rule. All the input files under output_root should be listed in
 * output_root_inputs attribute, this rule will create the SymlinkAction actions to symlink listed
 * files under <execroot>/<output_root>.
 */
public class NinjaGraphRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        .add(
            attr("ninja_srcs", LABEL_LIST)
                .allowedFileTypes(FileTypeSet.ANY_FILE)
                .setDoc("All included or subninja Ninja files describing the action graph."))
        .add(
            attr("main", LABEL)
                .allowedFileTypes(FileTypeSet.ANY_FILE)
                .mandatory()
                .setDoc("Main Ninja file."))
        .add(
            attr("output_root", STRING)
                .mandatory()
                .setDoc(
                    "<p>Directory under workspace, where all the intermediate and output artifacts"
                        + " will be created.</p><p>Must not be symlinked to the execroot. For"
                        + " that, toplevel_output_directories function should be used in"
                        + " WORKSPACE file.</p>"))
        .add(
            attr("output_root_inputs", STRING_LIST)
                .value(ImmutableList.of())
                .setDoc(
                    "<p>Paths under output_root, that are used as inputs to the Ninja"
                        + " file.</p><p>For each path, an action to symlink under"
                        + " <execroot>/<output_root> will be created by this rule."
                        + " <execroot>/<output_root> will be a separate directory, not a"
                        + " symlink.</p>"))
        .add(
            attr("working_directory", STRING)
                .value("")
                .setDoc(
                    "Directory under workspace's exec root to be the root for relative paths and "
                        + "working directory for all Ninja actions. "
                        + "Must be empty or set to the value or output_root."))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("ninja_graph")
        .type(RuleClassType.NORMAL)
        .ancestors(BaseRuleClasses.BaseRule.class)
        .factoryClass(NinjaGraph.class)
        .build();
  }
}
