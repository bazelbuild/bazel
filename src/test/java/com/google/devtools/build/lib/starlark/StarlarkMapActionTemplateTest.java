// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.starlark;

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.skyframe.ActionTemplateExpansionValue;
import com.google.devtools.build.lib.skyframe.ActionTemplateExpansionValue.ActionTemplateExpansionKey;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.testutil.SkyframeExecutorTestHelper;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import java.util.Optional;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(TestParameterInjector.class)
public final class StarlarkMapActionTemplateTest extends BuildIntegrationTestCase {

  @Before
  public void setUp() throws Exception {
    addOptions("--experimental_allow_map_directory");
    write(
        "test/BUILD",
        """
        load(":my_rule.bzl", "my_rule")
        my_rule(
            name = "target",
            data = ":data.txt",
            data2 = ":data2.txt",
            cat_tool = ":genrule_cat_tool",
            cp_dir_tool = ":genrule_cp_dir",
            gen_subdir_tool = ":genrule_gen_subdir_tool",
        )
        genrule(
            name = "genrule_cat_tool",
            outs = ["cat_tool"],
            executable = True,
            cmd = "echo 'cat $$@ > $$1' > $@",
        )
        genrule(
            name = "genrule_cp_dir",
            outs = ["cp_dir_tool"],
            executable = True,
            cmd = "echo 'cp -R -L $$2/* $$1' > $@",
        )
        genrule(
            name = "genrule_gen_subdir_tool",
            outs = ["gen_subdir_tool"],
            executable = True,
            cmd = "echo 'mkdir -p $$1; touch $$1/f1; touch $$1/f2;' > $@",
        )
        """);
    write(
        "test/my_rule.bzl",
        """
        load(":rule_def.bzl", "rule_impl")
        my_rule = rule(
            implementation = rule_impl,
            attrs = {
                "append_data": attr.bool(default = True),
                "data": attr.label(allow_single_file = True),
                "data2": attr.label(allow_single_file = True),
                "cat_tool": attr.label(cfg = "exec", executable = True),
                "cp_dir_tool": attr.label(cfg = "exec", executable = True),
                "gen_subdir_tool": attr.label(cfg = "exec", executable = True),
            },
        )
        """);
    write(
        "test/helpers.bzl",
        """
        def create_seed_dir(ctx, dir_name, start, end):
            input_dir = ctx.actions.declare_directory(ctx.attr.name + "_" + dir_name)
            ctx.actions.run_shell(
                mnemonic = "SeedData",
                outputs = [input_dir],
                command = "for i in {%d..%d}; do echo $i > %s/%s_f$i; done" % (
                    start, end, input_dir.path, dir_name
                ),
            )
            return input_dir

        def create_seed_subdir(template_ctx, subdir_name, output_directory, tools):
            subdir = template_ctx.declare_subdirectory(
                subdir_name,
                directory = output_directory,
            )
            args = template_ctx.args()
            args.add(subdir.path)
            template_ctx.run(
                outputs = [subdir],
                executable = tools["gen_subdir_tool"],
                arguments = [args],
            )
            return subdir

        def unused_impl(template_ctx, **kwargs):
            pass

        def simple_map_impl(template_ctx, input_directories, output_directories, tools, **kwargs):
            for f1 in input_directories["input_dir"].children:
                o1 = template_ctx.declare_file(
                    f1.basename + ".out", directory = output_directories["output_dir"])
                args = template_ctx.args()
                args.add_all([o1, f1])
                template_ctx.run(
                    inputs = [f1],
                    outputs = [o1],
                    executable = tools["cat_tool"],
                    arguments = [args],
                )

        def append_data_impl(
                template_ctx,
                input_directories,
                output_directories,
                additional_inputs,
                tools,
                additional_params):
            data = additional_inputs["data"]
            for f1 in input_directories["input_dir"].children:
                o1 = template_ctx.declare_file(
                    f1.basename + ".out", directory = output_directories["output_dir"])
                args = template_ctx.args()
                args.add_all([o1, f1])
                if additional_params["append_data"]:
                    args.add(data)
                template_ctx.run(
                    inputs = [f1, data],
                    outputs = [o1],
                    executable = tools["cat_tool"],
                    arguments = [args],
                )

        def zip_and_combine_impl(
                template_ctx,
                input_directories,
                output_directories,
                tools,
                **kwargs):
            input_dir1 = input_directories["input_dir1"]
            input_dir2 = input_directories["input_dir2"]
            for f1, f2 in zip(input_dir1.children, input_dir2.children):
                o1 = template_ctx.declare_file(
                    f1.basename + "_" + f2.basename + ".out",
                    directory = output_directories["output_dir"])
                args = template_ctx.args()
                args.add_all([o1, f1, f2])
                template_ctx.run(
                    inputs = [f1, f2],
                    outputs = [o1],
                    executable = tools["cat_tool"],
                    arguments = [args],
                )

        def split_directory_impl(
                template_ctx,
                input_directories,
                output_directories,
                tools,
                **kwargs):
            input_dir = input_directories["input_dir"]
            for i, f1 in enumerate(input_dir.children):
                output_dir_key = "output_dir1" if i % 2 == 0 else "output_dir2"
                o1 = template_ctx.declare_file(
                    f1.basename + ".out",
                    directory = output_directories[output_dir_key])
                args = template_ctx.args()
                args.add_all([o1, f1])
                template_ctx.run(
                    inputs = [f1],
                    outputs = [o1],
                    executable = tools["cat_tool"],
                    arguments = [args],
                )
        """);
    write("test/data.txt", "some data");
    write("test/data2.txt", "other data");
  }

  @Test
  public void doSimpleMappingWithAdditionalInputsAndParams() throws Exception {
    SkyframeExecutorTestHelper.process(getSkyframeExecutor());
    write(
        "test/rule_def.bzl",
        """
        load(":helpers.bzl", "create_seed_dir", "append_data_impl")

        def rule_impl(ctx):
            input_dir = create_seed_dir(ctx, "input_dir", 1, 3)
            output_dir = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
            ctx.actions.map_directory(
                implementation = append_data_impl,
                input_directories = {
                    "input_dir": input_dir,
                },
                output_directories = {
                    "output_dir": output_dir,
                },
                tools = {
                    "cat_tool": ctx.attr.cat_tool.files_to_run,
                },
                additional_params = {
                    "append_data": ctx.attr.append_data,
                },
                additional_inputs = {
                    "data": ctx.file.data,
                },
            )
            return [DefaultInfo(files = depset([output_dir]))]
        """);
    buildTarget("//test:target");
    SpecialArtifact outputTree = assertTreeBuilt("test/target_output_dir");
    assertTreeContainsFileWithContents(outputTree, "input_dir_f1.out", "1", "some data");
    assertTreeContainsFileWithContents(outputTree, "input_dir_f2.out", "2", "some data");
    assertTreeContainsFileWithContents(outputTree, "input_dir_f3.out", "3", "some data");
  }

  @Test
  public void multipleInputDirectories() throws Exception {
    SkyframeExecutorTestHelper.process(getSkyframeExecutor());
    write(
        "test/rule_def.bzl",
        """
        load(":helpers.bzl", "create_seed_dir", "zip_and_combine_impl")

        def rule_impl(ctx):
            input_dir1 = create_seed_dir(ctx, "input_dir1", 1, 3)
            input_dir2 = create_seed_dir(ctx, "input_dir2", 4, 6)
            output_dir = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
            ctx.actions.map_directory(
                implementation = zip_and_combine_impl,
                input_directories = {
                    "input_dir1": input_dir1,
                    "input_dir2": input_dir2,
                },
                output_directories = {
                    "output_dir": output_dir,
                },
                tools = {
                    "cat_tool": ctx.attr.cat_tool.files_to_run,
                },
            )
            return [DefaultInfo(files = depset([output_dir]))]
        """);
    buildTarget("//test:target");
    SpecialArtifact outputTree = assertTreeBuilt("test/target_output_dir");
    assertTreeContainsFileWithContents(outputTree, "input_dir1_f1_input_dir2_f4.out", "1", "4");
    assertTreeContainsFileWithContents(outputTree, "input_dir1_f2_input_dir2_f5.out", "2", "5");
    assertTreeContainsFileWithContents(outputTree, "input_dir1_f3_input_dir2_f6.out", "3", "6");
  }

  @Test
  public void multipleOutputDirectories() throws Exception {
    SkyframeExecutorTestHelper.process(getSkyframeExecutor());
    write(
        "test/rule_def.bzl",
        """
        load(":helpers.bzl", "create_seed_dir", "split_directory_impl")

        def rule_impl(ctx):
            input_dir = create_seed_dir(ctx, "input_dir", 1, 4)
            output_dir1 = ctx.actions.declare_directory(ctx.attr.name + "_output_dir1")
            output_dir2 = ctx.actions.declare_directory(ctx.attr.name + "_output_dir2")
            ctx.actions.map_directory(
                implementation = split_directory_impl,
                input_directories = {
                    "input_dir": input_dir,
                },
                output_directories = {
                    "output_dir1": output_dir1,
                    "output_dir2": output_dir2,
                },
                tools = {
                    "cat_tool": ctx.attr.cat_tool.files_to_run,
                },
            )
            return [DefaultInfo(files = depset([output_dir1, output_dir2]))]
        """);
    buildTarget("//test:target");
    SpecialArtifact outputTree1 = assertTreeBuilt("test/target_output_dir1");
    SpecialArtifact outputTree2 = assertTreeBuilt("test/target_output_dir2");
    assertTreeContainsFileWithContents(outputTree1, "input_dir_f1.out", "1");
    assertTreeContainsFileWithContents(outputTree1, "input_dir_f3.out", "3");
    assertTreeContainsFileWithContents(outputTree2, "input_dir_f2.out", "2");
    assertTreeContainsFileWithContents(outputTree2, "input_dir_f4.out", "4");
  }

  @Test
  public void outputDirectoriesCanBeChainedToSubsequentMapDirectoryCalls() throws Exception {
    SkyframeExecutorTestHelper.process(getSkyframeExecutor());
    write(
        "test/rule_def.bzl",
        """
        load(":helpers.bzl", "create_seed_dir", "append_data_impl", "zip_and_combine_impl")

        def rule_impl(ctx):
            input_dir = create_seed_dir(ctx, "input_dir1", 1, 3)
            output_dir = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
            ctx.actions.map_directory(
                implementation = append_data_impl,
                input_directories = {
                    "input_dir": input_dir,
                },
                output_directories = {
                    "output_dir": output_dir,
                },
                tools = {
                    "cat_tool": ctx.attr.cat_tool.files_to_run,
                },
                additional_params = {
                    "append_data": ctx.attr.append_data,
                },
                additional_inputs = {
                    "data": ctx.file.data,
                },
            )
            input_dir2 = create_seed_dir(ctx, "input_dir2", 4, 6)
            output_dir2 = ctx.actions.declare_directory(ctx.attr.name + "_output_dir2")
            ctx.actions.map_directory(
                implementation = zip_and_combine_impl,
                input_directories = {
                    "input_dir1": output_dir,
                    "input_dir2": input_dir2,
                },
                output_directories = {
                    "output_dir": output_dir2,
                },
                tools = {
                    "cat_tool": ctx.attr.cat_tool.files_to_run,
                },
            )
            return [DefaultInfo(files = depset([output_dir, output_dir2]))]
        """);
    buildTarget("//test:target");
    SpecialArtifact outputTree = assertTreeBuilt("test/target_output_dir");
    assertTreeContainsFileWithContents(outputTree, "input_dir1_f1.out", "1", "some data");
    assertTreeContainsFileWithContents(outputTree, "input_dir1_f2.out", "2", "some data");
    assertTreeContainsFileWithContents(outputTree, "input_dir1_f3.out", "3", "some data");

    buildTarget("//test:target");
    SpecialArtifact outputTree2 = assertTreeBuilt("test/target_output_dir2");
    assertTreeContainsFileWithContents(
        outputTree2, "input_dir1_f1.out_input_dir2_f4.out", "1", "some data", "4");
    assertTreeContainsFileWithContents(
        outputTree2, "input_dir1_f2.out_input_dir2_f5.out", "2", "some data", "5");
    assertTreeContainsFileWithContents(
        outputTree2, "input_dir1_f3.out_input_dir2_f6.out", "3", "some data", "6");
  }

  @Test
  public void executionRequirementsPropagatedToExpandedActions() throws Exception {
    SkyframeExecutorTestHelper.process(getSkyframeExecutor());
    write(
        "test/rule_def.bzl",
        """
        load(":helpers.bzl", "create_seed_dir", "simple_map_impl")

        def rule_impl(ctx):
            input_dir = create_seed_dir(ctx, "input_dir", 1, 3)
            output_dir = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
            ctx.actions.map_directory(
                implementation = simple_map_impl,
                input_directories = {
                    "input_dir": input_dir,
                },
                output_directories = {
                    "output_dir": output_dir,
                },
                tools = {
                    "cat_tool": ctx.attr.cat_tool.files_to_run,
                },
                execution_requirements = {
                    "local": "1",
                }
            )
            return [DefaultInfo(files = depset([output_dir]))]
        """);
    buildTarget("//test:target");
    SpecialArtifact outputTree = assertTreeBuilt("test/target_output_dir");
    TreeFileArtifact treeFileArtifact1 = getTreeFileArtifact(outputTree, "input_dir_f1.out", 0);
    TreeFileArtifact treeFileArtifact2 = getTreeFileArtifact(outputTree, "input_dir_f2.out", 1);
    TreeFileArtifact treeFileArtifact3 = getTreeFileArtifact(outputTree, "input_dir_f3.out", 2);
    SpawnAction action1 = (SpawnAction) getGeneratingAction(treeFileArtifact1);
    SpawnAction action2 = (SpawnAction) getGeneratingAction(treeFileArtifact2);
    SpawnAction action3 = (SpawnAction) getGeneratingAction(treeFileArtifact3);
    assertThat(action1.getExecutionInfo()).containsEntry("local", "1");
    assertThat(action2.getExecutionInfo()).containsEntry("local", "1");
    assertThat(action3.getExecutionInfo()).containsEntry("local", "1");
  }

  @Test
  public void actionEnvironmentPropagatedToExpandedActions() throws Exception {
    SkyframeExecutorTestHelper.process(getSkyframeExecutor());
    write(
        "test/rule_def.bzl",
        """
        load(":helpers.bzl", "create_seed_dir", "simple_map_impl")

        def rule_impl(ctx):
            input_dir = create_seed_dir(ctx, "input_dir", 1, 3)
            output_dir = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
            ctx.actions.map_directory(
                implementation = simple_map_impl,
                input_directories = {
                    "input_dir": input_dir,
                },
                output_directories = {
                    "output_dir": output_dir,
                },
                tools = {
                    "cat_tool": ctx.attr.cat_tool.files_to_run,
                },
                env = {
                    "SOME_ENV": "ENV_VALUE",
                }
            )
            return [DefaultInfo(files = depset([output_dir]))]
        """);
    buildTarget("//test:target");
    SpecialArtifact outputTree = assertTreeBuilt("test/target_output_dir");
    TreeFileArtifact treeFileArtifact1 = getTreeFileArtifact(outputTree, "input_dir_f1.out", 0);
    TreeFileArtifact treeFileArtifact2 = getTreeFileArtifact(outputTree, "input_dir_f2.out", 1);
    TreeFileArtifact treeFileArtifact3 = getTreeFileArtifact(outputTree, "input_dir_f3.out", 2);
    SpawnAction action1 = (SpawnAction) getGeneratingAction(treeFileArtifact1);
    SpawnAction action2 = (SpawnAction) getGeneratingAction(treeFileArtifact2);
    SpawnAction action3 = (SpawnAction) getGeneratingAction(treeFileArtifact3);
    assertThat(action1.getEnvironment().getFixedEnv()).containsEntry("SOME_ENV", "ENV_VALUE");
    assertThat(action2.getEnvironment().getFixedEnv()).containsEntry("SOME_ENV", "ENV_VALUE");
    assertThat(action3.getEnvironment().getFixedEnv()).containsEntry("SOME_ENV", "ENV_VALUE");
  }

  @Test
  // Only boolean integer and strings are allowed in additional_params.
  public void allowedAdditionalParams(@TestParameter({"1", "True", "\"some string\""}) String value)
      throws Exception {
    write(
        "test/rule_def.bzl",
        String.format(
            """
            load(":helpers.bzl", "create_seed_dir", "simple_map_impl")

            def rule_impl(ctx):
                input_dir = create_seed_dir(ctx, "input_dir", 1, 3)
                output_dir = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
                ctx.actions.map_directory(
                    implementation = simple_map_impl,
                    input_directories = {
                        "input_dir": input_dir,
                    },
                    output_directories = {
                        "output_dir": output_dir,
                    },
                    tools = {
                        "cat_tool": ctx.attr.cat_tool.files_to_run,
                    },
                    additional_params = {
                        "some_key": %s,
                    },
                )
                return [DefaultInfo(files = depset([output_dir]))]
            """,
            value));
    buildTarget("//test:target");
    SpecialArtifact outputTree = assertTreeBuilt("test/target_output_dir");
    assertTreeContainsFileWithContents(outputTree, "input_dir_f1.out", "1");
    assertTreeContainsFileWithContents(outputTree, "input_dir_f2.out", "2");
    assertTreeContainsFileWithContents(outputTree, "input_dir_f3.out", "3");
  }

  @Test
  @TestParameters("{inputs: '{\"input_dir\": input_dir}', outputs: '{}', errorType: 'output'}")
  @TestParameters("{inputs: '{}', outputs: '{\"output_dir\": output_dir}', errorType: 'input'}")
  public void emptyInputOrOutputDirectoriesNotAllowed(
      String inputs, String outputs, String errorType) throws Exception {
    SkyframeExecutorTestHelper.process(getSkyframeExecutor());
    write(
        "test/rule_def.bzl",
        String.format(
            """
            load(":helpers.bzl", "create_seed_dir", "unused_impl")

            def rule_impl(ctx):
                input_dir = create_seed_dir(ctx, "input_dir", 1, 3)
                output_dir = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
                ctx.actions.map_directory(
                    implementation = unused_impl,
                    input_directories = %s,
                    output_directories = %s,
                    tools = {
                        "cat_tool": ctx.attr.cat_tool.files_to_run,
                    },
                )
                return [DefaultInfo(files = depset([output_dir]))]
            """,
            inputs, outputs));
    RecordingOutErr recordingOutErr = new RecordingOutErr();
    this.outErr = recordingOutErr;
    assertThrows(ViewCreationFailedException.class, () -> buildTarget("//test:target"));
    assertThat(recordingOutErr.errAsLatin1())
        .contains(String.format("actions.map_directory() requires at least one %s.", errorType));
  }

  @Test
  public void failingImplementation() throws Exception {
    SkyframeExecutorTestHelper.process(getSkyframeExecutor());
    write(
        "test/rule_def.bzl",
        """
        load(":helpers.bzl", "create_seed_dir")

        def failing_impl(template_ctx, **kwargs):
            fail("This is a test failure.")

        def rule_impl(ctx):
                input_dir = create_seed_dir(ctx, "input_dir", 1, 3)
                output_dir = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
                ctx.actions.map_directory(
                    implementation = failing_impl,
                    input_directories = {
                        "input_dir": input_dir,
                    },
                    output_directories = {
                        "output_dir": output_dir,
                    },
                    tools = {
                        "cat_tool": ctx.attr.cat_tool.files_to_run,
                    },
                )
                return [DefaultInfo(files = depset([output_dir]))]
        """);
    RecordingOutErr recordingOutErr = new RecordingOutErr();
    this.outErr = recordingOutErr;
    assertThrows(BuildFailedException.class, () -> buildTarget("//test:target"));
    assertThat(recordingOutErr.errAsLatin1()).contains("This is a test failure.");
  }

  @Test
  public void cannotDeclareFileInNonOutputDirectory() throws Exception {
    SkyframeExecutorTestHelper.process(getSkyframeExecutor());
    write(
        "test/rule_def.bzl",
        """
        load(":helpers.bzl", "create_seed_dir")

        def wrong_declare_file_impl(template_ctx, input_directories, **kwargs):
            template_ctx.declare_file("child", directory = input_directories["input_dir"].directory)

        def rule_impl(ctx):
            input_dir = create_seed_dir(ctx, "input_dir", 1, 3)
            output_dir = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
            ctx.actions.map_directory(
                implementation = wrong_declare_file_impl,
                input_directories = {
                    "input_dir": input_dir,
                },
                output_directories = {
                    "output_dir": output_dir,
                },
                tools = {
                    "cat_tool": ctx.attr.cat_tool.files_to_run,
                },
            )
            return [DefaultInfo(files = depset([output_dir]))]
        """);
    RecordingOutErr recordingOutErr = new RecordingOutErr();
    this.outErr = recordingOutErr;
    assertThrows(BuildFailedException.class, () -> buildTarget("//test:target"));
    assertThat(recordingOutErr.errAsLatin1())
        .containsMatch(
            "Cannot declare file `child` in non-output directory File.*test/target_input_dir");
  }

  @Test
  public void actionConflicts_conflictingOutputsInSameDirectory() throws Exception {
    // Don't check serialization here, since the action conflict only occurs during execution,
    // but serialization checks end up throwing (due to action conflicts) before we get there.
    write(
        "test/rule_def.bzl",
        """
        load(":helpers.bzl", "create_seed_dir")

        def conflict_impl(template_ctx, output_directories, tools, **kwargs):
            output_dir = output_directories["output_dir"]
            for i in range(2):
                o1 = template_ctx.declare_file("child", directory = output_dir)
                template_ctx.run(
                    inputs = [],
                    outputs = [o1],
                    executable = tools["cat_tool"],
                )

        def rule_impl(ctx):
            input_dir = create_seed_dir(ctx, "input_dir", 1, 3)
            output_dir = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
            ctx.actions.map_directory(
                implementation = conflict_impl,
                input_directories = {
                    "input_dir": input_dir,
                },
                output_directories = {
                    "output_dir": output_dir,
                },
                tools = {
                    "cat_tool": ctx.attr.cat_tool.files_to_run,
                },
            )
            return [DefaultInfo(files = depset([output_dir]))]
        """);

    RecordingOutErr recordingOutErr = new RecordingOutErr();
    this.outErr = recordingOutErr;
    assertThrows(BuildFailedException.class, () -> buildTarget("//test:target"));
    assertThat(recordingOutErr.errAsLatin1())
        .contains(
            "ERROR: file 'test/target_output_dir/child' is generated by these conflicting"
                + " actions:");
  }

  @Test
  @TestParameters("{output: 'input_dir.directory', path: 'test/target_input_dir'}")
  @TestParameters("{output: 'input_dir.children[0]', path: 'test/target_input_dir/input_dir_f1'}")
  @TestParameters("{output: 'output_dir', path: 'test/target_output_dir'}")
  @TestParameters("{output: 'cat_tool.executable', path: 'test/cat_tool'}")
  @TestParameters("{output: 'some_file', path: 'test/some_file'}")
  public void actionConflicts_conflictingOutputsFromOtherContext(String output, String path)
      throws Exception {
    SkyframeExecutorTestHelper.process(getSkyframeExecutor());
    write(
        "test/rule_def.bzl",
        String.format(
            """
            load(":helpers.bzl", "create_seed_dir")

            def conflict_impl(
                    template_ctx,
                    input_directories,
                    output_directories,
                    tools,
                    additional_inputs,
                    **kwargs):
                output_dir = output_directories["output_dir"]
                input_dir = input_directories["input_dir"]
                some_file = additional_inputs["some_file"]
                cat_tool = tools["cat_tool"]
                template_ctx.run(
                    inputs = [],
                    outputs = [%s],
                    executable = cat_tool,
                    progress_message = "some conflicting action",
                )

            def rule_impl(ctx):
                input_dir = create_seed_dir(ctx, "input_dir", 1, 3)
                output_dir = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
                some_file = ctx.actions.declare_file("some_file")
                ctx.actions.write(output = some_file, content = "some content")
                ctx.actions.map_directory(
                    implementation = conflict_impl,
                    input_directories = {
                        "input_dir": input_dir,
                    },
                    output_directories = {
                        "output_dir": output_dir,
                    },
                    tools = {
                        "cat_tool": ctx.attr.cat_tool.files_to_run,
                    },
                    additional_inputs = {
                        "some_file": some_file,
                    },
                )
                return [DefaultInfo(files = depset([output_dir]))]
            """,
            output));

    RecordingOutErr recordingOutErr = new RecordingOutErr();
    this.outErr = recordingOutErr;
    assertThrows(BuildFailedException.class, () -> buildTarget("//test:target"));
    assertThat(recordingOutErr.errAsLatin1())
        .containsMatch(
            String.format(
                "action 'some conflicting action' has conflicting output '.*%s' that is an output"
                    + " of another action, thus causing an action conflict.",
                path));
  }

  @Test
  @TestParameters("{value: '1', repr: '1'}")
  @TestParameters("{value: 'True', repr: 'True'}")
  @TestParameters("{value: '[1]', repr: '\\[1\\]'}")
  @TestParameters("{value: '(1, 2)', repr: '\\(1, 2\\)'}")
  public void implementationWithNonNoneReturnValueDisallowed(String value, String repr)
      throws Exception {
    SkyframeExecutorTestHelper.process(getSkyframeExecutor());
    write(
        "test/rule_def.bzl",
        String.format(
            """
            load(":helpers.bzl", "create_seed_dir")

            def non_none_impl(template_ctx, input_directories, **kwargs):
                return %s

            def rule_impl(ctx):
                input_dir = create_seed_dir(ctx, "input_dir", 1, 3)
                output_dir = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
                ctx.actions.map_directory(
                    implementation = non_none_impl,
                    input_directories = {
                        "input_dir": input_dir,
                    },
                    output_directories = {
                        "output_dir": output_dir,
                    },
                    tools = {
                        "cat_tool": ctx.attr.cat_tool.files_to_run,
                    },
                )
                return [DefaultInfo(files = depset([output_dir]))]
            """,
            value));

    RecordingOutErr recordingOutErr = new RecordingOutErr();
    this.outErr = recordingOutErr;
    assertThrows(BuildFailedException.class, () -> buildTarget("//test:target"));
    assertThat(recordingOutErr.errAsLatin1())
        .containsMatch(
            String.format(
                "actions.map_directory\\(\\) implementation non_none_impl at .* may not return a"
                    + " non-None value \\(got %s\\)",
                repr));
  }

  @Test
  public void nonTopLevelImplementationsDisallowed(
      @TestParameter({"non_top_level_impl", "lambda_impl"}) String implementation)
      throws Exception {
    SkyframeExecutorTestHelper.process(getSkyframeExecutor());
    write(
        "test/rule_def.bzl",
        String.format(
            """
            load(":helpers.bzl", "create_seed_dir")

            def rule_impl(ctx):
                def non_top_level_impl(template_ctx, **kwargs):
                    pass

                lambda_impl = lambda template_ctx, **kwargs: None

                input_dir = create_seed_dir(ctx, "input_dir", 1, 3)
                output_dir = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
                ctx.actions.map_directory(
                    implementation = %s,
                    input_directories = {
                        "input_dir": input_dir,
                    },
                    output_directories = {
                        "output_dir": output_dir,
                    },
                    tools = {
                        "cat_tool": ctx.attr.cat_tool.files_to_run,
                    },
                )
                return [DefaultInfo(files = depset([output_dir]))]
            """,
            implementation));

    RecordingOutErr recordingOutErr = new RecordingOutErr();
    this.outErr = recordingOutErr;
    assertThrows(ViewCreationFailedException.class, () -> buildTarget("//test:target"));
    assertThat(recordingOutErr.errAsLatin1())
        .containsMatch(
            "Error in map_directory: to avoid unintended retention of analysis data structures,"
                + " the function \\(declared at .*/test/rule_def.bzl:.*\\) must be declared by a"
                + " top-level def statement");
  }

  @Test
  public void canDeclareSubdirectories() throws Exception {
    SkyframeExecutorTestHelper.process(getSkyframeExecutor());
    write(
        "test/rule_def.bzl",
        """
        load(":helpers.bzl", "create_seed_dir", "create_seed_subdir")

        def subdir_impl(
                template_ctx,
                input_directories,
                output_directories,
                tools,
                **kwargs):
            output_dir = output_directories["output_dir"]
            output_file = template_ctx.declare_file(
                "single_file.txt",
                directory = output_dir,
            )
            args = template_ctx.args()
            args.add(output_file.path)
            args.add(input_directories["input_dir"].children[0].path)
            template_ctx.run(
                inputs = [input_directories["input_dir"].children[0]],
                outputs = [output_file],
                executable = tools["cat_tool"],
                arguments = [args],
            )
            create_seed_subdir(template_ctx, "subdir_0", output_dir, tools)
            create_seed_subdir(template_ctx, "subdir_1", output_dir, tools)

        def rule_impl(ctx):
            input_dir = create_seed_dir(ctx, "input_dir", 1, 2)
            output_dir = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
            ctx.actions.map_directory(
                implementation = subdir_impl,
                input_directories = {
                    "input_dir": input_dir,
                },
                output_directories = {
                    "output_dir": output_dir,
                },
                tools = {
                    "cat_tool": ctx.attr.cat_tool.files_to_run,
                    "gen_subdir_tool": ctx.attr.gen_subdir_tool.files_to_run,
                },
            )
            return [DefaultInfo(files = depset([output_dir]))]
        """);
    buildTarget("//test:target");
    SpecialArtifact outputTree = assertTreeBuilt("test/target_output_dir");
    SpecialArtifact subdir1 = getSubdirArtifact(outputTree, "subdir_0", 1);
    SpecialArtifact subdir2 = getSubdirArtifact(outputTree, "subdir_1", 2);
    // The top-level tree artifact value should contain a flattened view of all the files under it
    // (including the files from its subdirectories).
    assertThat(getChildRelativePaths(outputTree, getTreeArtifactValueFromTemplate(outputTree)))
        .containsExactly(
            PathFragment.create("single_file.txt"),
            PathFragment.create("subdir_0/f1"),
            PathFragment.create("subdir_0/f2"),
            PathFragment.create("subdir_1/f1"),
            PathFragment.create("subdir_1/f2"));
    assertThat(getChildRelativePaths(subdir1, getTreeArtifactValue(subdir1)))
        .containsExactly(PathFragment.create("f1"), PathFragment.create("f2"));
    assertThat(getChildRelativePaths(subdir2, getTreeArtifactValue(subdir2)))
        .containsExactly(PathFragment.create("f1"), PathFragment.create("f2"));
  }

  @Test
  public void actionsCanTakeSubdirectoriesAsInputs() throws Exception {
    SkyframeExecutorTestHelper.process(getSkyframeExecutor());
    write(
        "test/rule_def.bzl",
        """
        load(":helpers.bzl", "create_seed_dir", "create_seed_subdir")

        def subdir_impl(
                template_ctx,
                input_directories,
                output_directories,
                tools,
                **kwargs):
            output_dir1 = output_directories["output_dir1"]
            subdir_0 = create_seed_subdir(template_ctx, "subdir_0", output_dir1, tools)
            subdir_1 = create_seed_subdir(template_ctx, "subdir_1", output_dir1, tools)
            output_dir2 = output_directories["output_dir2"]
            other_subdir_0 = template_ctx.declare_subdirectory(
                "other_subdir_0",
                directory = output_dir2,
            )
            other_subdir_1 = template_ctx.declare_subdirectory(
                "other_subdir_1",
                directory = output_dir2,
            )
            for input_subdir, output_subdir in [
                (subdir_0, other_subdir_0),
                (subdir_1, other_subdir_1),
            ]:
                args = template_ctx.args()
                args.add_all([output_subdir.path, input_subdir.path])
                template_ctx.run(
                    inputs = [input_subdir],
                    outputs = [output_subdir],
                    executable = tools["cp_dir_tool"],
                    arguments = [args],
                )

        def rule_impl(ctx):
            input_dir = create_seed_dir(ctx, "input_dir", 1, 2)
            output_dir1 = ctx.actions.declare_directory(ctx.attr.name + "_output_dir1")
            output_dir2 = ctx.actions.declare_directory(ctx.attr.name + "_output_dir2")
            ctx.actions.map_directory(
                implementation = subdir_impl,
                input_directories = {
                    "input_dir": input_dir,
                },
                output_directories = {
                    "output_dir1": output_dir1,
                    "output_dir2": output_dir2,
                },
                tools = {
                    "gen_subdir_tool": ctx.attr.gen_subdir_tool.files_to_run,
                    "cp_dir_tool": ctx.attr.cp_dir_tool.files_to_run,
                },
            )

            return [DefaultInfo(files = depset([output_dir2]))]
        """);
    buildTarget("//test:target");
    SpecialArtifact outputTree2 = assertTreeBuilt("test/target_output_dir2");
    SpecialArtifact otherSubdir0 = getSubdirArtifact(outputTree2, "other_subdir_0", 2);
    SpecialArtifact otherSubdir1 = getSubdirArtifact(outputTree2, "other_subdir_1", 3);
    // The top-level tree artifact value should contain a flattened view of all the files under it
    // (including the files from its subdirectories).
    assertThat(getChildRelativePaths(outputTree2, getTreeArtifactValueFromTemplate(outputTree2)))
        .containsExactly(
            PathFragment.create("other_subdir_0/f1"),
            PathFragment.create("other_subdir_0/f2"),
            PathFragment.create("other_subdir_1/f1"),
            PathFragment.create("other_subdir_1/f2"));
    assertThat(getChildRelativePaths(otherSubdir0, getTreeArtifactValue(otherSubdir0)))
        .containsExactly(PathFragment.create("f1"), PathFragment.create("f2"));
    assertThat(getChildRelativePaths(otherSubdir1, getTreeArtifactValue(otherSubdir1)))
        .containsExactly(PathFragment.create("f1"), PathFragment.create("f2"));
  }

  @Test
  public void actionsCanTakeTopLevelDirectoriesAsInputs() throws Exception {
    SkyframeExecutorTestHelper.process(getSkyframeExecutor());
    write(
        "test/rule_def.bzl",
        """
        load(":helpers.bzl", "create_seed_dir", "create_seed_subdir")

        def seed_subdir_impl(
                template_ctx,
                input_directories,
                output_directories,
                tools,
                **kwargs):
            output_dir = output_directories["output_dir"]
            create_seed_subdir(template_ctx, "subdir_0", output_dir, tools)
            create_seed_subdir(template_ctx, "subdir_1", output_dir, tools)

        def rule_impl(ctx):
            input_dir = create_seed_dir(ctx, "input_dir", 1, 2)
            output_dir1 = ctx.actions.declare_directory(ctx.attr.name + "_output_dir1")
            ctx.actions.map_directory(
                implementation = seed_subdir_impl,
                input_directories = {
                    "input_dir": input_dir,
                },
                output_directories = {
                    "output_dir": output_dir1,
                },
                tools = {
                    "gen_subdir_tool": ctx.attr.gen_subdir_tool.files_to_run,
                    "cp_dir_tool": ctx.attr.cp_dir_tool.files_to_run,
                },
            )
            output_dir2 = ctx.actions.declare_directory(ctx.attr.name + "_output_dir2")
            args = ctx.actions.args()
            args.add_all([output_dir2.path, output_dir1.path])
            ctx.actions.run(
                inputs = [output_dir1],
                outputs = [output_dir2],
                executable = ctx.attr.cp_dir_tool.files_to_run,
                arguments = [args],
            )
            return [DefaultInfo(files = depset([output_dir2]))]
        """);
    buildTarget("//test:target");
    SpecialArtifact outputTree = assertTreeBuilt("test/target_output_dir2");
    // The top-level tree artifact value should contain a flattened view of all the files under it
    // (including the files from its subdirectories).
    TreeArtifactValue treeArtifactValue = getTreeArtifactValue(outputTree);
    assertThat(getChildRelativePaths(outputTree, treeArtifactValue))
        .containsExactly(
            PathFragment.create("subdir_0/f1"),
            PathFragment.create("subdir_0/f2"),
            PathFragment.create("subdir_1/f1"),
            PathFragment.create("subdir_1/f2"));
  }

  @Test
  public void actionConflicts_declaredFileWithPrefixOfSubdir() throws Exception {
    SkyframeExecutorTestHelper.process(getSkyframeExecutor());
    write(
        "test/rule_def.bzl",
        """
        load(":helpers.bzl", "create_seed_dir", "create_seed_subdir")

        def subdir_impl(
                template_ctx,
                input_directories,
                output_directories,
                tools,
                **kwargs):
            output_dir = output_directories["output_dir"]
            output_file = template_ctx.declare_file(
                "prefix_conflict/single_file.txt",
                directory = output_dir,
            )
            args = template_ctx.args()
            args.add(output_file.path)
            args.add(input_directories["input_dir"].children[0].path)
            template_ctx.run(
                inputs = [input_directories["input_dir"].children[0]],
                outputs = [output_file],
                executable = tools["cat_tool"],
                arguments = [args],
            )

            create_seed_subdir(template_ctx, "prefix_conflict", output_dir, tools)

        def rule_impl(ctx):
            input_dir = create_seed_dir(ctx, "input_dir", 1, 2)
            output_dir = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
            ctx.actions.map_directory(
                implementation = subdir_impl,
                input_directories = {
                    "input_dir": input_dir,
                },
                output_directories = {
                    "output_dir": output_dir,
                },
                tools = {
                    "cat_tool": ctx.attr.cat_tool.files_to_run,
                    "gen_subdir_tool": ctx.attr.gen_subdir_tool.files_to_run,
                },
            )
            return [DefaultInfo(files = depset([output_dir]))]
        """);
    RecordingOutErr recordingOutErr = new RecordingOutErr();
    this.outErr = recordingOutErr;
    assertThrows(BuildFailedException.class, () -> buildTarget("//test:target"));
    assertThat(recordingOutErr.errAsLatin1())
        .containsMatch(
            "ERROR: One of the output paths '.*test/target_output_dir/prefix_conflict'.* and "
                + "'.*test/target_output_dir/prefix_conflict/single_file.txt'.*"
                + " is a prefix of the other");
  }

  @Test
  public void actionConflicts_declaredSubdirWithPrefixOfSubdir() throws Exception {
    SkyframeExecutorTestHelper.process(getSkyframeExecutor());
    write(
        "test/rule_def.bzl",
        """
        load(":helpers.bzl", "create_seed_dir", "create_seed_subdir")

        def subdir_impl(
                template_ctx,
                input_directories,
                output_directories,
                tools,
                **kwargs):
            output_dir = output_directories["output_dir"]
            create_seed_subdir(template_ctx, "prefix_conflict", output_dir, tools)
            create_seed_subdir(template_ctx, "prefix_conflict/subdir", output_dir, tools)

        def rule_impl(ctx):
            input_dir = create_seed_dir(ctx, "input_dir", 1, 2)
            output_dir = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
            ctx.actions.map_directory(
                implementation = subdir_impl,
                input_directories = {
                    "input_dir": input_dir,
                },
                output_directories = {
                    "output_dir": output_dir,
                },
                tools = {
                    "cat_tool": ctx.attr.cat_tool.files_to_run,
                    "gen_subdir_tool": ctx.attr.gen_subdir_tool.files_to_run,
                },
            )
            return [DefaultInfo(files = depset([output_dir]))]
        """);
    RecordingOutErr recordingOutErr = new RecordingOutErr();
    this.outErr = recordingOutErr;
    assertThrows(BuildFailedException.class, () -> buildTarget("//test:target"));
    assertThat(recordingOutErr.errAsLatin1())
        .containsMatch(
            "ERROR: One of the output paths '.*test/target_output_dir/prefix_conflict'.* and "
                + "'.*test/target_output_dir/prefix_conflict/subdir'.*"
                + " is a prefix of the other");
  }

  @Test
  public void actionConflicts_subdirAsParentOfAnotherSubdir() throws Exception {
    SkyframeExecutorTestHelper.process(getSkyframeExecutor());
    write(
        "test/rule_def.bzl",
        """
        load(":helpers.bzl", "create_seed_dir", "create_seed_subdir")

        def subdir_impl(
                template_ctx,
                input_directories,
                output_directories,
                tools,
                **kwargs):
            output_dir = output_directories["output_dir"]
            subdir1 = create_seed_subdir(template_ctx, "subdir1", output_dir, tools)
            create_seed_subdir(template_ctx, "subdir2", subdir1, tools)

        def rule_impl(ctx):
            input_dir = create_seed_dir(ctx, "input_dir", 1, 2)
            output_dir = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
            ctx.actions.map_directory(
                implementation = subdir_impl,
                input_directories = {
                    "input_dir": input_dir,
                },
                output_directories = {
                    "output_dir": output_dir,
                },
                tools = {
                    "cat_tool": ctx.attr.cat_tool.files_to_run,
                    "gen_subdir_tool": ctx.attr.gen_subdir_tool.files_to_run,
                },
            )
            return [DefaultInfo(files = depset([output_dir]))]
        """);
    RecordingOutErr recordingOutErr = new RecordingOutErr();
    this.outErr = recordingOutErr;
    assertThrows(BuildFailedException.class, () -> buildTarget("//test:target"));
    assertThat(recordingOutErr.errAsLatin1())
        .containsMatch(
            ".*Cannot declare subdirectory `.*subdir2`.* in another subdirectory "
                + ".*test/target_output_dir/subdir1.*");
  }

  private SpecialArtifact assertTreeBuilt(String rootRelativePath) throws Exception {
    ImmutableList<Artifact> artifacts = getArtifacts("//test:target");
    Optional<Artifact> maybeTree =
        artifacts.stream()
            .filter(a -> a.getRootRelativePathString().equals(rootRelativePath))
            .findFirst();
    assertThat(maybeTree).isPresent();
    return (SpecialArtifact) maybeTree.get();
  }

  private TreeFileArtifact getTreeFileArtifact(
      SpecialArtifact tree, String relativeFilePath, int actionIndex) {
    // The actionIndex of the ActionTemplateExpansionKey should correspond to the actionIndex of the
    // StarlarkMapActionTemplate instance.
    ActionTemplateExpansionKey key =
        ActionTemplateExpansionValue.key(
            tree.getArtifactOwner(), tree.getGeneratingActionKey().getActionIndex());
    TreeFileArtifact treeFileArtifact =
        TreeFileArtifact.createTemplateExpansionOutput(tree, relativeFilePath, key);
    // OTOH, the actionIndex of the TreeFileArtifact's ActionLookupData should correspond to the
    // actionIndex of the action (created with template_ctx) that generated the file.
    treeFileArtifact.setGeneratingActionKey(ActionLookupData.create(key, actionIndex));
    return treeFileArtifact;
  }

  private ImmutableSet<PathFragment> getChildRelativePaths(
      SpecialArtifact tree, TreeArtifactValue treeValue) {
    return treeValue.getChildren().stream()
        .map(child -> child.getExecPath().relativeTo(tree.getExecPath()))
        .collect(toImmutableSet());
  }

  private SpecialArtifact getSubdirArtifact(
      SpecialArtifact parent, String subdir, int actionIndex) {
    ActionTemplateExpansionKey key =
        ActionTemplateExpansionValue.key(
            parent.getArtifactOwner(), parent.getGeneratingActionKey().getActionIndex());
    return SpecialArtifact.createSubTreeArtifact(
        parent, PathFragment.create(subdir), ActionLookupData.create(key, actionIndex));
  }

  private TreeArtifactValue getTreeArtifactValueFromTemplate(SpecialArtifact artifact)
      throws InterruptedException {
    return (TreeArtifactValue) getSkyframeExecutor().getEvaluator().getExistingValue(artifact);
  }

  private void assertTreeContainsFileWithContents(
      SpecialArtifact tree, String relativeFilePath, String... expectedContents) throws Exception {
    Path execRoot = directories.getExecRoot(TestConstants.WORKSPACE_NAME);
    Path path = execRoot.getRelative(tree.getExecPath().getChild(relativeFilePath));
    assertThat(path.exists()).isTrue();
    String actualContents = new String(FileSystemUtils.readContentAsLatin1(path));
    for (String expected : expectedContents) {
      assertThat(actualContents).contains(expected);
    }
  }
}
