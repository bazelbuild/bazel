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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(TestParameterInjector.class)
public final class StarlarkActionTemplateTest extends BuildIntegrationTestCase {

  @Before
  public void setUp() throws Exception {
    addOptions("--experimental_starlark_action_templates_api");
    write(
        "test/BUILD",
        """
        load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
        load(":rules.bzl", "my_rule")
        my_rule(name = "target")
        cc_binary(
            name = "foo",
            srcs = ["foo.cc"],
        )
        filegroup(
            name = "data",
            srcs = ["data.txt"],
        )
        """);
    write("test/foo.cc", "int main() { return 0; }");
    write("test/data.txt", "data");
  }

  @Test
  public void testBuildsSuccessfully() throws Exception {
    write(
        "test/rules.bzl",
        """
        def _transform_dir_impl(actions, input_directory_listing, output_directory):
            for file in input_directory_listing:
                output_file = actions.declare_file(
                    file.tree_relative_path + ".out", directory_or_sibling = output_directory
                )
                actions.run_shell(
                    command = "cp $1 $2",
                    inputs = [file],
                    outputs = [output_file],
                    arguments = [file.path, output_file.path],
                )

        def _my_rule_impl(ctx):
          initial_directory = ctx.actions.declare_directory(
              ctx.attr.name + "_initial_directory"
          )
          ctx.actions.run_shell(
              mnemonic = "SeedData",
              outputs = [initial_directory],
              command = "for i in {1..3}; do echo $i > %s/file_$i; done" % initial_directory.path,
          )
          output_directory = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
          ctx.actions.transform_directory(
              implementation = _transform_dir_impl,
              input_directory = initial_directory,
              output_directory = output_directory,
          )
          return [DefaultInfo(files = depset([output_directory]))]

        my_rule = rule(implementation = _my_rule_impl)
        """);

    buildTarget("//test:target");
    ImmutableList<Artifact> artifacts = getArtifacts("//test:target");
    // Contains the output directory.
    assertThat(artifacts).hasSize(1);
    Artifact output = artifacts.get(0);
    assertThat(output.getRootRelativePathString()).isEqualTo("test/target_output_dir");
    assertThat(output).isInstanceOf(SpecialArtifact.class);
    SpecialArtifact outputTree = (SpecialArtifact) output;
    Path execRoot = directories.getExecRoot(TestConstants.WORKSPACE_NAME);
    // Files should be created.
    Path file1 = execRoot.getRelative(outputTree.getExecPath().getChild("file_1.out"));
    assertThat(file1.exists()).isTrue();
    assertContents("1", file1);
    Path file2 = execRoot.getRelative(outputTree.getExecPath().getChild("file_2.out"));
    assertThat(file2.exists()).isTrue();
    assertContents("2", file2);
    Path file3 = execRoot.getRelative(outputTree.getExecPath().getChild("file_3.out"));
    assertThat(file3.exists()).isTrue();
    assertContents("3", file3);
  }

  @Test
  public void testNonDirectoryFilesCannotBeDeclaredInTransformDirectory() throws Exception {
    write(
        "test/rules.bzl",
        """
        def _transform_dir_impl(actions, input_directory_listing, output_directory):
            for file in input_directory_listing:
                output_file = actions.declare_file(
                    file.tree_relative_path + ".out", directory_or_sibling = output_directory
                )
                # This should fail - cannot declare a file that is not in a directory.
                non_directory_file_output = actions.declare_file("non_directory_file.out")
                actions.run_shell(
                    command = "cp $1 $2",
                    inputs = [file],
                    outputs = [output_file, non_directory_file_output],
                    arguments = [file.path, output_file.path],
                )

        def _my_rule_impl(ctx):
            initial_directory = ctx.actions.declare_directory(ctx.attr.name + "_initial_directory")
            ctx.actions.run_shell(
                mnemonic = "SeedData",
                outputs = [initial_directory],
                command = "for i in {1..5}; do echo $i > %s/file_$i.txt; done"
                % initial_directory.path,
            )
            output_directory = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
            ctx.actions.transform_directory(
                implementation = _transform_dir_impl,
                input_directory = initial_directory,
                output_directory = output_directory,
            )
            return [DefaultInfo(files = depset([output_directory]))]

        my_rule = rule(implementation = _my_rule_impl)
        """);

    RecordingOutErr recordingOutErr = new RecordingOutErr();
    this.outErr = recordingOutErr;
    assertThrows(BuildFailedException.class, () -> buildTarget("//test:target"));
    assertThat(recordingOutErr.errAsLatin1())
        .contains(
            "Cannot declare a file outside of a directory within a ctx.actions.transform_directory"
                + " `implementation` function.");
  }

  @Test
  public void testDirectoryFilesCannotBeDeclaredOutsideTransformDirectory() throws Exception {
    write(
        "test/rules.bzl",
        """
        def _my_rule_impl(ctx):
            initial_directory = ctx.actions.declare_directory(ctx.attr.name + "_initial_directory")
            ctx.actions.run_shell(
                mnemonic = "SeedData",
                outputs = [initial_directory],
                command = "for i in {1..5}; do echo $i > %s/file_$i.txt; done"
                    % initial_directory.path,
            )
            output_directory = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
            # This should fail - cannot declare a directory file outside of `transform_directory()`.
            directory_file = ctx.actions.declare_file(
                "directory_file.out", directory_or_sibling = output_directory
            )
            return [DefaultInfo(files = depset([output_directory]))]

        my_rule = rule(implementation = _my_rule_impl)
        """);

    RecordingOutErr recordingOutErr = new RecordingOutErr();
    this.outErr = recordingOutErr;
    assertThrows(ViewCreationFailedException.class, () -> buildTarget("//test:target"));
    assertThat(recordingOutErr.errAsLatin1())
        .contains(
            "Cannot declare a directory file outside of a ctx.actions.transform_directory"
                + " `implementation` function.");
  }

  @Test
  public void testDirectoryFilesCannotBeDeclaredInNonOutputDirectories() throws Exception {
    write(
        "test/rules.bzl",
        """
        def _transform_dir_impl(
                actions,
                input_directory_listing,
                output_directory,
                additional_input_dir):
            for file in input_directory_listing:
                output_file = actions.declare_file(
                    file.tree_relative_path + ".out", directory_or_sibling = output_directory
                )
                # This should fail - `additional_input_dir` is not an output directory.
                additional_input_dir_file = actions.declare_file(
                    "directory_file.out", directory_or_sibling = additional_input_dir
                )
                actions.run_shell(
                    command = "cp $1 $2",
                    inputs = [file],
                    outputs = [output_file],
                    arguments = [file.path, output_file.path],
                )

        def _my_rule_impl(ctx):
            initial_directory = ctx.actions.declare_directory(ctx.attr.name + "_initial_directory")
            ctx.actions.run_shell(
                mnemonic = "SeedData",
                outputs = [initial_directory],
                command = "for i in {1..5}; do echo $i > %s/file_$i.txt; done"
                    % initial_directory.path,
            )

            additional_input_dir = ctx.actions.declare_directory(ctx.attr.name + "_input_dir")
            ctx.actions.run_shell(
                mnemonic = "SeedData2",
                outputs = [additional_input_dir],
                command = "for i in {1..5}; do echo $i > %s/file_$i.txt; done"
                    % additional_input_dir.path,
            )

            output_directory = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
            ctx.actions.transform_directory(
                implementation = _transform_dir_impl,
                input_directory = initial_directory,
                output_directory = output_directory,
                additional_input_dir = additional_input_dir,
            )
            return [DefaultInfo(files = depset([output_directory]))]

        my_rule = rule(implementation = _my_rule_impl)
        """);

    RecordingOutErr recordingOutErr = new RecordingOutErr();
    this.outErr = recordingOutErr;
    assertThrows(BuildFailedException.class, () -> buildTarget("//test:target"));
    assertThat(recordingOutErr.errAsLatin1())
        .contains("Cannot declare a directory file in a non-output directory.");
  }

  @Test
  public void testNonDirectoriesNotAllowedAsAdditionalOutputs() throws Exception {
    write(
        "test/rules.bzl",
        """
        def _transform_dir_impl(
              actions,
              input_directory_listing,
              output_directory,
              non_directory_file_output):
            pass

        def _my_rule_impl(ctx):
            initial_directory = ctx.actions.declare_directory(ctx.attr.name + "_initial_directory")
            ctx.actions.run_shell(
                mnemonic = "SeedData",
                outputs = [initial_directory],
                command = "for i in {1..5}; do echo $i > %s/file_$i.txt; done"
                    % initial_directory.path,
            )
            non_directory_file_output = ctx.actions.declare_file("non_directory_file.out")
            output_directory = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
            ctx.actions.transform_directory(
                implementation = _transform_dir_impl,
                input_directory = initial_directory,
                output_directory = output_directory,
                # This should fail - passing in a non-directory files as additional_outputs is not
                # allowed.
                additional_outputs = [non_directory_file_output],
                non_directory_file_output = non_directory_file_output,
            )
            return [DefaultInfo(files = depset([output_directory]))]

        my_rule = rule(implementation = _my_rule_impl)
        """);

    RecordingOutErr recordingOutErr = new RecordingOutErr();
    this.outErr = recordingOutErr;
    assertThrows(ViewCreationFailedException.class, () -> buildTarget("//test:target"));
    assertThat(recordingOutErr.errAsLatin1())
        .contains("Expected directory artifact for additional_outputs[0], but got a File");
  }

  @Test
  public void testActionConflictCheckingDuplicateFiles() throws Exception {
    write(
        "test/rules.bzl",
        """
        def _transform_dir_impl(actions, input_directory_listing, output_directory):
            for file in input_directory_listing:
                # This should fail - the same file is declared twice.
                output_file = actions.declare_file(
                    "duplicate_file.txt", directory_or_sibling = output_directory
                )
                actions.run_shell(
                    command = "cp $1 $2",
                    inputs = [file],
                    outputs = [output_file],
                    arguments = [file.path, output_file.path],
                )

        def _my_rule_impl(ctx):
            initial_directory = ctx.actions.declare_directory(ctx.attr.name + "_initial_directory")
            ctx.actions.run_shell(
                mnemonic = "SeedData",
                outputs = [initial_directory],
                command = "for i in {1..2}; do echo $i > %s/file_$i.txt; done"
                    % initial_directory.path,
            )

            output_directory = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
            ctx.actions.transform_directory(
                implementation = _transform_dir_impl,
                input_directory = initial_directory,
                output_directory = output_directory,
            )
            return [DefaultInfo(files = depset([output_directory]))]

        my_rule = rule(implementation = _my_rule_impl)
        """);
    RecordingOutErr recordingOutErr = new RecordingOutErr();
    this.outErr = recordingOutErr;
    assertThrows(BuildFailedException.class, () -> buildTarget("//test:target"));
    assertThat(recordingOutErr.errAsLatin1())
        .contains(
            "file 'test/target_output_dir/duplicate_file.txt' is generated by these conflicting"
                + " actions");
  }

  @Test
  public void testActionConflictCheckingPrefixConflicts() throws Exception {
    write(
        "test/rules.bzl",
        """
        def _transform_dir_impl(actions, input_directory_listing, output_directory):
            file1, file2 = input_directory_listing
            # This should fail - two files where one is a prefix of the other are declared.
            output_file1 = actions.declare_file(
                "prefix", directory_or_sibling = output_directory
            )
            output_file2 = actions.declare_file(
                "prefix/file.txt", directory_or_sibling = output_directory
            )
            actions.run_shell(
                command = "cp $1 $2",
                inputs = [file1],
                outputs = [output_file1],
                arguments = [file1.path, output_file1.path],
            )
            actions.run_shell(
                command = "cp $1 $2",
                inputs = [file2],
                outputs = [output_file2],
                arguments = [file2.path, output_file2.path],
            )

        def _my_rule_impl(ctx):
            initial_directory = ctx.actions.declare_directory(ctx.attr.name + "_initial_directory")
            ctx.actions.run_shell(
                mnemonic = "SeedData",
                outputs = [initial_directory],
                command = "for i in {1..2}; do echo $i > %s/file_$i.txt; done" % initial_directory.path,
            )

            output_directory = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
            ctx.actions.transform_directory(
                implementation = _transform_dir_impl,
                input_directory = initial_directory,
                output_directory = output_directory,
            )
            return [DefaultInfo(files = depset([output_directory]))]

        my_rule = rule(implementation = _my_rule_impl)
        """);
    RecordingOutErr recordingOutErr = new RecordingOutErr();
    this.outErr = recordingOutErr;
    assertThrows(BuildFailedException.class, () -> buildTarget("//test:target"));
    assertThat(recordingOutErr.errAsLatin1())
        .containsMatch(
            "One of the output paths '.*/test/target_output_dir/prefix' \\(belonging to"
                + " //test:target\\) and '.*/test/target_output_dir/prefix/file.txt' \\(belonging"
                + " to //test:target\\) is a prefix of the other.");
  }

  @Test
  public void testActionConflictCheckingAcrossTransformDirectoryImplementationBoundary()
      throws Exception {
    // An output of an action declared outside of transform_directory can be passed in as an input
    // to transform_directory, but that should not be then passed as an output of an action declared
    // within transform_directory - which is an action conflict.
    write(
        "test/rules.bzl",
        """
        def _transform_dir_impl(
                actions,
                input_directory_listing,
                output_directory,
                some_file):
            for file in input_directory_listing:
                output_file = actions.declare_file(
                    file.tree_relative_path + ".out",
                    directory_or_sibling = output_directory,
                )
                actions.run_shell(
                    command = "cp $1 $2",
                    inputs = [file],
                    # This should fail - `some_file` should not be passed as an output.
                    outputs = [output_file, some_file],
                    arguments = [file.path, output_file.path],
                )

        def _my_rule_impl(ctx):
            some_file = ctx.actions.declare_file(ctx.attr.name + "_some_file.txt")
            ctx.actions.write(
                mnemonic = "SomeData",
                content = "some data",
                output = some_file,
            )

            initial_directory = ctx.actions.declare_directory(ctx.attr.name + "_initial_directory")
            ctx.actions.run_shell(
                mnemonic = "SeedData",
                outputs = [initial_directory],
                command = "for i in {1..5}; do echo $i > %s/file_$i.txt; done" % initial_directory.path,
            )

            output_directory = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
            ctx.actions.transform_directory(
                implementation = _transform_dir_impl,
                input_directory = initial_directory,
                output_directory = output_directory,
                some_file = some_file,
            )
            return [DefaultInfo(files = depset([output_directory]))]

        my_rule = rule(implementation = _my_rule_impl)
        """);
    RecordingOutErr recordingOutErr = new RecordingOutErr();
    this.outErr = recordingOutErr;
    assertThrows(BuildFailedException.class, () -> buildTarget("//test:target"));
    assertThat(recordingOutErr.errAsLatin1())
        .containsMatch(
            "`ctx.actions.transform_directory` function generated an action with an output"
                + " .*test/target_some_file.txt"
                + " that belongs to an external action.");
  }

  @Test
  public void testDisallowedKwargs() throws Exception {
    write(
        "test/rules.bzl",
        """
        def _transform_dir_impl(
            actions,
            input_directory_listing,
            output_directory,
            some_data,
            some_files_to_run):
            pass

        def _my_rule_impl(ctx):
            initial_directory = ctx.actions.declare_directory(ctx.attr.name + "_initial_directory")
            ctx.actions.run_shell(
                mnemonic = "SeedData",
                outputs = [initial_directory],
                command = "for i in {1..2}; do echo $i > %s/file_$i.txt; done" % initial_directory.path,
            )

            output_directory = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
            ctx.actions.transform_directory(
                implementation = _transform_dir_impl,
                input_directory = initial_directory,
                output_directory = output_directory,
                some_files_to_run = ctx.attr._foo_binary.files_to_run,
                some_data = ctx.files.data[0],
                # This should fail - only File and FilesToRunProvider are allowed.
                unallowed_arg = (1, 2, 3),
            )
            return [DefaultInfo(files = depset([output_directory]))]


        my_rule = rule(
            implementation = _my_rule_impl,
            attrs = {
                "_foo_binary": attr.label(
                    executable = True,
                    cfg = "exec",
                    default = Label("//test:foo"),
                ),
                "data": attr.label(
                    default = Label("//test:data"),
                ),
            }
        )
        """);

    RecordingOutErr recordingOutErr = new RecordingOutErr();
    this.outErr = recordingOutErr;
    assertThrows(ViewCreationFailedException.class, () -> buildTarget("//test:target"));
    assertThat(recordingOutErr.errAsLatin1())
        .contains(
            "Only File(s) and FilesToRunProvider(s) are allowed as kwargs in"
                + "ctx.actions.transform_directory(), but got tuple instead");
  }

  @Test
  // SpecialArtifact: CONSTANT_METADATA(s) are not allowed.
  @TestParameters(
      "{input: 'ctx.version_file', output: 'output_directory', additionalOutputs: '[]',"
          + " whatErrored: 'input_directory', unexpectedType: 'File[CONSTANT_METADATA]'}")
  @TestParameters(
      "{input: 'input_directory', output: 'ctx.version_file', additionalOutputs: '[]',"
          + " whatErrored: 'output_directory', unexpectedType: 'File[CONSTANT_METADATA]'}")
  @TestParameters(
      "{input: 'input_directory', output: 'output_directory', additionalOutputs:"
          + " '[ctx.version_file]', whatErrored: 'additional_outputs[0]', unexpectedType:"
          + " 'File[CONSTANT_METADATA]'}")
  // SpecialArtifact: UNRESOLVED_SYMLINK(s) are not allowed.
  @TestParameters(
      "{input: 'symlink', output: 'output_directory', additionalOutputs: '[]',"
          + " whatErrored: 'input_directory', unexpectedType: 'File[UNRESOLVED_SYMLINK]'}")
  @TestParameters(
      "{input: 'input_directory', output: 'symlink', additionalOutputs: '[]',"
          + " whatErrored: 'output_directory', unexpectedType: 'File[UNRESOLVED_SYMLINK]'}")
  @TestParameters(
      "{input: 'input_directory', output: 'output_directory', additionalOutputs:"
          + " '[symlink]', whatErrored: 'additional_outputs[0]', unexpectedType:"
          + " 'File[UNRESOLVED_SYMLINK]'}")
  public void testNonDirectorySpecialArtifactsNotAllowed(
      String input,
      String output,
      String additionalOutputs,
      String whatErrored,
      String unexpectedType)
      throws Exception {
    addOptions("--allow_unresolved_symlinks");
    write(
        "test/rules.bzl",
        String.format(
            """
            def _transform_dir_impl(
                actions,
                input_directory_listing,
                output_directory):
                pass

            def _my_rule_impl(ctx):
                input_directory = ctx.actions.declare_directory(
                    ctx.attr.name + "_initial_directory"
                )
                symlink = ctx.actions.declare_symlink("unresolved_symlink")
                output_directory = ctx.actions.declare_directory(ctx.attr.name + "_output_dir")
                ctx.actions.transform_directory(
                    implementation = _transform_dir_impl,
                    input_directory = %s,
                    output_directory = %s,
                    additional_outputs = %s,
                )
                return [DefaultInfo(files = depset([output_directory]))]

            my_rule = rule(implementation = _my_rule_impl)
            """,
            input, output, additionalOutputs));
    RecordingOutErr recordingOutErr = new RecordingOutErr();
    this.outErr = recordingOutErr;
    assertThrows(ViewCreationFailedException.class, () -> buildTarget("//test:target"));
    assertThat(recordingOutErr.errAsLatin1())
        .contains(
            String.format(
                "Expected directory artifact for %s, but got a %s instead",
                whatErrored, unexpectedType));
  }
}
