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

package com.google.devtools.build.lib.blackbox.tests;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.blackbox.framework.BuilderRunner;
import com.google.devtools.build.lib.blackbox.framework.ProcessResult;
import com.google.devtools.build.lib.blackbox.junit.AbstractBlackBoxTest;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.Test;

/** Integration test for Ninja execution functionality. */
public class NinjaBlackBoxTest extends AbstractBlackBoxTest {
  @Test
  public void testOneTarget() throws Exception {
    context()
        .write(
            WORKSPACE,
            "workspace(name = 'test')",
            "dont_symlink_directories_in_execroot(paths = ['build_dir'])");
    context().write("build_dir/input.txt", "World");
    context()
        .write(
            "build_dir/build.ninja",
            "rule echo",
            "  command = echo \"Hello $$(cat ${in})!\" > ${out}",
            "build hello.txt: echo input.txt");
    context()
        .write(
            "BUILD",
            "ninja_graph(name = 'graph', output_root = 'build_dir',",
            " working_directory = 'build_dir',",
            " main = 'build_dir/build.ninja',",
            " output_root_inputs = ['input.txt'],",
            " output_groups = {'group': ['hello.txt']})");

    BuilderRunner bazel = context().bazel().withFlags("--experimental_ninja_actions");
    assertConfigured(bazel.build("//:graph"));
    Path path = context().resolveExecRootPath(bazel, "build_dir/hello.txt");
    assertThat(path.toFile().exists()).isTrue();
    assertThat(Files.readAllLines(path)).containsExactly("Hello World!");

    // React to input change.
    context().write("build_dir/input.txt", "Sun");
    assertNothingConfigured(bazel.build("//:graph"));
    assertThat(Files.readAllLines(path)).containsExactly("Hello Sun!");

    // React to Ninja file change.
    context()
        .write(
            "build_dir/build.ninja",
            "rule echo",
            "  command = echo \"Hello $$(cat ${in}):)\" > ${out}",
            "build hello.txt: echo input.txt");
    assertConfigured(bazel.build("//:graph"));
    assertThat(Files.readAllLines(path)).containsExactly("Hello Sun:)");
  }

  @Test
  public void testWithoutExperimentalFlag() throws Exception {
    context()
        .write(
            WORKSPACE,
            "workspace(name = 'test')",
            "dont_symlink_directories_in_execroot(paths = ['build_dir'])");
    context().write("build_dir/input.txt", "World");
    context()
        .write(
            "build_dir/build.ninja",
            "rule echo",
            "  command = echo \"Hello $$(cat ${in})!\" > ${out}",
            "build hello.txt: echo input.txt");
    context()
        .write(
            "BUILD",
            "ninja_graph(name = 'graph', output_root = 'build_dir',",
            " working_directory = 'build_dir',",
            " main = 'build_dir/build.ninja',",
            " output_root_inputs = ['input.txt'],",
            " output_groups = {'group': ['hello.txt']})");

    BuilderRunner bazel = context().bazel();
    ProcessResult result = bazel.shouldFail().build("//:graph");
    assertThat(result.errString())
        .contains("name 'dont_symlink_directories_in_execroot' is not defined");
    assertThat(result.errString()).contains("FAILED: Build did NOT complete successfully");
  }

  @Test
  public void testWithoutMainNinja() throws Exception {
    context()
        .write(
            WORKSPACE,
            "workspace(name = 'test')",
            "dont_symlink_directories_in_execroot(paths = ['build_dir'])");
    context().write("build_dir/input.txt", "World");
    context()
        .write(
            "build_dir/build.ninja",
            "rule echo",
            "  command = echo \"Hello $$(cat ${in})!\" > ${out}",
            "build hello.txt: echo input.txt");
    context()
        .write(
            "BUILD",
            "ninja_graph(name = 'graph', output_root = 'build_dir',",
            " working_directory = 'build_dir',",
            " output_root_inputs = ['input.txt'],",
            " output_groups = {'group': ['hello.txt']})");

    BuilderRunner bazel = context().bazel().withFlags("--experimental_ninja_actions");
    ProcessResult result = bazel.shouldFail().build("//:graph");
    assertThat(result.errString())
        .contains("//:graph: missing value for mandatory attribute 'main' in 'ninja_graph' rule");
    assertThat(result.errString()).contains("FAILED: Build did NOT complete successfully");
  }

  @Test
  public void testSourceFileIsMissing() throws Exception {
    context()
        .write(
            WORKSPACE,
            "workspace(name = 'test')",
            "dont_symlink_directories_in_execroot(paths = ['build_dir'])");
    context().write("input.txt", "World");
    context()
        .write(
            "build_dir/build.ninja",
            "rule echo",
            "  command = echo \"Hello $$(cat ${in})!\" > ${out}",
            "build hello.txt: echo ../input.txt");
    context()
        .write(
            "BUILD",
            "ninja_graph(name = 'graph', output_root = 'build_dir',",
            " working_directory = 'build_dir',",
            " main = 'build_dir/build.ninja',",
            " output_groups = {'group': ['hello.txt']})");

    BuilderRunner bazel = context().bazel().withFlags("--experimental_ninja_actions");
    ProcessResult result = bazel.shouldFail().build("//:graph");
    assertThat(result.errString())
        .contains(
            "in ninja_graph rule //:graph: Ninja actions are allowed to create outputs only "
                + "under output_root, path '../input.txt' is not allowed.");
    assertThat(result.errString()).contains("FAILED: Build did NOT complete successfully");
  }

  @Test
  public void testSourceFileIsMissingUnderOutputRoot() throws Exception {
    context()
        .write(
            WORKSPACE,
            "workspace(name = 'test')",
            "dont_symlink_directories_in_execroot(paths = ['build_dir'])");
    context().write("input.txt", "World");
    context()
        .write(
            "build_dir/build.ninja",
            "rule echo",
            "  command = echo \"Hello $$(cat ${in})!\" > ${out}",
            "build hello.txt: echo build_dir/input.txt");
    context()
        .write(
            "BUILD",
            "ninja_graph(name = 'graph', output_root = 'build_dir',",
            " working_directory = 'build_dir',",
            " main = 'build_dir/build.ninja',",
            " output_groups = {'group': ['hello.txt']})");

    BuilderRunner bazel = context().bazel().withFlags("--experimental_ninja_actions");
    ProcessResult result = bazel.shouldFail().build("//:graph");
    assertThat(result.errString())
        .contains(
            "in ninja_graph rule //:graph: The following artifacts do not have a generating "
                + "action in Ninja file: build_dir/build_dir/input.txt");
    assertThat(result.errString()).contains("FAILED: Build did NOT complete successfully");
  }

  private static void assertNothingConfigured(ProcessResult result) {
    assertThat(result.errString())
        .contains("INFO: Analyzed target //:graph (0 packages loaded, 0 targets configured).");
  }

  private static void assertConfigured(ProcessResult result) {
    assertThat(result.errString())
        .doesNotContain(
            "INFO: Analyzed target //:graph (0 packages loaded, 0 targets configured).");
  }

  @Test
  public void testNullBuild() throws Exception {
    context().write(".bazelignore", "build_config");
    context()
        .write(
            WORKSPACE,
            "workspace(name = 'test')",
            "dont_symlink_directories_in_execroot(paths = ['build_config'])");
    // Print nanoseconds fraction of the current time into the output file.
    context()
        .write(
            "build_config/build.ninja",
            "rule echo_time",
            "  command = date +%N >> ${out}",
            "build nano.txt: echo_time");
    context()
        .write(
            "BUILD",
            "ninja_graph(name = 'graph', ",
            "output_root = 'build_config',",
            " working_directory = 'build_config',",
            " main = 'build_config/build.ninja',",
            " output_groups = {'main': ['nano.txt']})");

    BuilderRunner bazel = context().bazel().withFlags("--experimental_ninja_actions");
    assertConfigured(bazel.build("//:graph"));
    Path path = context().resolveExecRootPath(bazel, "build_config/nano.txt");
    assertThat(path.toFile().exists()).isTrue();
    List<String> text = Files.readAllLines(path);
    assertThat(text).isNotEmpty();
    long lastModified = path.toFile().lastModified();

    // Should be null build, as nothing changed.
    assertNothingConfigured(bazel.build("//:graph"));
    assertThat(Files.readAllLines(path)).containsExactly(text.get(0));
    assertThat(path.toFile().lastModified()).isEqualTo(lastModified);
  }
}
