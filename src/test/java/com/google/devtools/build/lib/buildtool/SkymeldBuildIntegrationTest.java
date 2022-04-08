// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.vfs.Path;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Integration tests for project Skymeld: interleaving Skyframe's analysis and execution phases. */
@RunWith(TestParameterInjector.class)
public class SkymeldBuildIntegrationTest extends BuildIntegrationTestCase {

  @Before
  public void setUp() {
    addOptions("--experimental_merged_skyframe_analysis_execution");
  }

  /** A simple rule that has srcs, deps and writes these attributes to its output. */
  private void writeMyRuleBzl() throws IOException {
    write(
        "foo/my_rule.bzl",
        "def _path(file):",
        "  return file.path",
        "def _impl(ctx):",
        "  inputs = depset(",
        "    ctx.files.srcs, transitive = [dep[DefaultInfo].files for dep in ctx.attr.deps])",
        "  output = ctx.actions.declare_file(ctx.attr.name + '.out')",
        "  command = 'echo $@ > %s' % (output.path)",
        "  args = ctx.actions.args()",
        "  args.add_all(inputs, map_each=_path)",
        "  ctx.actions.run_shell(",
        "    inputs = inputs,",
        "    outputs = [output],",
        "    command = command,",
        "    arguments = [args]",
        "  )",
        "  return DefaultInfo(files = depset([output]))",
        "",
        "my_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'srcs': attr.label_list(allow_files = True),",
        "    'deps': attr.label_list(providers = ['DefaultInfo']),",
        "  }",
        ")");
  }

  private void assertSingleOutputBuilt(String target) throws Exception {
    assertThat(Iterables.getOnlyElement(getArtifacts(target)).getPath().isFile()).isTrue();
  }

  @Test
  public void multiTargetBuild_success() throws Exception {
    writeMyRuleBzl();
    write(
        "foo/BUILD",
        "load('//foo:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'bar', srcs = ['bar.in'])",
        "my_rule(name = 'foo', srcs = ['foo.in'])");
    write("foo/foo.in");
    write("foo/bar.in");

    BuildResult result = buildTarget("//foo:foo", "//foo:bar");

    assertThat(result.getSuccess()).isTrue();
    assertSingleOutputBuilt("//foo:foo");
    assertSingleOutputBuilt("//foo:bar");
  }

  @Test
  public void onlyExecutionFailure(@TestParameter boolean keepGoing) throws Exception {
    addOptions("--keep_going=" + keepGoing);
    writeMyRuleBzl();
    write(
        "foo/BUILD",
        "load('//foo:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'execution_failure', srcs = ['missing'])",
        "my_rule(name = 'foo', srcs = ['foo.in'])");
    write("foo/foo.in");

    assertThrows(
        BuildFailedException.class, () -> buildTarget("//foo:foo", "//foo:execution_failure"));
    if (keepGoing) {
      assertSingleOutputBuilt("//foo:foo");
    }
    events.assertContainsError(
        "Action foo/execution_failure.out failed: missing input file '//foo:missing'");
  }

  @Test
  public void onlyAnalysisFailure(@TestParameter boolean keepGoing) throws Exception {
    addOptions("--keep_going=" + keepGoing);
    writeMyRuleBzl();
    write(
        "foo/BUILD",
        "load('//foo:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'analysis_failure', srcs = ['foo.in'], deps = [':missing'])",
        "my_rule(name = 'foo', srcs = ['foo.in'])");
    write("foo/foo.in");

    if (keepGoing) {
      assertThrows(
          BuildFailedException.class, () -> buildTarget("//foo:foo", "//foo:analysis_failure"));
      assertSingleOutputBuilt("//foo:foo");
    } else {
      assertThrows(
          ViewCreationFailedException.class,
          () -> buildTarget("//foo:foo", "//foo:analysis_failure"));
    }
    events.assertContainsError("rule '//foo:missing' does not exist");
  }

  @Test
  public void analysisAndExecutionFailure_keepGoing_bothReported() throws Exception {
    addOptions("--keep_going");
    writeMyRuleBzl();
    write(
        "foo/BUILD",
        "load('//foo:my_rule.bzl', 'my_rule')",
        "my_rule(name = 'execution_failure', srcs = ['missing'])",
        "my_rule(name = 'analysis_failure', srcs = ['foo.in'], deps = [':missing'])");
    write("foo/foo.in");

    assertThrows(
        BuildFailedException.class,
        () -> buildTarget("//foo:analysis_failure", "//foo:execution_failure"));
    events.assertContainsError(
        "Action foo/execution_failure.out failed: missing input file '//foo:missing'");
    events.assertContainsError("rule '//foo:missing' does not exist");
  }

  @Test
  public void symlinkPlantedLocalAction_success() throws Exception {
    addOptions("--spawn_strategy=standalone");
    write(
        "foo/BUILD",
        "genrule(",
        "  name = 'foo',",
        "  srcs = ['foo.in'],",
        "  outs = ['foo.out'],",
        "  cmd = 'cp $< $@'",
        ")");
    write("foo/foo.in");

    BuildResult result = buildTarget("//foo:foo");

    assertThat(result.getSuccess()).isTrue();
    assertSingleOutputBuilt("//foo:foo");
  }

  @Test
  public void symlinksPlanted() throws Exception {
    Path execroot = directories.getExecRoot(directories.getWorkspace().getBaseName());
    writeMyRuleBzl();
    Path fooDir =
        write(
                "foo/BUILD",
                "load('//foo:my_rule.bzl', 'my_rule')",
                "my_rule(name = 'foo', srcs = ['foo.in'])")
            .getParentDirectory();
    write("foo/foo.in");
    Path unusedDir = write("unused/dummy").getParentDirectory();

    // Before the build: no symlink.
    assertThat(execroot.getRelative("foo").exists()).isFalse();

    buildTarget("//foo:foo");

    // After the build: symlinks to the source directory, even unused packages.
    assertThat(execroot.getRelative("foo").resolveSymbolicLinks()).isEqualTo(fooDir);
    assertThat(execroot.getRelative("unused").resolveSymbolicLinks()).isEqualTo(unusedDir);
  }

  @Test
  public void symlinksReplantedEachBuild() throws Exception {
    Path execroot = directories.getExecRoot(directories.getWorkspace().getBaseName());
    writeMyRuleBzl();
    Path fooDir =
        write(
                "foo/BUILD",
                "load('//foo:my_rule.bzl', 'my_rule')",
                "my_rule(name = 'foo', srcs = ['foo.in'])")
            .getParentDirectory();
    write("foo/foo.in");
    Path unusedDir = write("unused/dummy").getParentDirectory();

    buildTarget("//foo:foo");

    // After the 1st build: symlinks to the source directory, even unused packages.
    assertThat(execroot.getRelative("foo").resolveSymbolicLinks()).isEqualTo(fooDir);
    assertThat(execroot.getRelative("unused").resolveSymbolicLinks()).isEqualTo(unusedDir);

    unusedDir.deleteTree();

    buildTarget("//foo:foo");

    // After the 2nd build: symlink to unusedDir is gone, since the package itself was deleted.
    assertThat(execroot.getRelative("foo").resolveSymbolicLinks()).isEqualTo(fooDir);
    assertThat(execroot.getRelative("unused").exists()).isFalse();
  }
}
