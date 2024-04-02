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

import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Integration test for {@link com.google.devtools.build.lib.skyframe.BuildResultListener}. */
@RunWith(TestParameterInjector.class)
public class BuildResultListenerIntegrationTest extends BuildIntegrationTestCase {
  @TestParameter boolean mergedAnalysisExecution;

  @Before
  public final void setUp() {
    addOptions("--experimental_merged_skyframe_analysis_execution=" + mergedAnalysisExecution);
  }

  /** A simple rule that has srcs, deps and writes these attributes to its output. */
  private void writeMyRuleBzl() throws IOException {
    write(
        "foo/my_rule.bzl",
        """
        def _path(file):
            return file.path

        def _impl(ctx):
            inputs = depset(
                ctx.files.srcs,
                transitive = [dep[DefaultInfo].files for dep in ctx.attr.deps],
            )
            output = ctx.actions.declare_file(ctx.attr.name + ".out")
            command = "echo $@ > %s" % (output.path)
            args = ctx.actions.args()
            args.add_all(inputs, map_each = _path)
            ctx.actions.run_shell(
                inputs = inputs,
                outputs = [output],
                command = command,
                arguments = [args],
            )
            return DefaultInfo(files = depset([output]))

        my_rule = rule(
            implementation = _impl,
            attrs = {
                "srcs": attr.label_list(allow_files = True),
                "deps": attr.label_list(providers = ["DefaultInfo"]),
            },
        )
        """);
  }

  private void writeAnalysisFailureAspectBzl() throws IOException {
    write(
        "foo/aspect.bzl",
        """
        def _aspect_impl(target, ctx):
            malformed

        analysis_err_aspect = aspect(implementation = _aspect_impl)
        """);
  }

  private void writeExecutionFailureAspectBzl() throws IOException {
    write(
        "foo/aspect.bzl",
        """
        def _aspect_impl(target, ctx):
            output = ctx.actions.declare_file("aspect_output")
            ctx.actions.run_shell(
                outputs = [output],
                command = "false",
            )
            return [OutputGroupInfo(
                files = depset([output]),
            )]

        execution_err_aspect = aspect(implementation = _aspect_impl)
        """);
  }

  private void writeSuccessfulAspectBzl() throws IOException {
    write(
        "foo/aspect.bzl",
        """
        def _aspect_impl(target, ctx):
            print("hello")
            return []

        successful_aspect = aspect(implementation = _aspect_impl)
        """);
  }

  private void writeEnvironmentRules(String... defaults) throws Exception {
    StringBuilder defaultsBuilder = new StringBuilder();
    for (String defaultEnv : defaults) {
      defaultsBuilder.append("'").append(defaultEnv).append("', ");
    }

    write(
        "buildenv/BUILD",
        "environment_group(",
        "    name = 'group',",
        "    environments = [':one', ':two'],",
        "    defaults = [" + defaultsBuilder + "])",
        "environment(name = 'one')",
        "environment(name = 'two')");
  }

  @Test
  public void multiTargetBuild_success() throws Exception {
    writeMyRuleBzl();
    writeSuccessfulAspectBzl();
    write(
        "foo/BUILD",
        """
        load("//foo:my_rule.bzl", "my_rule")

        my_rule(
            name = "bar",
            srcs = ["bar.in"],
        )

        my_rule(
            name = "foo",
            srcs = ["foo.in"],
        )
        """);
    write("foo/foo.in");
    write("foo/bar.in");
    addOptions("--aspects=//foo:aspect.bzl%successful_aspect");

    BuildResult result = buildTarget("//foo:foo", "//foo:bar");

    assertThat(result.getSuccess()).isTrue();
    assertThat(getLabelsOfAnalyzedTargets()).containsExactly("//foo:foo", "//foo:bar");
    assertThat(getLabelsOfBuiltTargets()).containsExactly("//foo:foo", "//foo:bar");
    assertThat(getLabelsOfAnalyzedAspects()).containsExactly("//foo:foo", "//foo:bar");
    assertThat(getLabelsOfBuiltAspects()).containsExactly("//foo:foo", "//foo:bar");
  }

  @Test
  public void aspectAnalysisFailure_consistentWithNonSkymeld() throws Exception {
    writeMyRuleBzl();
    writeAnalysisFailureAspectBzl();
    write(
        "foo/BUILD",
        """
        load("//foo:my_rule.bzl", "my_rule")

        my_rule(
            name = "foo",
            srcs = ["foo.in"],
        )
        """);
    write("foo/foo.in");

    addOptions("--aspects=//foo:aspect.bzl%analysis_err_aspect", "--output_groups=files");

    assertThrows(ViewCreationFailedException.class, () -> buildTarget("//foo:foo"));

    assertThat(getLabelsOfAnalyzedAspects()).isEmpty();
  }

  @Test
  public void aspectExecutionFailure_consistentWithNonSkymeld(@TestParameter boolean keepGoing)
      throws Exception {
    addOptions("--keep_going=" + keepGoing);
    writeMyRuleBzl();
    writeExecutionFailureAspectBzl();
    write(
        "foo/BUILD",
        """
        load("//foo:my_rule.bzl", "my_rule")

        my_rule(
            name = "foo",
            srcs = ["foo.in"],
        )
        """);
    write("foo/foo.in");

    addOptions("--aspects=//foo:aspect.bzl%execution_err_aspect", "--output_groups=files");

    assertThrows(BuildFailedException.class, () -> buildTarget("//foo:foo"));

    assertThat(getLabelsOfAnalyzedAspects()).contains("//foo:foo");
    assertThat(getLabelsOfBuiltAspects()).isEmpty();
  }

  @Test
  public void targetExecutionFailure_consistentWithNonSkymeld(@TestParameter boolean keepGoing)
      throws Exception {
    addOptions("--keep_going=" + keepGoing);
    writeMyRuleBzl();
    write(
        "foo/BUILD",
        """
        load("//foo:my_rule.bzl", "my_rule")

        my_rule(
            name = "execution_failure",
            srcs = ["missing"],
        )

        my_rule(
            name = "foo",
            srcs = ["foo.in"],
        )
        """);
    write("foo/foo.in");

    assertThrows(
        BuildFailedException.class, () -> buildTarget("//foo:foo", "//foo:execution_failure"));

    assertThat(getLabelsOfAnalyzedTargets()).contains("//foo:execution_failure");
    if (keepGoing) {
      assertThat(getLabelsOfAnalyzedTargets())
          .containsExactly("//foo:foo", "//foo:execution_failure");
      assertThat(getLabelsOfBuiltTargets()).containsExactly("//foo:foo");
    }
  }

  @Test
  public void targetAnalysisFailure_consistentWithNonSkymeld(@TestParameter boolean keepGoing)
      throws Exception {
    addOptions("--keep_going=" + keepGoing);
    writeMyRuleBzl();
    write(
        "foo/BUILD",
        """
        load("//foo:my_rule.bzl", "my_rule")

        my_rule(
            name = "analysis_failure",
            srcs = ["foo.in"],
            deps = [":missing"],
        )

        my_rule(
            name = "foo",
            srcs = ["foo.in"],
        )
        """);
    write("foo/foo.in");

    if (keepGoing) {
      assertThrows(
          BuildFailedException.class, () -> buildTarget("//foo:foo", "//foo:analysis_failure"));
      assertThat(getLabelsOfAnalyzedTargets()).contains("//foo:foo");
      assertThat(getLabelsOfBuiltTargets()).containsExactly("//foo:foo");
    } else {
      assertThrows(
          ViewCreationFailedException.class,
          () -> buildTarget("//foo:foo", "//foo:analysis_failure"));
      assertThat(getBuildResultListener().getBuiltTargets()).isEmpty();
    }
  }

  @Test
  public void targetSkipped_consistentWithNonSkymeld() throws Exception {
    writeEnvironmentRules();
    write(
        "foo/BUILD",
        """
        sh_library(
            name = "good",
            srcs = ["bar.sh"],
            restricted_to = ["//buildenv:one"],
        )

        sh_library(
            name = "bad",
            srcs = ["bar.sh"],
            compatible_with = ["//buildenv:two"],
        )
        """);
    write("foo/bar.sh");
    addOptions("--auto_cpu_environment_group=//buildenv:group", "--cpu=one");

    buildTarget("//foo:all");
    assertThat(getLabelsOfAnalyzedTargets()).containsExactly("//foo:good", "//foo:bad");
    assertThat(getLabelsOfBuiltTargets()).containsExactly("//foo:good");
    assertThat(getLabelsOfSkippedTargets()).containsExactly("//foo:bad");
  }

  @Test
  public void nullIncrementalBuild_correctAnalyzedAndBuiltTargets() throws Exception {
    writeMyRuleBzl();
    write(
        "foo/BUILD",
        """
        load("//foo:my_rule.bzl", "my_rule")

        my_rule(
            name = "foo",
            srcs = ["foo.in"],
        )
        """);
    write("foo/foo.in");

    BuildResult result = buildTarget("//foo:foo");

    assertThat(result.getSuccess()).isTrue();
    assertThat(getLabelsOfAnalyzedTargets()).containsExactly("//foo:foo");
    assertThat(getLabelsOfBuiltTargets()).containsExactly("//foo:foo");

    result = buildTarget("//foo:foo");

    assertThat(result.getSuccess()).isTrue();
    assertThat(getLabelsOfAnalyzedTargets()).containsExactly("//foo:foo");
    assertThat(getLabelsOfBuiltTargets()).containsExactly("//foo:foo");
  }
}
