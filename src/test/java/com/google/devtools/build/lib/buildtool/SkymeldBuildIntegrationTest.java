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

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.TestConstants.WORKSPACE_NAME;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.analysis.AnalysisPhaseCompleteEvent;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.skyframe.SkymeldModule;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.TopLevelEntityAnalysisConcludedEvent;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Integration tests for project Skymeld: interleaving Skyframe's analysis and execution phases. */
@RunWith(TestParameterInjector.class)
public class SkymeldBuildIntegrationTest extends BuildIntegrationTestCase {
  private EventsSubscriber eventsSubscriber;

  @Before
  public void setUp() {
    addOptions("--experimental_merged_skyframe_analysis_execution");
    this.eventsSubscriber = new EventsSubscriber();
    runtimeWrapper.registerSubscriber(eventsSubscriber);
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

  @CanIgnoreReturnValue
  private Path assertSingleOutputBuilt(String target) throws Exception {
    Path singleOutput = Iterables.getOnlyElement(getArtifacts(target)).getPath();
    assertThat(singleOutput.isFile()).isTrue();

    return singleOutput;
  }

  @Test
  public void nobuild_warning() throws Exception {
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
    addOptions("--nobuild");

    RecordingOutErr recordedOutput = divertInfoLogToOutErr();
    BuildResult result = buildTarget("//foo:foo");

    assertThat(result.getSuccess()).isTrue();
    assertThat(recordedOutput.errAsLatin1())
        .containsMatch(
            "--experimental_merged_skyframe_analysis_execution is incompatible with --nobuild"
                + " and will be ignored");
  }

  @Test
  public void multiTargetBuild_success() throws Exception {
    writeMyRuleBzl();
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

    BuildResult result = buildTarget("//foo:foo", "//foo:bar");

    assertThat(result.getSuccess()).isTrue();
    assertSingleOutputBuilt("//foo:foo");
    assertSingleOutputBuilt("//foo:bar");

    assertThat(getLabelsOfAnalyzedTargets()).containsExactly("//foo:foo", "//foo:bar");
    assertThat(getLabelsOfBuiltTargets()).containsExactly("//foo:foo", "//foo:bar");

    assertThat(eventsSubscriber.getTopLevelEntityAnalysisConcludedEvents()).hasSize(2);
    assertSingleAnalysisPhaseCompleteEventWithLabels("//foo:foo", "//foo:bar");

    assertThat(directories.getOutputPath(WORKSPACE_NAME).getRelative("build-info.txt").isFile())
        .isTrue();
    assertThat(
            directories.getOutputPath(WORKSPACE_NAME).getRelative("build-changelist.txt").isFile())
        .isTrue();
  }

  @Test
  public void multiTargetNullIncrementalBuild_success() throws Exception {
    writeMyRuleBzl();
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

    // First build, ignored.
    buildTarget("//foo:foo", "//foo:bar");
    BuildResult result = buildTarget("//foo:foo", "//foo:bar");

    assertThat(result.getSuccess()).isTrue();
    assertSingleOutputBuilt("//foo:foo");
    assertSingleOutputBuilt("//foo:bar");

    assertThat(directories.getOutputPath(WORKSPACE_NAME).getRelative("build-info.txt").isFile())
        .isTrue();
    assertThat(
        directories.getOutputPath(WORKSPACE_NAME).getRelative("build-changelist.txt").isFile())
        .isTrue();
  }

  @Test
  public void aspectAnalysisFailure_consistentWithNonSkymeld(
      @TestParameter boolean keepGoing, @TestParameter boolean mergedAnalysisExecution)
      throws Exception {
    addOptions("--keep_going=" + keepGoing);
    addOptions("--experimental_merged_skyframe_analysis_execution=" + mergedAnalysisExecution);
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
    if (keepGoing) {
      assertThrows(BuildFailedException.class, () -> buildTarget("//foo:foo"));
    } else {
      assertThrows(ViewCreationFailedException.class, () -> buildTarget("//foo:foo"));
    }
    events.assertContainsError("compilation of module 'foo/aspect.bzl' failed");
  }

  @Test
  public void aspectExecutionFailure_consistentWithNonSkymeld(
      @TestParameter boolean keepGoing, @TestParameter boolean mergedAnalysisExecution)
      throws Exception {
    addOptions("--keep_going=" + keepGoing);
    addOptions("--experimental_merged_skyframe_analysis_execution=" + mergedAnalysisExecution);
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
    events.assertContainsError(
        "Action foo/aspect_output failed: (Exit 1): bash failed: error executing Action command");
  }

  @Test
  public void targetExecutionFailure_consistentWithNonSkymeld(
      @TestParameter boolean keepGoing, @TestParameter boolean mergedAnalysisExecution)
      throws Exception {
    addOptions("--keep_going=" + keepGoing);
    addOptions("--experimental_merged_skyframe_analysis_execution=" + mergedAnalysisExecution);
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
    if (keepGoing) {
      assertSingleOutputBuilt("//foo:foo");
    }
    events.assertContainsError(
        "Action foo/execution_failure.out failed: missing input file '//foo:missing'");
  }

  @Test
  public void targetAnalysisFailure_consistentWithNonSkymeld(
      @TestParameter boolean keepGoing, @TestParameter boolean mergedAnalysisExecution)
      throws Exception {
    addOptions("--keep_going=" + keepGoing);
    addOptions("--experimental_merged_skyframe_analysis_execution=" + mergedAnalysisExecution);
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
      assertSingleOutputBuilt("//foo:foo");
    } else {
      assertThrows(
          ViewCreationFailedException.class,
          () -> buildTarget("//foo:foo", "//foo:analysis_failure"));
    }
    events.assertContainsError("rule '//foo:missing' does not exist");
  }

  // Regression test for https://github.com/bazelbuild/bazel/issues/20443
  @Test
  public void testKeepGoingWarningContainsDetails() throws Exception {
    addOptions("--keep_going");
    write(
        "foo/BUILD",
        """
        constraint_setting(name = "incompatible_setting")

        constraint_value(
            name = "incompatible",
            constraint_setting = ":incompatible_setting",
            visibility = ["//visibility:public"],
        )

        cc_library(
            name = "foo",
            srcs = ["foo.cc"],
            target_compatible_with = ["//foo:incompatible"],
        )
        """);
    assertThrows(BuildFailedException.class, () -> buildTarget("//foo:foo"));
    events.assertContainsWarning(
        "errors encountered while analyzing target '//foo:foo', it will not be built.");
    // The details.
    events.assertContainsWarning("Dependency chain:");
  }

  @Test
  public void analysisAndExecutionFailure_keepGoing_bothReported() throws Exception {
    addOptions("--keep_going");
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
            name = "analysis_failure",
            srcs = ["foo.in"],
            deps = [":missing"],
        )
        """);
    write("foo/foo.in");

    assertThrows(
        BuildFailedException.class,
        () -> buildTarget("//foo:analysis_failure", "//foo:execution_failure"));
    events.assertContainsError(
        "Action foo/execution_failure.out failed: missing input file '//foo:missing'");
    events.assertContainsError("rule '//foo:missing' does not exist");

    assertThat(getLabelsOfAnalyzedTargets()).contains("//foo:execution_failure");
    assertThat(getLabelsOfBuiltTargets()).isEmpty();
  }

  @Test
  public void symlinkPlantedLocalAction_success() throws Exception {
    addOptions("--spawn_strategy=standalone");
    write(
        "foo/BUILD",
        """
        genrule(
            name = "foo",
            srcs = ["foo.in"],
            outs = ["foo.out"],
            cmd = "cp $< $@",
        )
        """);
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
                """
                load("//foo:my_rule.bzl", "my_rule")

                my_rule(
                    name = "foo",
                    srcs = ["foo.in"],
                )
                """)
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
  public void symlinksPlantedExceptProductNamePrefixAndIgnoredPaths() throws Exception {
    String productName = getRuntime().getProductName();
    Path execroot = directories.getExecRoot(directories.getWorkspace().getBaseName());
    writeMyRuleBzl();
    Path fooDir =
        write(
                "foo/BUILD",
                """
                load("//foo:my_rule.bzl", "my_rule")

                my_rule(
                    name = "foo",
                    srcs = ["foo.in"],
                )
                """)
            .getParentDirectory();
    write("foo/foo.in");
    Path unusedDir = write("unused/dummy").getParentDirectory();
    write(".bazelignore", "ignored");
    write("ignored/dummy");
    write(productName + "-dir/dummy");

    // Before the build: no symlink.
    assertThat(execroot.getRelative("foo").exists()).isFalse();

    buildTarget("//foo:foo");

    // After the build: symlinks to the source directory, even unused packages, except for those
    // in the .bazelignore file and those with the bazel- prefix.
    assertThat(execroot.getRelative("foo").resolveSymbolicLinks()).isEqualTo(fooDir);
    assertThat(execroot.getRelative("unused").resolveSymbolicLinks()).isEqualTo(unusedDir);
    assertThat(execroot.getRelative("ignored").exists()).isFalse();
    assertThat(execroot.getRelative(productName + "-dir").exists()).isFalse();
  }

  @Test
  public void symlinksReplantedEachBuild() throws Exception {
    Path execroot = directories.getExecRoot(directories.getWorkspace().getBaseName());
    writeMyRuleBzl();
    Path fooDir =
        write(
                "foo/BUILD",
                """
                load("//foo:my_rule.bzl", "my_rule")

                my_rule(
                    name = "foo",
                    srcs = ["foo.in"],
                )
                """)
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

  @Test
  public void targetAnalysisFailure_skymeld_correctAnalysisEvents(@TestParameter boolean keepGoing)
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

      assertThat(eventsSubscriber.getTopLevelEntityAnalysisConcludedEvents()).hasSize(2);
      assertSingleAnalysisPhaseCompleteEventWithLabels("//foo:foo");
    } else {
      assertThrows(
          ViewCreationFailedException.class,
          () -> buildTarget("//foo:foo", "//foo:analysis_failure"));
      assertThat(eventsSubscriber.getAnalysisPhaseCompleteEvents()).isEmpty();
    }
  }

  @Test
  public void aspectAnalysisFailure_skymeld_correctAnalysisEvents(@TestParameter boolean keepGoing)
      throws Exception {
    addOptions("--keep_going=" + keepGoing);
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
    if (keepGoing) {
      assertThrows(BuildFailedException.class, () -> buildTarget("//foo:foo"));
      assertThat(eventsSubscriber.getTopLevelEntityAnalysisConcludedEvents()).hasSize(2);
      assertSingleAnalysisPhaseCompleteEventWithLabels("//foo:foo");
    } else {
      assertThrows(ViewCreationFailedException.class, () -> buildTarget("//foo:foo"));
      assertThat(eventsSubscriber.getAnalysisPhaseCompleteEvents()).isEmpty();
    }
    events.assertContainsError("compilation of module 'foo/aspect.bzl' failed");
  }

  @Test
  public void targetSkipped_skymeld_correctAnalysisEvents(@TestParameter boolean keepGoing)
      throws Exception {
    writeEnvironmentRules();
    addOptions("--keep_going=" + keepGoing);
    write(
        "foo/BUILD",
        """
        sh_library(
            name = "good_bar",
            srcs = ["bar.sh"],
            compatible_with = ["//buildenv:one"],
        )

        sh_library(
            name = "bad_bar",
            srcs = ["bar.sh"],
            compatible_with = ["//buildenv:two"],
        )
        """);
    write("foo/bar.sh");
    addOptions("--target_environment=//buildenv:one");
    if (keepGoing) {
      assertThrows(
          BuildFailedException.class, () -> buildTarget("//foo:good_bar", "//foo:bad_bar"));

      assertThat(eventsSubscriber.getTopLevelEntityAnalysisConcludedEvents()).hasSize(2);
      assertThat(eventsSubscriber.getAnalysisPhaseCompleteEvents()).hasSize(1);
      AnalysisPhaseCompleteEvent analysisPhaseCompleteEvent =
          Iterables.getOnlyElement(eventsSubscriber.getAnalysisPhaseCompleteEvents());
      assertThat(analysisPhaseCompleteEvent.getTimeInMs()).isGreaterThan(0);
      assertThat(getLabelsOfAnalyzedTargets(analysisPhaseCompleteEvent))
          .containsExactly("//foo:good_bar", "//foo:bad_bar");
    } else {
      assertThrows(
          ViewCreationFailedException.class, () -> buildTarget("//foo:good_bar", "//foo:bad_bar"));
      assertThat(eventsSubscriber.getAnalysisPhaseCompleteEvents()).isEmpty();
    }
  }

  @Test
  public void targetWithNoConfiguration_success() throws Exception {
    write("foo/BUILD", "exports_files(['bar.txt'])");
    write("foo/bar.txt", "This is just a test file to pretend to build.");
    BuildResult result = buildTarget("//foo:bar.txt");

    assertThat(result.getSuccess()).isTrue();
  }

  @Test
  public void explain_ignoreSkymeldWithWarning() throws Exception {
    addOptions("--explain=/dev/null");
    write("foo/BUILD", "genrule(name = 'foo', outs = ['foo.out'], cmd = 'touch $@')");
    RecordingOutErr recordedOutput = divertInfoLogToOutErr();
    BuildResult buildResult = buildTarget("//foo");

    assertThat(buildResult.getSuccess()).isTrue();

    assertThat(recordedOutput.errAsLatin1())
        .containsMatch(
            "--experimental_merged_skyframe_analysis_execution is incompatible with --explain"
                + " and will be ignored.");
  }

  @Test
  public void multiplePackagePath_ignoreSkymeldWithWarning() throws Exception {
    write("foo/BUILD", "genrule(name = 'foo', outs = ['foo.out'], cmd = 'touch $@')");
    write("otherroot/bar/BUILD", "genrule(name = 'bar', outs = ['bar.out'], cmd = 'touch $@')");
    addOptions("--package_path=%workspace%:otherroot");

    RecordingOutErr recordedOutput = divertInfoLogToOutErr();
    BuildResult buildResult = buildTarget("//foo", "//bar");

    assertThat(buildResult.getSuccess()).isTrue();

    assertThat(recordedOutput.errAsLatin1())
        .containsMatch(
            "--experimental_merged_skyframe_analysis_execution is incompatible with multiple"
                + " --package_path.*and its value will be ignored.");
  }

  // Regression test for b/245919888.
  @Test
  public void outputFileRemoved_regeneratedWithIncrementalBuild() throws Exception {
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
    Path fooOut = assertSingleOutputBuilt("//foo:foo");

    fooOut.delete();

    BuildResult incrementalBuild = buildTarget("//foo:foo");

    assertThat(incrementalBuild.getSuccess()).isTrue();
    assertSingleOutputBuilt("//foo:foo");
  }

  // Regression test for b/245922900.
  @Test
  public void executionFailure_discardAnalysisCache_doesNotCrash() throws Exception {
    addOptions("--experimental_merged_skyframe_analysis_execution", "--discard_analysis_cache");
    writeExecutionFailureAspectBzl();
    write(
        "foo/BUILD",
        """
        cc_library(
            name = "foo",
            srcs = ["foo.cc"],
            deps = [":bar"],
        )

        cc_library(
            name = "bar",
            srcs = ["bar.cc"],
        )
        """);
    write("foo/foo.cc");
    write("foo/bar.cc");
    addOptions("--aspects=//foo:aspect.bzl%execution_err_aspect", "--output_groups=files");

    // Verify that the build did not crash.
    assertThrows(BuildFailedException.class, () -> buildTarget("//foo:foo"));
    events.assertContainsError(
        "Action foo/aspect_output failed: (Exit 1): bash failed: error executing Action command");
  }

  @Test
  public void targetCycle_doesNotCrash() throws Exception {
    write(
        "a/BUILD",
        """
        alias(
            name = "a",
            actual = ":b",
        )

        alias(
            name = "b",
            actual = ":c",
        )

        alias(
            name = "c",
            actual = ":a",
        )

        filegroup(
            name = "d",
            srcs = [":c"],
        )
        """);
    assertThrows(ViewCreationFailedException.class, () -> buildTarget("//a:d"));
    events.assertContainsError("cycle in dependency graph");
  }

  @Test
  public void analysisOverlapPercentageSanityCheck_success() throws Exception {
    writeMyRuleBzl();
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

    addOptions("--experimental_skymeld_analysis_overlap_percentage=5");
    BuildResult result = buildTarget("//foo:foo", "//foo:bar");

    assertThat(result.getSuccess()).isTrue();
    assertSingleOutputBuilt("//foo:foo");
    assertSingleOutputBuilt("//foo:bar");

    assertThat(getLabelsOfAnalyzedTargets()).containsExactly("//foo:foo", "//foo:bar");
    assertThat(getLabelsOfBuiltTargets()).containsExactly("//foo:foo", "//foo:bar");

    assertThat(eventsSubscriber.getTopLevelEntityAnalysisConcludedEvents()).hasSize(2);
    assertSingleAnalysisPhaseCompleteEventWithLabels("//foo:foo", "//foo:bar");
  }

  // Regression test for b/277783687.
  @Test
  public void targetAnalysisFailureNullBuild_correctErrorsPropagated(
      @TestParameter boolean keepGoing) throws Exception {
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
        """);
    write("foo/foo.in");

    if (keepGoing) {
      assertThrows(BuildFailedException.class, () -> buildTarget("//foo:analysis_failure"));

    } else {
      assertThrows(ViewCreationFailedException.class, () -> buildTarget("//foo:analysis_failure"));
    }
    events.assertContainsError(
        "in deps attribute of my_rule rule //foo:analysis_failure: rule '//foo:missing' does not"
            + " exist");
    events.clear();

    // Null build
    if (keepGoing) {
      assertThrows(BuildFailedException.class, () -> buildTarget("//foo:analysis_failure"));

    } else {
      assertThrows(ViewCreationFailedException.class, () -> buildTarget("//foo:analysis_failure"));
    }
    events.assertContainsError(
        "in deps attribute of my_rule rule //foo:analysis_failure: rule '//foo:missing' does not"
            + " exist");
  }

  // Regression test for b/300391729.
  @Test
  public void executionFailure_keepGoing_doesNotSpamWarnings() throws Exception {
    addOptions("--keep_going");
    writeExecutionFailureAspectBzl();
    write(
        "foo/BUILD",
        """
        cc_library(
            name = "foo",
            srcs = ["foo.cc"],
            deps = [":bar"],
        )

        cc_library(
            name = "bar",
            srcs = ["bar.cc"],
        )
        """);
    write("foo/foo.cc");
    write("foo/bar.cc");
    addOptions("--aspects=//foo:aspect.bzl%execution_err_aspect", "--output_groups=files");

    assertThrows(BuildFailedException.class, () -> buildTarget("//foo/..."));
    // No warnings.
    events.assertNoWarnings();
  }

  // Regression test for b/301289073.
  @Test
  public void conflictCheck_doesNotTimeout() throws Exception {
    addOptions("--keep_going");
    write(
        "foo/BUILD",
        """
        BASE_SIZE = 500

        TOP_SIZE = 100

        genrule(
            name = "base_0",
            outs = ["base_0.txt"],
            cmd = "touch $@",
        )

        [genrule(
            name = "base_%s" % x,
            srcs = ["base_%s.txt" % (x - 1)],
            outs = ["base_%s.txt" % x],
            cmd = "touch $@",
        ) for x in range(1, BASE_SIZE)]

        [genrule(
            name = "level_%s" % y,
            srcs = ["base_%s.txt" % (
                x,
            ) for x in range(0, BASE_SIZE)],
            outs = ["level_%s.txt" % y],
            cmd = "touch $@",
        ) for y in range(0, TOP_SIZE)]

        genrule(
            name = "conflict",
            outs = ["conflict"],
            cmd = "touch $@",
        )
        """);
    write(
        "foo/conflict/BUILD",
        """
        genrule(
            name = "conflict",
            outs = ["conflict"],
            cmd = "touch $@",
        )
        """);

    // Building a set of targets with recursive dependencies that would trivially finish in time
    // with memoization and time out without.
    assertThrows(BuildFailedException.class, () -> buildTarget("//foo/..."));
    events.assertContainsError("is a prefix of the other");
  }

  private void assertSingleAnalysisPhaseCompleteEventWithLabels(String... labels) {
    assertThat(eventsSubscriber.getAnalysisPhaseCompleteEvents()).hasSize(1);
    AnalysisPhaseCompleteEvent analysisPhaseCompleteEvent =
        Iterables.getOnlyElement(eventsSubscriber.getAnalysisPhaseCompleteEvents());
    assertThat(analysisPhaseCompleteEvent.getTimeInMs()).isGreaterThan(0);
    assertThat(getLabelsOfAnalyzedTargets(analysisPhaseCompleteEvent))
        .containsExactlyElementsIn(labels);
  }

  private static ImmutableSet<String> getLabelsOfAnalyzedTargets(AnalysisPhaseCompleteEvent event) {
    return event.getTopLevelTargets().stream()
        .map(x -> x.getOriginalLabel().getCanonicalForm())
        .collect(toImmutableSet());
  }

  private RecordingOutErr divertInfoLogToOutErr() {
    // Divert output into recorder:
    RecordingOutErr recordedOutput = new RecordingOutErr();
    this.outErr = recordedOutput;
    divertLogging(
        Level.INFO, outErr, ImmutableList.of(Logger.getLogger(SkymeldModule.class.getName())));
    return recordedOutput;
  }

  private static final class EventsSubscriber {

    private final List<TopLevelEntityAnalysisConcludedEvent> topLevelEntityAnalysisConcludedEvents =
        Collections.synchronizedList(new ArrayList<>());

    private final List<AnalysisPhaseCompleteEvent> analysisPhaseCompleteEvents =
        Collections.synchronizedList(new ArrayList<>());

    EventsSubscriber() {}

    @Subscribe
    void recordTopLevelEntityAnalysisConcludedEvent(TopLevelEntityAnalysisConcludedEvent event) {
      topLevelEntityAnalysisConcludedEvents.add(event);
    }

    @Subscribe
    void recordAnalysisPhaseCompleteEvent(AnalysisPhaseCompleteEvent event) {
      analysisPhaseCompleteEvents.add(event);
    }

    public List<TopLevelEntityAnalysisConcludedEvent> getTopLevelEntityAnalysisConcludedEvents() {
      return topLevelEntityAnalysisConcludedEvents;
    }

    public List<AnalysisPhaseCompleteEvent> getAnalysisPhaseCompleteEvents() {
      return analysisPhaseCompleteEvents;
    }
  }
}
