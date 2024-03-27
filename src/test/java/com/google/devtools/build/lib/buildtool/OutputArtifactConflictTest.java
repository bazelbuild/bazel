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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.TruthJUnit.assume;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertNoEvents;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.MutableActionGraph;
import com.google.devtools.build.lib.analysis.AnalysisFailureEvent;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.TargetCompletedId;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.Analysis.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.Path;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for action conflicts. */
@RunWith(TestParameterInjector.class)
public class OutputArtifactConflictTest extends BuildIntegrationTestCase {

  @TestParameter boolean skymeld;
  @TestParameter boolean minimizeMemory;

  static class AnalysisFailureEventListener extends BlazeModule {

    private final List<TargetCompletedId> eventIds = new ArrayList<>();
    private final List<String> failedTargetNames = new ArrayList<>();

    @Override
    public void beforeCommand(CommandEnvironment env) {
      env.getEventBus().register(this);
    }

    @Subscribe
    public void onAnalysisFailure(AnalysisFailureEvent event) {
      eventIds.add(event.getEventId().getTargetCompleted());
      failedTargetNames.add(event.getFailedTarget().getLabel().toString());
    }
  }

  private final AnalysisFailureEventListener eventListener = new AnalysisFailureEventListener();

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder().addBlazeModule(eventListener);
  }

  @Before
  public void setup() {
    addOptions("--experimental_merged_skyframe_analysis_execution=" + skymeld);
    if (minimizeMemory) {
      addOptions(
          "--notrack_incremental_state",
          "--discard_analysis_cache",
          "--nokeep_state_after_build",
          "--heuristically_drop_nodes");
    }
  }

  private void writeConflictBzl() throws IOException {
    write(
        "foo/conflict.bzl",
        """
        def _conflict_impl(ctx):
            inputs = depset(
                ctx.files.srcs,
                transitive = [dep[DefaultInfo].files for dep in ctx.attr.deps],
            )
            conflict_output = ctx.actions.declare_file("conflict_output")
            other = ctx.actions.declare_file("other" + ctx.attr.name)
            ctx.actions.run_shell(
                inputs = inputs,
                outputs = [conflict_output, other],
                command = "touch %s %s" % (conflict_output.path, other.path),
            )
            return [DefaultInfo(files = depset([conflict_output, other]))]

        my_rule = rule(
            implementation = _conflict_impl,
            attrs = {
                "srcs": attr.label_list(allow_files = True),
                "deps": attr.label_list(providers = [DefaultInfo]),
            },
        )
        """);
  }

  /**
   * Builds the provided targets and asserts expected exceptions.
   *
   * @return the exit code extracted from the failure detail.
   */
  private Code assertThrowsExceptionWhenBuildingTargets(boolean keepGoing, String... targets) {
    FailureDetail failureDetail =
        keepGoing
            ? assertThrows(BuildFailedException.class, () -> buildTarget(targets))
                .getDetailedExitCode()
                .getFailureDetail()
            : assertThrows(ViewCreationFailedException.class, () -> buildTarget(targets))
                .getFailureDetail();
    return Preconditions.checkNotNull(failureDetail).getAnalysis().getCode();
  }

  @Test
  public void testArtifactPrefix(
      @TestParameter boolean keepGoing, @TestParameter boolean modifyBuildFile) throws Exception {
    write("x/y/BUILD", "genrule(name = 'y', outs = ['whatever'], cmd = 'touch $@')");
    if (modifyBuildFile) {
      write("x/BUILD", "genrule(name = 'y', outs = ['not_y'], cmd = 'touch $@')");
      buildTarget("//x:y", "//x/y:y");
      write("x/BUILD", "genrule(name = 'y', outs = ['y'], cmd = 'touch $@')");
    } else {
      write("x/BUILD", "genrule(name = 'y', outs = ['y'], cmd = 'touch $@')");
      buildTarget("//x/y:y");
    }

    assertNoEvents(events.errors());
    assertThat(eventListener.failedTargetNames).isEmpty();

    addOptions("--keep_going=" + keepGoing);
    Code errorCode = assertThrowsExceptionWhenBuildingTargets(keepGoing, "//x/y:y", "//x:y");
    assertThat(errorCode)
        .isEqualTo(keepGoing ? Code.NOT_ALL_TARGETS_ANALYZED : Code.ARTIFACT_PREFIX_CONFLICT);

    if (keepGoing) {
      assertThat(eventListener.failedTargetNames).containsExactly("//x:y", "//x/y:y");
    } else {
      assertThat(eventListener.failedTargetNames).containsAnyOf("//x:y", "//x/y:y");
    }

    events.assertContainsError("One of the output paths '" + TestConstants.PRODUCT_NAME + "-out/");
    events.assertContainsError("/bin/x/y/whatever' (belonging to //x/y:y)");
    events.assertContainsError("/bin/x/y' (belonging to //x:y)");
    events.assertContainsError("is a prefix of the other");
    assertThat(events.errors()).hasSize(1);
  }

  @Test
  public void testAspectArtifactSharesPrefixWithTargetArtifact(
      @TestParameter boolean keepGoing, @TestParameter boolean modifyBuildFile) throws Exception {
    if (modifyBuildFile) {
      write("x/BUILD", "genrule(name = 'y', outs = ['y.out'], cmd = 'touch $@')");
    } else {
      write("x/BUILD", "genrule(name = 'y', outs = ['y.bad'], cmd = 'touch $@')");
    }
    write("x/y/BUILD", "genrule(name = 'y', outs = ['whatever'], cmd = 'touch $@')");
    write(
        "x/aspect.bzl",
        """
        def _aspect_impl(target, ctx):
            if not getattr(ctx.rule.attr, "outs", None):
                return struct(output_groups = {})
            conflict_outputs = list()
            for out in ctx.rule.attr.outs:
                if out.name[1:] == ".bad":
                    aspect_out = ctx.actions.declare_file(out.name[:1])
                    conflict_outputs.append(aspect_out)
                    cmd = "echo %s > %s" % (out.name, aspect_out.path)
                    ctx.actions.run_shell(
                        outputs = [aspect_out],
                        command = cmd,
                    )
            return [OutputGroupInfo(
                files = depset(conflict_outputs),
            )]

        my_aspect = aspect(implementation = _aspect_impl)
        """);

    if (modifyBuildFile) {
      buildTarget("//x/y", "//x:y");
      write("x/BUILD", "genrule(name = 'y', outs = ['y.bad'], cmd = 'touch $@')");
    } else {
      buildTarget("//x/y");
    }
    assertNoEvents(events.errors());
    assertThat(eventListener.failedTargetNames).isEmpty();

    addOptions("--aspects=//x:aspect.bzl%my_aspect", "--output_groups=files");
    addOptions("--keep_going=" + keepGoing);
    Code errorCode = assertThrowsExceptionWhenBuildingTargets(keepGoing, "//x/y", "//x:y");
    assertThat(errorCode)
        .isEqualTo(keepGoing ? Code.NOT_ALL_TARGETS_ANALYZED : Code.ARTIFACT_PREFIX_CONFLICT);
    events.assertContainsError("One of the output paths '" + TestConstants.PRODUCT_NAME + "-out/");
    events.assertContainsError("/bin/x/y/whatever' (belonging to //x/y:y)");
    events.assertContainsError("/bin/x/y' (belonging to //x:y)");
    events.assertContainsError("is a prefix of the other");

    // As we have --output_groups=file, the CTs won't actually be built. Only the
    // AnalysisFailureEvent from Aspect(//x:y) is expected even though there are 2 conflicting
    // actions.
    assertThat(events.errors()).hasSize(1);
    assertThat(eventListener.failedTargetNames).containsExactly("//x:y");
    assertThat(eventListener.eventIds.get(0).getAspect()).isEqualTo("//x:aspect.bzl%my_aspect");
  }

  @Test
  public void testAspectArtifactPrefix(
      @TestParameter boolean keepGoing, @TestParameter boolean modifyBuildFile) throws Exception {
    // TODO(b/245923465) Limitation with Skymeld.
    if (skymeld) {
      assume().that(minimizeMemory).isFalse();
    }
    if (modifyBuildFile) {
      write(
          "x/BUILD",
          """
          genrule(
              name = "y",
              outs = ["y.out"],
              cmd = "touch $@",
          )

          genrule(
              name = "ydir",
              outs = ["y.dir"],
              cmd = "touch $@",
          )
          """);
    } else {
      write(
          "x/BUILD",
          """
          genrule(
              name = "y",
              outs = ["y.bad"],
              cmd = "touch $@",
          )

          genrule(
              name = "ydir",
              outs = ["y.dir"],
              cmd = "touch $@",
          )
          """);
    }
    write(
        "x/aspect.bzl",
        """
        def _aspect_impl(target, ctx):
            if not getattr(ctx.rule.attr, "outs", None):
                return struct(output_groups = {})
            conflict_outputs = list()
            for out in ctx.rule.attr.outs:
                if out.name[1:] == ".bad":
                    aspect_out = ctx.actions.declare_file(out.name[:1])
                    conflict_outputs.append(aspect_out)
                    cmd = "echo %s > %s" % (out.name, aspect_out.path)
                    ctx.actions.run_shell(
                        outputs = [aspect_out],
                        command = cmd,
                    )
                elif out.name[1:] == ".dir":
                    aspect_out = ctx.actions.declare_file(out.name[:1] + "/" + out.name)
                    conflict_outputs.append(aspect_out)
                    out_dir = aspect_out.path[:len(aspect_out.path) - len(out.name) + 1]
                    cmd = "mkdir %s && echo %s > %s" % (out_dir, out.name, aspect_out.path)
                    ctx.actions.run_shell(
                        outputs = [aspect_out],
                        command = cmd,
                    )
            return [OutputGroupInfo(
                files = depset(conflict_outputs),
            )]

        my_aspect = aspect(implementation = _aspect_impl)
        """);

    if (modifyBuildFile) {
      buildTarget("//x:y", "//x:ydir");
      write(
          "x/BUILD",
          """
          genrule(
              name = "y",
              outs = ["y.bad"],
              cmd = "touch $@",
          )

          genrule(
              name = "ydir",
              outs = ["y.dir"],
              cmd = "touch $@",
          )
          """);
    } else {
      buildTarget("//x:y");
    }
    assertNoEvents(events.errors());
    assertThat(eventListener.failedTargetNames).isEmpty();

    addOptions("--aspects=//x:aspect.bzl%my_aspect", "--output_groups=files");
    addOptions("--keep_going=" + keepGoing);
    Code errorCode = assertThrowsExceptionWhenBuildingTargets(keepGoing, "//x:ydir", "//x:y");
    assertThat(errorCode)
        .isEqualTo(keepGoing ? Code.NOT_ALL_TARGETS_ANALYZED : Code.ARTIFACT_PREFIX_CONFLICT);
    events.assertContainsError("One of the output paths '" + TestConstants.PRODUCT_NAME + "-out/");
    events.assertContainsError("bin/x/y' (belonging to //x:y)");
    events.assertContainsError("bin/x/y/y.dir' (belonging to //x:ydir)");
    events.assertContainsError("is a prefix of the other");
    assertThat(events.errors()).hasSize(1);
    assertThat(eventListener.eventIds.get(0).getAspect()).isEqualTo("//x:aspect.bzl%my_aspect");
    if (keepGoing) {
      assertThat(eventListener.failedTargetNames).containsExactly("//x:y", "//x:ydir");
    } else {
      assertThat(eventListener.failedTargetNames).containsAnyOf("//x:y", "//x:ydir");
    }
  }

  @Test
  public void testInvalidatedConflict() throws Exception {
    writeConflictBzl();
    write(
        "foo/BUILD",
        """
        load("//foo:conflict.bzl", "my_rule")

        my_rule(name = "first")

        my_rule(name = "second")
        """);

    assertThrows(
        ViewCreationFailedException.class, () -> buildTarget("//foo:first", "//foo:second"));
    assertThat(eventListener.failedTargetNames).containsAnyOf("//foo:first", "//foo:second");

    write(
        "foo/BUILD",
        """
        load("//foo:conflict.bzl", "my_rule")

        my_rule(name = "first")
        """);
    events.clear();
    buildTarget("//foo:first");

    events.assertNoWarningsOrErrors();
  }

  @Test
  public void testNewTargetConflict(@TestParameter boolean keepGoing) throws Exception {
    addOptions("--keep_going=" + keepGoing);
    writeConflictBzl();
    write(
        "foo/BUILD",
        """
        load("//foo:conflict.bzl", "my_rule")

        my_rule(name = "first")

        my_rule(name = "second")
        """);
    buildTarget("//foo:first");
    events.assertNoWarningsOrErrors();

    Code errorCode =
        assertThrowsExceptionWhenBuildingTargets(keepGoing, "//foo:first", "//foo:second");
    assertThat(errorCode)
        .isEqualTo(keepGoing ? Code.NOT_ALL_TARGETS_ANALYZED : Code.ACTION_CONFLICT);
    events.assertContainsError(
        "file 'foo/conflict_output' is generated by these conflicting actions:");
    assertThat(eventListener.failedTargetNames).hasSize(1);
    assertThat(eventListener.failedTargetNames).containsAnyOf("//foo:first", "//foo:second");
  }

  @Test
  public void testTwoOverlappingBuildsHasNoConflict(@TestParameter boolean keepGoing)
      throws Exception {
    addOptions("--keep_going=" + keepGoing);
    writeConflictBzl();
    write(
        "foo/BUILD",
        """
        load("//foo:conflict.bzl", "my_rule")

        my_rule(name = "first")

        my_rule(name = "second")
        """);

    // Verify that together they fail, even though no new targets have been analyzed
    Code errorCode =
        assertThrowsExceptionWhenBuildingTargets(keepGoing, "//foo:first", "//foo:second");
    assertThat(errorCode)
        .isEqualTo(keepGoing ? Code.NOT_ALL_TARGETS_ANALYZED : Code.ACTION_CONFLICT);
    events.clear();

    // Verify that they still don't fail individually, so no state remains
    buildTarget("//foo:first");
    events.assertNoWarningsOrErrors();
    buildTarget("//foo:second");
    events.assertNoWarningsOrErrors();
  }

  @Test
  public void testFailingTargetsDoNotCauseActionConflicts() throws Exception {
    addOptions("--keep_going");
    write(
        "x/bad_rule.bzl",
        """
        def _impl(ctx):
            return list().this_method_does_not_exist()

        bad_rule = rule(_impl, attrs = {"deps": attr.label_list()})
        """);
    write(
        "x/BUILD",
        """
        load("//x:bad_rule.bzl", "bad_rule")

        cc_binary(
            name = "y",
            srcs = ["y.cc"],
            malloc = "//base:system_malloc",
        )

        bad_rule(
            name = "bad",
            deps = [":y"],
        )
        """);
    write("x/y/y.cc", "");
    write("x/y/BUILD", "cc_library(name = 'y', srcs=['y.cc'])");
    write("x/y.cc", "int main() { return 0; }");

    try {
      buildTarget("//x:y", "//x/y");
      fail();
    } catch (ViewCreationFailedException e) {
      fail("Unexpected artifact prefix conflict: " + e);
    } catch (BuildFailedException e) {
      // Expected.
    }
  }

  // Regression test for b/184944522.
  @Test
  public void testConflictErrorAndAnalysisError() throws Exception {
    addOptions("--keep_going");
    writeConflictBzl();
    write(
        "foo/BUILD",
        """
        load("//foo:conflict.bzl", "my_rule")

        my_rule(name = "first")

        my_rule(name = "second")
        """);
    write("x/BUILD", "sh_library(name = 'x', deps = ['//y:y'])");
    write("y/BUILD", "sh_library(name = 'y', visibility = ['//visibility:private'])");

    assertThrows(
        BuildFailedException.class, () -> buildTarget("//x:x", "//foo:first", "//foo:second"));
    events.assertContainsError(
        "file 'foo/conflict_output' is generated by these conflicting actions:");
    // When targets have conflicting artifacts, one of them "wins" and is successfully built. All
    // of the other targets with conflicting artifacts fail.
    assertThat(eventListener.failedTargetNames).contains("//x:x");
    assertThat(eventListener.failedTargetNames).hasSize(2);
    assertThat(eventListener.failedTargetNames).containsAnyOf("//foo:first", "//foo:second");
  }

  // Verify that an aspect whose analysis is unfinished doesn't fail the conflict reporting process.
  @Test
  public void testConflictErrorAndUnfinishedAspectAnalysis_mergedAnalysisExecution(
      @TestParameter boolean keepGoing) throws Exception {
    assume().that(skymeld).isTrue();
    addOptions("--keep_going=" + keepGoing);
    write(
        "x/aspect.bzl",
        """
        def _aspect_impl(target, ctx):
            if not getattr(ctx.rule.attr, "outs", None):
                return struct(output_groups = {})
            conflict_outputs = list()
            for out in ctx.rule.attr.outs:
                if out.name[1:] == ".bad":
                    aspect_out = ctx.actions.declare_file(out.name[:1])
                    conflict_outputs.append(aspect_out)
                    cmd = "echo %s > %s" % (out.name, aspect_out.path)
                    ctx.actions.run_shell(
                        outputs = [aspect_out],
                        command = cmd,
                    )
            return [OutputGroupInfo(
                files = depset(conflict_outputs),
            )]

        my_aspect = aspect(implementation = _aspect_impl)
        """);

    write(
        "x/BUILD",
        """
        genrule(
            name = "y",
            outs = ["y.bad"],
            cmd = "touch $@",
        )

        sh_library(
            name = "fail_analysis",
            deps = ["//private:y"],
        )
        """);
    write("x/y/BUILD", "genrule(name = 'y', outs = ['whatever'], cmd = 'touch $@')");
    write("private/BUILD", "sh_library(name = 'y', visibility = ['//visibility:private'])");
    addOptions("--aspects=//x:aspect.bzl%my_aspect", "--output_groups=files");

    Code errorCode =
        assertThrowsExceptionWhenBuildingTargets(
            keepGoing, "//x/y:y", "//x:y", "//x:fail_analysis");
    if (keepGoing) {
      assertThat(errorCode).isEqualTo(Code.NOT_ALL_TARGETS_ANALYZED);
      events.assertContainsError(
          "One of the output paths '" + TestConstants.PRODUCT_NAME + "-out/");
      events.assertContainsError("/bin/x/y/whatever' (belonging to //x/y:y)");
      events.assertContainsError("/bin/x/y' (belonging to //x:y)");
      events.assertContainsError("is a prefix of the other");
      events.assertContainsError("Analysis of target '//x:fail_analysis' failed");

      assertThat(eventListener.failedTargetNames).containsExactly("//x:y", "//x:fail_analysis");
    } else {
      assertThat(errorCode)
          .isAnyOf(Code.ARTIFACT_PREFIX_CONFLICT, Code.CONFIGURED_VALUE_CREATION_FAILED);
      assertThat(eventListener.failedTargetNames).containsAnyOf("//x:y", "//x:fail_analysis");
    }
  }

  // This test is documenting current behavior more than enforcing a contract: it might be ok for
  // Bazel to suppress the error message about an action conflict, since the relevant actions are
  // not run in this build. However, that might cause problems for users who aren't immediately
  // alerted when they introduce an action conflict. We already skip exhaustive checks for action
  // conflicts in the name of performance and that has prompted complaints, so suppressing actual
  // conflicts seems like a bad idea.
  //
  // While this test is written with aspects, any actions that generate conflicting outputs but
  // aren't run would exhibit this behavior.
  @Test
  public void unusedActionsStillConflict() throws Exception {
    // TODO(b/245923465) Limitation with Skymeld.
    assume().that(skymeld).isFalse();
    write(
        "foo/aspect.bzl",
        "def _aspect1_impl(target, ctx):",
        "  outfile = ctx.actions.declare_file('aspect.out')",
        "  ctx.actions.run_shell(",
        "    outputs = [outfile],",
        "    progress_message = 'Action for aspect 1',",
        "    command = 'echo \"1\" > ' + outfile.path,",
        "  )",
        "  return [OutputGroupInfo(files1 = [outfile])]",
        "",
        "def _aspect2_impl(target, ctx):",
        "  outfile = ctx.actions.declare_file('aspect.out')",
        "  ctx.actions.run_shell(",
        "    outputs = [outfile],",
        "    progress_message = 'Action for aspect 2',",
        "    command = 'echo \"2\" > ' + outfile.path,",
        "  )",
        "  return [OutputGroupInfo(files2 = [outfile])]",
        "",
        "def _rule_impl(ctx):",
        "  outfile = ctx.actions.declare_file('file.out')",
        "  ctx.actions.run_shell(",
        "    outputs = [outfile],",
        "    progress_message = 'Action for target',",
        "    command = 'touch ' + outfile.path,",
        "  )",
        "  return [DefaultInfo(files = depset([outfile]))]",
        "aspect1 = aspect(implementation = _aspect1_impl)",
        "aspect2 = aspect(implementation = _aspect2_impl)",
        "",
        "bad_rule = rule(implementation = _rule_impl, attrs = {'deps' : attr.label_list(aspects ="
            + " [aspect1, aspect2])})");
    write(
        "foo/BUILD",
        """
        load("//foo:aspect.bzl", "bad_rule")

        sh_library(
            name = "dep",
            srcs = ["dep.sh"],
        )

        bad_rule(
            name = "foo",
            deps = [":dep"],
        )
        """);
    addOptions("--keep_going");
    // If Bazel decides to permit this scenario, the build should succeed instead of throwing here.
    BuildFailedException buildFailedException =
        assertThrows(BuildFailedException.class, () -> buildTarget("//foo:foo"));
    assertThat(buildFailedException)
        .hasMessageThat()
        .contains("command succeeded, but not all targets were analyzed");
    // We successfully built the output file despite the supposed failure.
    Iterable<Artifact> artifacts = getArtifacts("//foo:foo");
    assertThat(artifacts).hasSize(1);
    assertThat(Iterables.getOnlyElement(artifacts).getPath().exists()).isTrue();
    assertThat(
            buildFailedException.getDetailedExitCode().getFailureDetail().getAnalysis().getCode())
        .isEqualTo(FailureDetails.Analysis.Code.NOT_ALL_TARGETS_ANALYZED);
    events.assertContainsError("file 'foo/aspect.out' is generated by these conflicting actions:");
    events.assertContainsError(
        Pattern.compile(
            "Aspects: \\[//foo:aspect.bzl%aspect[12]], \\[//foo:aspect.bzl%aspect[12]]"));
  }

  @Test
  public void testMultipleConflictErrors() throws Exception {
    // TODO(b/245923465) Limitation with Skymeld.
    if (skymeld) {
      assume().that(minimizeMemory).isFalse();
    }
    writeConflictBzl();
    write(
        "foo/BUILD",
        """
        load("//foo:conflict.bzl", "my_rule")

        my_rule(name = "first")

        my_rule(name = "second")
        """);
    write("x/BUILD", "genrule(name = 'y', outs = ['y'], cmd = 'touch $@')");
    write("x/y/BUILD", "genrule(name = 'y', outs = ['whatever'], cmd = 'touch $@')");

    addOptions("--keep_going");

    assertThrows(
        BuildFailedException.class,
        () -> buildTarget("//x/y", "//x:y", "//foo:first", "//foo:second"));
    events.assertContainsError(
        "file 'foo/conflict_output' is generated by these conflicting actions:");
    events.assertContainsError("One of the output paths '" + TestConstants.PRODUCT_NAME + "-out/");
    events.assertContainsError("bin/x/y' (belonging to //x:y)");
    events.assertContainsError("is a prefix of the other");
    // When targets have conflicting artifacts, one of them "wins" and is successfully built. All
    // of the other targets with conflicting artifacts fail.
    assertThat(eventListener.failedTargetNames).containsAtLeast("//x:y", "//x/y:y");
    assertThat(eventListener.failedTargetNames).hasSize(3);
    assertThat(eventListener.failedTargetNames).containsAnyOf("//foo:first", "//foo:second");
  }

  @Test
  public void repeatedConflictBuild() throws Exception {
    writeConflictBzl();
    write(
        "foo/BUILD",
        """
        load("//foo:conflict.bzl", "my_rule")

        my_rule(name = "first")

        my_rule(name = "second")
        """);
    ViewCreationFailedException e =
        assertThrows(
            ViewCreationFailedException.class, () -> buildTarget("//foo:first", "//foo:second"));
    assertThat(e)
        .hasCauseThat()
        .hasCauseThat()
        .isInstanceOf(MutableActionGraph.ActionConflictException.class);
    assertThat(eventListener.failedTargetNames).containsAnyOf("//foo:first", "//foo:second");
    eventListener.failedTargetNames.clear();

    e =
        assertThrows(
            ViewCreationFailedException.class, () -> buildTarget("//foo:first", "//foo:second"));
    assertThat(e)
        .hasCauseThat()
        .hasCauseThat()
        .isInstanceOf(MutableActionGraph.ActionConflictException.class);
    assertThat(eventListener.failedTargetNames).containsAnyOf("//foo:first", "//foo:second");
  }

  @Test
  public void testConflictAfterNullBuild(@TestParameter boolean keepGoing) throws Exception {
    addOptions("--aspects=//x:aspect.bzl%my_aspect", "--output_groups=files");
    addOptions("--keep_going=" + keepGoing);
    write("x/BUILD", "genrule(name = 'y', outs = ['y.out'], cmd = 'touch $@')");
    write("x/y/BUILD", "genrule(name = 'y', outs = ['whatever'], cmd = 'touch $@')");
    write(
        "x/aspect.bzl",
        """
        def _aspect_impl(target, ctx):
            if not getattr(ctx.rule.attr, "outs", None):
                return struct(output_groups = {})
            conflict_outputs = list()
            for out in ctx.rule.attr.outs:
                if out.name[1:] == ".bad":
                    aspect_out = ctx.actions.declare_file(out.name[:1])
                    conflict_outputs.append(aspect_out)
                    cmd = "echo %s > %s" % (out.name, aspect_out.path)
                    ctx.actions.run_shell(
                        outputs = [aspect_out],
                        command = cmd,
                    )
            return [OutputGroupInfo(
                files = depset(conflict_outputs),
            )]

        my_aspect = aspect(implementation = _aspect_impl)
        """);
    // First build: no conflict expected.
    buildTarget("//x/y", "//x:y");
    // Null build
    buildTarget("//x/y", "//x:y");
    assertNoEvents(events.errors());
    assertThat(eventListener.failedTargetNames).isEmpty();

    // Modify BUILD file to introduce a conflict.
    write("x/BUILD", "genrule(name = 'y', outs = ['y.bad'], cmd = 'touch $@')");

    Code errorCode = assertThrowsExceptionWhenBuildingTargets(keepGoing, "//x/y", "//x:y");
    assertThat(errorCode)
        .isEqualTo(keepGoing ? Code.NOT_ALL_TARGETS_ANALYZED : Code.ARTIFACT_PREFIX_CONFLICT);
    events.assertContainsError("One of the output paths '" + TestConstants.PRODUCT_NAME + "-out/");
    events.assertContainsError("/bin/x/y/whatever' (belonging to //x/y:y)");
    events.assertContainsError("/bin/x/y' (belonging to //x:y)");
    events.assertContainsError("is a prefix of the other");
    assertThat(events.errors()).hasSize(1);
    assertThat(eventListener.failedTargetNames).containsExactly("//x:y");
    assertThat(eventListener.eventIds.get(0).getAspect()).isEqualTo("//x:aspect.bzl%my_aspect");
  }

  // There exists a discrepancy between skymeld and noskymeld modes in case of --keep_going.
  // noskymeld: bazel would stop at the end of the analysis phase and build nothing.
  // skymeld: we either finish building one of the 2 conflicting artifacts, or none at all.
  //
  // The overall build would still fail in both cases.
  @Test
  public void testTwoConflictingTargets_keepGoing_behaviorDifferences() throws Exception {
    addOptions("--keep_going");
    write("x/BUILD", "genrule(name = 'y', outs = ['y'], cmd = 'touch $@')");
    write("x/y/BUILD", "genrule(name = 'y', outs = ['whatever'], cmd = 'touch $@')");

    Code errorCode =
        assertThrowsExceptionWhenBuildingTargets(/*keepGoing=*/ true, "//x:y", "//x/y:y");

    assertThat(errorCode).isEqualTo(Code.NOT_ALL_TARGETS_ANALYZED);

    if (minimizeMemory) {
      // The states might have been dropped, so we can't check further here.
      return;
    }

    Path outputXY = Iterables.getOnlyElement(getArtifacts("//x:y")).getPath();
    Path outputXYY = Iterables.getOnlyElement(getArtifacts("//x/y:y")).getPath();

    if (skymeld) {
      // Verify that these 2 conflicting artifacts can't both exist.
      assertThat(outputXYY.isFile() && outputXY.isFile()).isFalse();
    } else {
      // Verify that none of the output artifacts were built.
      assertThat(outputXY.exists()).isFalse();
      assertThat(outputXYY.exists()).isFalse();
    }
  }

  @Test
  public void dependencyHasConflict_keepGoing_bothTopLevelTargetsFail() throws Exception {
    // TODO(b/326363176) Known bug.
    assume().that(minimizeMemory).isFalse();
    addOptions("--keep_going");
    writeConflictBzl();
    write(
        "foo/dummy.bzl",
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
            return [DefaultInfo(files = depset([output]))]

        dummy = rule(
            implementation = _impl,
            attrs = {
                "srcs": attr.label_list(allow_files = True),
                "deps": attr.label_list(providers = [DefaultInfo]),
            },
        )
        """);
    write(
        "foo/BUILD",
        """
        load("//foo:conflict.bzl", "my_rule")
        load("//foo:dummy.bzl", "dummy")

        my_rule(name = "conflict_first")

        my_rule(
            name = "conflict_second",
            deps = [":conflict_first"],
        )

        dummy(
            name = "top_level_a",
            deps = [":conflict_second"],
        )

        dummy(
            name = "top_level_b",
            deps = [":conflict_second"],
        )
        """);
    assertThrows(
        BuildFailedException.class, () -> buildTarget("//foo:top_level_a", "//foo:top_level_b"));
    events.assertContainsError(
        "file 'foo/conflict_output' is generated by these conflicting actions:");
    assertThat(eventListener.failedTargetNames)
        .containsExactly("//foo:top_level_a", "//foo:top_level_b");
  }

  @Test
  public void conflict_noTrackIncrementalState_detected() throws Exception {
    assume().that(minimizeMemory).isTrue();
    writeConflictBzl();
    write(
        "foo/BUILD",
        """
        load("//foo:conflict.bzl", "my_rule")

        my_rule(name = "first")

        my_rule(name = "second")
        """);

    assertThrows(
        ViewCreationFailedException.class, () -> buildTarget("//foo:first", "//foo:second"));
    events.assertContainsError(
        "file 'foo/conflict_output' is generated by these conflicting actions:");
  }

  @Test
  public void directoryWithNestedFile() throws Exception {
    write(
        "foo/conflict.bzl",
        """
        def _impl(ctx):
            dir = ctx.actions.declare_directory(ctx.label.name + ".dir")
            file = ctx.actions.declare_file(ctx.label.name + ".dir/file.txt")
            ctx.actions.run_shell(
                outputs = [dir, file],
                command = "mkdir -p $1 && touch $2",
                arguments = [dir.path, file.path],
            )
            return [DefaultInfo(files = depset([dir, file]))]

        my_rule = rule(implementation = _impl)
        """);
    write(
        "foo/BUILD",
        """
        load(":conflict.bzl", "my_rule")

        my_rule(name = "bar")
        """);

    assertThrows(ViewCreationFailedException.class, () -> buildTarget("//foo:bar"));
    events.assertContainsError("One of the output paths");
    events.assertContainsError("is a prefix of the other");
  }

  @Test
  public void directoryWithNestedDirectory() throws Exception {
    write(
        "foo/conflict.bzl",
        """
        def _impl(ctx):
            dir = ctx.actions.declare_directory(ctx.label.name + ".dir")
            subdir = ctx.actions.declare_directory(ctx.label.name + ".dir/subdir")
            ctx.actions.run_shell(
                outputs = [dir, subdir],
                command = "mkdir -p $1 && mkdir -p $2",
                arguments = [dir.path, subdir.path],
            )
            return [DefaultInfo(files = depset([dir, subdir]))]

        my_rule = rule(implementation = _impl)
        """);
    write(
        "foo/BUILD",
        """
        load(":conflict.bzl", "my_rule")

        my_rule(name = "bar")
        """);

    assertThrows(ViewCreationFailedException.class, () -> buildTarget("//foo:bar"));
    events.assertContainsError("One of the output paths");
    events.assertContainsError("is a prefix of the other");
  }
}
