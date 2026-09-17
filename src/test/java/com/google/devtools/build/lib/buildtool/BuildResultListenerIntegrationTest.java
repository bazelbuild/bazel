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

import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ActionStartedEvent;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.analysis.TargetConfiguredEvent;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.buildtool.buildevent.ExecutionPhaseCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.ExecutionStartingEvent;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.skyframe.BuildResultListener;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
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
    interruptModule.clear();
  }

  private final InterruptModule interruptModule = new InterruptModule();

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder().addBlazeModule(interruptModule);
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
                "deps": attr.label_list(),
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

  private static class InterruptModule extends BlazeModule {
    private Class<?> eventToInterruptOn = null;
    private final List<ExecutionPhaseCompleteEvent> executionPhaseCompleteEvents =
        new ArrayList<>();

    void setEventToInterruptOn(Class<?> eventClass) {
      this.eventToInterruptOn = eventClass;
    }

    void clear() {
      this.eventToInterruptOn = null;
      this.executionPhaseCompleteEvents.clear();
    }

    List<ExecutionPhaseCompleteEvent> getExecutionPhaseCompleteEvents() {
      return executionPhaseCompleteEvents;
    }

    @Override
    public void beforeCommand(CommandEnvironment env) {
      env.getEventBus()
          .register(
              new Object() {
                @Subscribe
                public void onExecutionPhaseComplete(ExecutionPhaseCompleteEvent event) {
                  executionPhaseCompleteEvents.add(event);
                }

                @Subscribe
                public void onTargetConfigured(TargetConfiguredEvent event) {
                  if (TargetConfiguredEvent.class.equals(eventToInterruptOn)) {
                    Thread.currentThread().interrupt();
                  }
                }

                @Subscribe
                public void onActionStarted(ActionStartedEvent event) {
                  if (ActionStartedEvent.class.equals(eventToInterruptOn)) {
                    Thread.currentThread().interrupt();
                  }
                }

                @Subscribe
                public void onExecutionStarting(ExecutionStartingEvent event) {
                  if (ExecutionStartingEvent.class.equals(eventToInterruptOn)) {
                    Thread.currentThread().interrupt();
                  }
                }
              });
    }
  }

  @Test
  public void testSuccessfulBuild_postsExecutionPhaseCompleteEvent() throws Exception {
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

    buildTarget("//foo:foo");

    assertThat(interruptModule.getExecutionPhaseCompleteEvents()).hasSize(1);
    assertThat(interruptModule.getExecutionPhaseCompleteEvents().get(0).getTimeInMs())
        .isGreaterThan(0L);
  }

  @Test
  public void testBuildInterrupted_duringAnalysisPhase_recordsAnalysisDurationOnly()
      throws Exception {
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

    interruptModule.setEventToInterruptOn(TargetConfiguredEvent.class);

    assertThrows(InterruptedException.class, () -> buildTarget("//foo:foo"));

    BuildResultListener listener = getCommandEnvironment().getBuildResultListener();
    long analysisDuration = listener.getAnalysisPhaseTimeInMillis();
    assertThat(analysisDuration).isGreaterThan(0L);
    assertThat(listener.getExecutionPhaseTimeInMillis()).isEqualTo(0L);

    Thread.sleep(Duration.ofMillis(10));
    assertThat(listener.getAnalysisPhaseTimeInMillis()).isEqualTo(analysisDuration);
    assertThat(interruptModule.getExecutionPhaseCompleteEvents()).isEmpty();
  }

  @Test
  public void testBuildInterrupted_duringExecutionPhase_recordsBothDurations() throws Exception {
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

    interruptModule.setEventToInterruptOn(ActionStartedEvent.class);

    assertThrows(InterruptedException.class, () -> buildTarget("//foo:foo"));

    BuildResultListener listener = getCommandEnvironment().getBuildResultListener();
    long analysisDuration = listener.getAnalysisPhaseTimeInMillis();
    long executionDuration = listener.getExecutionPhaseTimeInMillis();
    assertThat(analysisDuration).isGreaterThan(0L);
    assertThat(executionDuration).isGreaterThan(0L);

    Thread.sleep(Duration.ofMillis(10));
    assertThat(listener.getAnalysisPhaseTimeInMillis()).isEqualTo(analysisDuration);
    assertThat(listener.getExecutionPhaseTimeInMillis()).isEqualTo(executionDuration);
    assertThat(interruptModule.getExecutionPhaseCompleteEvents()).hasSize(1);
  }

  @Test
  public void testBuildInterrupted_skymeldOverlappingAnalysisAndExecution_recordsBothDurations()
      throws Exception {
    writeMyRuleBzl();
    write(
        "foo/BUILD",
        """
        load("//foo:my_rule.bzl", "my_rule")

        my_rule(
            name = "target1",
            srcs = ["target1.in"],
        )
        my_rule(
            name = "target2",
            srcs = ["target2.in"],
        )
        """);
    write("foo/target1.in");
    write("foo/target2.in");

    interruptModule.setEventToInterruptOn(ActionStartedEvent.class);

    assertThrows(InterruptedException.class, () -> buildTarget("//foo:target1", "//foo:target2"));

    BuildResultListener listener = getCommandEnvironment().getBuildResultListener();
    long analysisDuration = listener.getAnalysisPhaseTimeInMillis();
    long executionDuration = listener.getExecutionPhaseTimeInMillis();
    assertThat(analysisDuration).isGreaterThan(0L);
    assertThat(executionDuration).isGreaterThan(0L);

    Thread.sleep(Duration.ofMillis(10));
    assertThat(listener.getAnalysisPhaseTimeInMillis()).isEqualTo(analysisDuration);
    assertThat(listener.getExecutionPhaseTimeInMillis()).isEqualTo(executionDuration);
    assertThat(interruptModule.getExecutionPhaseCompleteEvents()).hasSize(1);
  }

  @Test
  public void testBuildInterrupted_duringExecutionSetup_recordsExecutionDuration()
      throws Exception {
    if (mergedAnalysisExecution) {
      // ExecutionStartingEvent is not posted in Skymeld.
      return;
    }
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

    interruptModule.setEventToInterruptOn(ExecutionStartingEvent.class);

    assertThrows(InterruptedException.class, () -> buildTarget("//foo:foo"));

    BuildResultListener listener = getCommandEnvironment().getBuildResultListener();
    long analysisDuration = listener.getAnalysisPhaseTimeInMillis();
    long executionDuration = listener.getExecutionPhaseTimeInMillis();
    assertThat(analysisDuration).isGreaterThan(0L);
    assertThat(executionDuration).isGreaterThan(0L);

    Thread.sleep(Duration.ofMillis(10));
    assertThat(listener.getExecutionPhaseTimeInMillis()).isEqualTo(executionDuration);
  }
}
