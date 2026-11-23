// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.same;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.analysis.AspectCompleteEvent;
import com.google.devtools.build.lib.analysis.TargetCompleteEvent;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.TargetSummaryId;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildStartingEvent;
import com.google.devtools.build.lib.buildtool.buildevent.TestFilteringCompleteEvent;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.runtime.commands.BuildCommand;
import com.google.devtools.build.lib.runtime.commands.TestCommand;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.TopLevelTargetPendingExecutionEvent;
import com.google.devtools.build.lib.skyframe.ToplevelAspectsIdentifiedEvent;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.ArgumentCaptor;

/** Unit tests for {@link TargetSummaryPublisher}. */
@RunWith(TestParameterInjector.class)
public final class TargetSummaryPublisherIntegrationTest extends BuildIntegrationTestCase {

  private static final String TARGET_SUMMARY_FLAG = "--experimental_bep_target_summary";

  private RecordingOutErr recordingOutErr;
  private BlazeCommandDispatcher dispatcher;
  private Object eventListener;
  private EventBus eventBus;

  private static final class RecordingEventListener {
    BuildStartingEvent buildStarting;
    TestFilteringCompleteEvent testFilteringComplete;
    BuildCompleteEvent buildComplete;
    final Set<TopLevelTargetPendingExecutionEvent> topLevelTargetPendingExecutions =
        new HashSet<>();
    final List<AspectCompleteEvent> aspectCompletions = new ArrayList<>();
    final List<ToplevelAspectsIdentifiedEvent> toplevelAspectsIdentifiedEvents = new ArrayList<>();
    final List<TargetCompleteEvent> targetCompletions = new ArrayList<>();
    final List<TestSummary> testSummaries = new ArrayList<>();
    final List<TargetSummaryEvent> targetSummaries = new ArrayList<>();
    final List<Object> relevantEvents = Collections.synchronizedList(new ArrayList<>());

    @Subscribe
    public void buildStarting(BuildStartingEvent event) {
      assertWithMessage("Got duplicate %s", event).that(buildStarting).isNull();
      buildStarting = event;
      relevantEvents.add(event);
    }

    @Subscribe
    public void testFilteringComplete(TestFilteringCompleteEvent event) {
      assertWithMessage("Got duplicate %s", event).that(testFilteringComplete).isNull();
      testFilteringComplete = event;
      relevantEvents.add(event);
    }

    @Subscribe
    public void topLevelTargetPendingExecution(TopLevelTargetPendingExecutionEvent event) {
      topLevelTargetPendingExecutions.add(event);
    }

    @Subscribe
    public void topLevelAspectsDetailsComplete(ToplevelAspectsIdentifiedEvent event) {
      this.toplevelAspectsIdentifiedEvents.add(event);
    }

    @Subscribe
    public void targetComplete(TargetCompleteEvent event) {
      targetCompletions.add(event);
      relevantEvents.add(event);
    }

    @Subscribe
    public void aspectComplete(AspectCompleteEvent event) {
      aspectCompletions.add(event);
      relevantEvents.add(event);
    }

    @Subscribe
    public void buildComplete(BuildCompleteEvent event) {
      assertWithMessage("Got duplicate %s", event).that(buildComplete).isNull();
      buildComplete = event;
      relevantEvents.add(event);
    }

    @Subscribe
    public void testSummary(TestSummary event) {
      testSummaries.add(event);
      relevantEvents.add(event);
    }

    @Subscribe
    public void targetSummary(TargetSummaryEvent event) {
      targetSummaries.add(event);
      relevantEvents.add(event);
    }
  }

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder()
        .addBlazeModule(
            new BlazeModule() {
              @Override
              public void beforeCommand(CommandEnvironment env) {
                TargetSummaryPublisherIntegrationTest.this.eventBus = env.getEventBus();
                if (eventListener != null) {
                  eventBus.register(eventListener);
                }
                eventBus.register(
                    new TargetSummaryPublisher(
                        env.getEventBus(), env::withMergedAnalysisAndExecutionSourceOfTruth));
              }
            });
  }

  @Before
  public void createDispatcher() throws Exception {
    AnalysisMock.get().setupMockToolsRepository(mockToolsConfig);
    getRuntime().overrideCommands(ImmutableList.of(new BuildCommand(), new TestCommand()));
    recordingOutErr = new RecordingOutErr();
    dispatcher = new BlazeCommandDispatcher(getRuntime());
  }

  @After
  public void resetEventListener() {
    eventListener = null;
  }

  private int runTest(String... args) throws Exception {
    ImmutableList.Builder<String> params =
        ImmutableList.<String>builder()
            .add(
                "test",
                "--default_visibility=public",
                "--test_output=all",
                "--noexperimental_collect_system_network_usage",
                "--client_env=PATH=/bin:/usr/bin:/usr/sbin:/sbin")
            .addAll(TestConstants.PRODUCT_SPECIFIC_FLAGS)
            .addAll(TestConstants.PRODUCT_SPECIFIC_BUILD_LANG_OPTIONS);
    if (AnalysisMock.get().isThisBazel()) {
      params.add("--override_repository=bazel_tools=embedded_tools");
    }
    params.add(args);
    return dispatcher
        .exec(params.build(), "test", recordingOutErr)
        .getExitCode()
        .getNumericExitCode();
  }

  @Before
  public void configureBuildOptions() {
    addOptions(TARGET_SUMMARY_FLAG);
  }

  @Test
  public void buildTestTarget_targetSummaryWithoutTestSummary() throws Exception {
    RecordingEventListener listener = new RecordingEventListener();
    eventListener = listener;

    // Run a simple test so that we can swipe its events
    write("a/a.sh", "exit 0").setExecutable(true);
    write(
        "a/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name='a', srcs=['a.sh'])");
    assertThat(buildTarget("//a:a").getSuccess()).isTrue();
    assertThat(listener.testSummaries).isEmpty();
    assertThat(listener.targetCompletions).hasSize(1);
    assertThat(listener.targetSummaries).hasSize(1);

    TargetCompleteEvent targetComplete = listener.targetCompletions.get(0);
    assertThat(targetComplete.failed()).isFalse();

    TargetSummaryEvent targetSummary = listener.targetSummaries.get(0);
    assertThat(targetSummary.isOverallBuildSuccess()).isTrue();
    assertThat(targetSummary.getOverallTestStatus()).isNull();
    assertThat(targetSummary.postedAfter()).containsExactly(targetComplete.getEventId());
    assertThat(targetComplete.getChildrenEvents()).containsExactly(targetSummary.getEventId());

    assertThat(listener.relevantEvents).containsAtLeast(targetComplete, targetSummary).inOrder();
  }

  @Test
  public void successfulTest_targetSummaryAfterTestSummary() throws Exception {
    RecordingEventListener listener = new RecordingEventListener();
    eventListener = listener;

    // Run a simple test so that we can swipe its events
    write("a/a.sh", "exit 0").setExecutable(true);
    write(
        "a/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name='a', srcs=['a.sh'])");
    assertThat(runTest(TARGET_SUMMARY_FLAG, "//a:a")).isEqualTo(0);
    assertThat(listener.testSummaries).hasSize(1);
    assertThat(listener.targetCompletions).hasSize(1);
    assertThat(listener.targetSummaries).hasSize(1);

    TargetCompleteEvent targetComplete = listener.targetCompletions.get(0);
    assertThat(targetComplete.failed()).isFalse();

    TestSummary testSummary = listener.testSummaries.get(0);
    assertThat(testSummary.getStatus()).isEqualTo(BlazeTestStatus.PASSED);

    TargetSummaryEvent targetSummary = listener.targetSummaries.get(0);
    assertThat(targetSummary.isOverallBuildSuccess()).isTrue();
    assertThat(targetSummary.getOverallTestStatus()).isEqualTo(BlazeTestStatus.PASSED);
    assertThat(targetSummary.postedAfter())
        .containsExactly(targetComplete.getEventId(), testSummary.getEventId());
    assertThat(targetComplete.getChildrenEvents())
        .containsAtLeast(targetSummary.getEventId(), testSummary.getEventId());

    assertThat(listener.relevantEvents)
        .containsAtLeast(targetComplete, testSummary, targetSummary)
        .inOrder();

    // To test TargetSummaryPublisher's handling of BuildCompleteEvent, replay just that event
    // with new publisher instance and mocked event bus.
    EventBus eventBus = mock(EventBus.class);
    ArgumentCaptor<TargetSummaryEvent> actual = ArgumentCaptor.forClass(TargetSummaryEvent.class);
    TargetSummaryPublisher replay =
        new TargetSummaryPublisher(
            eventBus, getCommandEnvironment()::withMergedAnalysisAndExecutionSourceOfTruth);
    replay.buildStarting(listener.buildStarting);
    replayAspectsIdentified(listener, replay);
    replayPopulateTargets(listener, replay);
    replay.buildCompleteEvent(listener.buildComplete);
    // TestSummary event needed before TargetSummary is published
    verify(eventBus, never()).post(any());
    replay.testSummaryEvent(testSummary);
    verify(eventBus).post(actual.capture());
    assertContentEquals(targetSummary, actual.getValue());
  }

  @Test
  public void testFailure_targetSummaryAfterTestSummary() throws Exception {
    RecordingEventListener listener = new RecordingEventListener();
    eventListener = listener;

    // Run a simple test so that we can swipe its events
    write("a/a.sh", "exit 1").setExecutable(true);
    write(
        "a/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name='a', srcs=['a.sh'])");
    assertThat(runTest(TARGET_SUMMARY_FLAG, "//a:a")).isEqualTo(3);
    assertThat(listener.testSummaries).hasSize(1);
    assertThat(listener.targetCompletions).hasSize(1);
    assertThat(listener.targetSummaries).hasSize(1);

    TargetCompleteEvent targetComplete = listener.targetCompletions.get(0);
    assertThat(targetComplete.failed()).isFalse();

    TestSummary testSummary = listener.testSummaries.get(0);
    assertThat(testSummary.getStatus()).isEqualTo(BlazeTestStatus.FAILED);

    TargetSummaryEvent targetSummary = listener.targetSummaries.get(0);
    assertThat(targetSummary.isOverallBuildSuccess()).isTrue();
    assertThat(targetSummary.getOverallTestStatus()).isEqualTo(BlazeTestStatus.FAILED);
    assertThat(targetSummary.postedAfter())
        .containsExactly(targetComplete.getEventId(), testSummary.getEventId());
    assertThat(targetComplete.getChildrenEvents())
        .containsAtLeast(targetSummary.getEventId(), testSummary.getEventId());

    assertThat(listener.relevantEvents)
        .containsAtLeast(targetComplete, testSummary, targetSummary)
        .inOrder();

    // To test TargetSummaryPublisher's handling of BuildCompleteEvent, replay just that event
    // with new publisher instance and mocked event bus.
    EventBus eventBus = mock(EventBus.class);
    ArgumentCaptor<TargetSummaryEvent> actual = ArgumentCaptor.forClass(TargetSummaryEvent.class);
    TargetSummaryPublisher replay =
        new TargetSummaryPublisher(
            eventBus, getCommandEnvironment()::withMergedAnalysisAndExecutionSourceOfTruth);
    replay.buildStarting(listener.buildStarting);
    replayAspectsIdentified(listener, replay);
    replayPopulateTargets(listener, replay);
    replay.buildCompleteEvent(listener.buildComplete);
    // TestSummary event needed before TargetSummary is published
    verify(eventBus, never()).post(any());
    replay.testSummaryEvent(testSummary);
    verify(eventBus).post(actual.capture());
    assertContentEquals(targetSummary, actual.getValue());
  }

  @Test
  public void testBuildFailure_targetSummaryAfterTestSummary() throws Exception {
    RecordingEventListener listener = new RecordingEventListener();
    eventListener = listener;

    // Run a simple test so that we can swipe its events
    write("a/a.sh", "exit 0"); // should be executable but isn't
    write(
        "a/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name='a', srcs=['a.sh'])");
    assertThat(runTest(TARGET_SUMMARY_FLAG, "//a:a")).isEqualTo(1);
    assertThat(listener.testSummaries).hasSize(1);
    assertThat(listener.targetCompletions).hasSize(1);
    assertThat(listener.targetSummaries).hasSize(1);

    TargetCompleteEvent targetComplete = listener.targetCompletions.get(0);
    assertThat(targetComplete.failed()).isTrue();

    TestSummary testSummary = listener.testSummaries.get(0);
    assertThat(testSummary.getStatus()).isEqualTo(BlazeTestStatus.FAILED_TO_BUILD);

    TargetSummaryEvent targetSummary = listener.targetSummaries.get(0);
    assertThat(targetSummary.isOverallBuildSuccess()).isFalse();
    assertThat(targetSummary.getOverallTestStatus()).isNull(); // test status not recorded
    assertThat(targetSummary.postedAfter())
        .containsExactly(targetComplete.getEventId(), testSummary.getEventId());
    assertThat(targetComplete.getChildrenEvents()).contains(targetSummary.getEventId());
    assertThat(targetComplete.getChildrenEvents()).doesNotContain(testSummary.getEventId());

    assertThat(listener.relevantEvents).containsAtLeast(targetComplete, targetSummary).inOrder();

    // To test TargetSummaryPublisher's handling of BuildCompleteEvent, replay just that event
    // with new publisher instance and mocked event bus.
    EventBus eventBus = mock(EventBus.class);
    ArgumentCaptor<TargetSummaryEvent> actual = ArgumentCaptor.forClass(TargetSummaryEvent.class);
    TargetSummaryPublisher replay =
        new TargetSummaryPublisher(
            eventBus, getCommandEnvironment()::withMergedAnalysisAndExecutionSourceOfTruth);
    replay.buildStarting(listener.buildStarting);
    replayAspectsIdentified(listener, replay);
    replayPopulateTargets(listener, replay);
    replay.buildCompleteEvent(listener.buildComplete);
    // TestSummary event not needed since BuildCompleteEvent indicates build failure
    verify(eventBus).post(actual.capture());
    assertContentEquals(targetSummary, actual.getValue());
    replay.testSummaryEvent(testSummary);
    verify(eventBus).post(same(actual.getValue())); // still only match the first invocation *once*
  }

  @Test
  public void successfulBuildWithAspect() throws Exception {
    addOptions("--experimental_merged_skyframe_analysis_execution");
    RecordingEventListener listener = new RecordingEventListener();
    eventListener = listener;

    // Run a simple test so that we can swipe its events
    write("a/a.sh", "exit 0").setExecutable(true);
    write(
        "a/BUILD",
        "load('//test_defs:foo_binary.bzl', 'foo_binary')",
        "foo_binary(name='a', srcs=['a.sh'])");
    write(
        "a/simpleaspect.bzl",
        """
        def _simple_aspect_impl(target, ctx):
            aspect_out = ctx.actions.declare_file(ctx.label.name + ".aspect")
            ctx.actions.write(
                output = aspect_out,
                content = "Hello from aspect",
            )
            return [OutputGroupInfo(**{
                "aspect-out": depset([aspect_out]),
            })]

        simple_aspect = aspect(implementation = _simple_aspect_impl)
        """);

    addOptions("--aspects=//a:simpleaspect.bzl%simple_aspect", "--output_groups=+aspect-out");
    assertThat(buildTarget("//a:a").getSuccess()).isTrue();
    assertThat(listener.targetSummaries).hasSize(1);
    assertThat(listener.targetCompletions).hasSize(1);
    assertThat(listener.aspectCompletions).hasSize(1);

    TargetSummaryEvent targetSummary = listener.targetSummaries.get(0);
    assertThat(targetSummary.isOverallBuildSuccess()).isTrue();
    assertThat(targetSummary.getOverallTestStatus()).isNull();

    TargetCompleteEvent targetComplete = listener.targetCompletions.get(0);
    assertThat(targetSummary.postedAfter()).containsExactly(targetComplete.getEventId());
    assertThat(listener.relevantEvents).containsAtLeast(targetComplete, targetSummary).inOrder();

    assertThat(listener.buildComplete.getResult().getSuccess()).isTrue();
    assertThat(listener.buildComplete.getResult().getSuccessfulTargets()).hasSize(1);
    assertThat(listener.buildComplete.getResult().getSuccessfulAspects()).hasSize(1);

    // To test TargetSummaryPublisher's handling of BuildCompleteEvent, replay just that event
    // with new publisher instance and mocked event bus.
    EventBus eventBus = mock(EventBus.class);
    ArgumentCaptor<TargetSummaryEvent> actual = ArgumentCaptor.forClass(TargetSummaryEvent.class);
    TargetSummaryPublisher replay =
        new TargetSummaryPublisher(
            eventBus, getCommandEnvironment()::withMergedAnalysisAndExecutionSourceOfTruth);
    replay.buildStarting(listener.buildStarting);
    replayAspectsIdentified(listener, replay);
    replayPopulateTargets(listener, replay);
    replay.buildCompleteEvent(listener.buildComplete);
    verify(eventBus).post(actual.capture());
    assertContentEquals(targetSummary, actual.getValue());
  }

  @Test
  public void successfulAliasBuildWithAspect() throws Exception {
    // Run a simple test so that we can swipe its events
    write("a/a.sh", "exit 0").setExecutable(true);
    write(
        "a/BUILD",
        """
        load("//test_defs:foo_test.bzl", "foo_test")

        foo_test(
            name = "a",
            srcs = ["a.sh"],
        )

        alias(
            name = "b",
            actual = ":a",
        )

        alias(
            name = "c",
            actual = ":b",
        )
        """);
    write(
        "a/simpleaspect.bzl",
        """
        def _simple_aspect_impl(target, ctx):
            aspect_out = ctx.actions.declare_file(ctx.label.name + ".aspect")
            ctx.actions.write(
                output = aspect_out,
                content = "Hello from aspect",
            )
            return [OutputGroupInfo(**{
                "aspect-out": depset([aspect_out]),
            })]

        simple_aspect = aspect(implementation = _simple_aspect_impl)
        """);

    // Successively build aliases and their targets
    ImmutableList<String> targets = ImmutableList.of("//a:c", "//a:b", "//a:a");
    for (int i = 1; i <= 3; ++i) {
      RecordingEventListener listener = new RecordingEventListener();
      eventListener = listener;

      String[] args =
          ImmutableList.<String>builder()
              .add("--aspects=//a:simpleaspect.bzl%simple_aspect", "--output_groups=+aspect-out")
              .add(TARGET_SUMMARY_FLAG)
              .addAll(targets.subList(0, i))
              .build()
              .toArray(new String[0]);
      // blaze test won't run tests unless they're directly requested (instead of through alias)
      assertThat(runTest(args)).isEqualTo(i == 3 ? 0 : 4); // 4 = "no tests to run"
      assertThat(listener.targetSummaries).hasSize(i);
      assertThat(listener.targetCompletions).hasSize(i);
      assertThat(listener.aspectCompletions).hasSize(i);
      assertThat(listener.testSummaries).hasSize(i / 3);

      assertThat(
              listener.targetSummaries.stream()
                  .map(summary -> summary.getEventId().getTargetSummary().getLabel()))
          .containsExactlyElementsIn(targets.subList(0, i));

      // Make sure we announce target summary for each label, alias or not (b/187508599)
      ImmutableList.Builder<String> completeLabels = ImmutableList.builder();
      for (TargetCompleteEvent completion : listener.targetCompletions) {
        String label = completion.getEventId().getTargetCompleted().getLabel();
        completeLabels.add(label);
        assertThat(completion.getChildrenEvents())
            .contains(
                BuildEventId.newBuilder()
                    .setTargetSummary(
                        TargetSummaryId.newBuilder()
                            .setLabel(label)
                            .setConfiguration(
                                completion.getEventId().getTargetCompleted().getConfiguration()))
                    .build());
      }
      assertThat(completeLabels.build()).containsExactlyElementsIn(targets.subList(0, i));

      // Only expect test status for actual test target (//a:a), not its aliases
      ImmutableList.Builder<String> summaryLabels = ImmutableList.builder();
      for (TargetSummaryEvent summary : listener.targetSummaries) {
        String label = summary.getEventId().getTargetSummary().getLabel();
        summaryLabels.add(label);
        assertWithMessage(label).that(summary.isOverallBuildSuccess()).isTrue();
        if (label.equals("//a:a")) {
          assertWithMessage(label)
              .that(summary.getOverallTestStatus())
              .isEqualTo(BlazeTestStatus.PASSED);
        } else {
          assertWithMessage(label).that(summary.getOverallTestStatus()).isNull();
        }
      }
      assertThat(summaryLabels.build()).containsExactlyElementsIn(targets.subList(0, i));
    }
  }

  @Test
  public void failingAspect_targetSummaryIndicatesFailure(
      @TestParameter boolean startsWithKeepGoing) throws Exception {
    // Run a simple test so that we can swipe its events
    write("a/a.sh", "exit 0").setExecutable(true);
    write(
        "a/BUILD",
        "load('//test_defs:foo_binary.bzl', 'foo_binary')",
        "foo_binary(name='a', srcs=['a.sh'])");
    write(
        "a/failingaspect.bzl",
        """
        def _failing_aspect_impl(target, ctx):
            aspect_out = ctx.actions.declare_file(ctx.label.name + ".aspect")
            ctx.actions.run_shell(
                outputs = [aspect_out],
                command = "false",
            )
            return [OutputGroupInfo(**{
                "aspect-out": depset([aspect_out]),
            })]

        failing_aspect = aspect(implementation = _failing_aspect_impl)
        """);

    addOptions("--aspects=//a:failingaspect.bzl%failing_aspect", "--output_groups=+aspect-out");
    // Run this test with and without -k
    boolean keepGoingValue = startsWithKeepGoing;
    for (int i = 0; i < 2; ++i, keepGoingValue = !keepGoingValue) {
      addOptions("--keep_going=" + keepGoingValue);
      RecordingEventListener listener = new RecordingEventListener();
      eventListener = listener;

      assertThrows(BuildFailedException.class, () -> buildTarget("//a:a"));

      assertThat(listener.targetSummaries).hasSize(1);
      assertThat(listener.aspectCompletions).hasSize(1);
      assertThat(listener.aspectCompletions.get(0).failed()).isTrue();

      TargetSummaryEvent targetSummary = listener.targetSummaries.get(0);
      assertThat(targetSummary.isOverallBuildSuccess()).isFalse();
      assertThat(targetSummary.getOverallTestStatus()).isNull();

      if (keepGoingValue) {
        // Make sure we got a successful target completion with -k
        assertThat(listener.targetCompletions).hasSize(1);
        assertThat(listener.targetCompletions.get(0).failed()).isFalse();
        assertThat(listener.buildComplete.getResult().getSuccessfulTargets()).hasSize(1);
      }

      assertThat(listener.buildComplete.getResult().getSuccess()).isFalse();
      assertThat(listener.buildComplete.getResult().getSuccessfulAspects()).isEmpty();

      // To test TargetSummaryPublisher's handling of BuildCompleteEvent, replay just that event
      // with new publisher instance and mocked event bus.
      EventBus eventBus = mock(EventBus.class);
      ArgumentCaptor<TargetSummaryEvent> actual = ArgumentCaptor.forClass(TargetSummaryEvent.class);
      TargetSummaryPublisher replay =
          new TargetSummaryPublisher(
              eventBus, getCommandEnvironment()::withMergedAnalysisAndExecutionSourceOfTruth);
      replay.buildStarting(listener.buildStarting);
      replayAspectsIdentified(listener, replay);
      replayPopulateTargets(listener, replay);
      replay.buildCompleteEvent(listener.buildComplete);
      verify(eventBus).post(actual.capture());
      assertContentEquals(targetSummary, actual.getValue());
    }
  }

  @Test
  public void failingAspect_aspectNotAnalyzed_targetSummaryIndicatesFailure(
      @TestParameter boolean useSkymeld) throws Exception {
    addOptions("--experimental_merged_skyframe_analysis_execution=" + useSkymeld);
    // Run a simple test so that we can swipe its events
    write("a/a.sh", "exit 0").setExecutable(true);
    write(
        "a/BUILD",
        "load('//test_defs:foo_binary.bzl', 'foo_binary')",
        "foo_binary(name='a', srcs=['a.sh'])");
    write(
        "a/failingaspect.bzl",
        """
        def _failing_aspect_impl(target, ctx):
            aspect_out = ctx.actions.declare_file(ctx.label.name + ".aspect")
            ctx.actions.run_shell(
                outputs = [aspect_out],
                command = "false",
            )
            return [OutputGroupInfo(**{
                "aspect-out": depset([aspect_out]),
            })]

        failing_aspect = aspect(implementation = _failing_aspect_impl)
        """);

    addOptions("--aspects=//a:failingaspect.bzl%failing_aspect", "--output_groups=+aspect-out");

    RecordingEventListener listener = new RecordingEventListener();
    eventListener = listener;

    assertThrows(BuildFailedException.class, () -> buildTarget("//a:a"));

    assertThat(listener.targetSummaries).hasSize(1);
    assertThat(listener.aspectCompletions).hasSize(1);
    assertThat(listener.aspectCompletions.get(0).failed()).isTrue();

    TargetSummaryEvent targetSummary = listener.targetSummaries.get(0);
    assertThat(targetSummary.isOverallBuildSuccess()).isFalse();
    assertThat(targetSummary.getOverallTestStatus()).isNull();

    assertThat(listener.buildComplete.getResult().getSuccess()).isFalse();
    assertThat(listener.buildComplete.getResult().getSuccessfulAspects()).isEmpty();

    // To test TargetSummaryPublisher's handling of BuildCompleteEvent, replay just that event
    // with new publisher instance and mocked event bus.
    EventBus eventBus = mock(EventBus.class);
    ArgumentCaptor<TargetSummaryEvent> actual = ArgumentCaptor.forClass(TargetSummaryEvent.class);
    TargetSummaryPublisher replay =
        new TargetSummaryPublisher(
            eventBus, getCommandEnvironment()::withMergedAnalysisAndExecutionSourceOfTruth);
    replay.buildStarting(listener.buildStarting);
    // Do *not* call replayAspectsIdentified. This simulates the --nokeep_going scenario where
    // the build stops early, before analyzing the aspects. We should still conclude the target
    // failed to build, because we know the user requested at least 1 aspect and we never even got
    // the ToplevelAspectsIdentifiedEvent.
    replayPopulateTargets(listener, replay);
    replay.buildCompleteEvent(listener.buildComplete);
    verify(eventBus).post(actual.capture());
    assertContentEquals(targetSummary, actual.getValue());
  }

  @Test
  public void aspectsFailAndSucceed_targetSummaryIndicatesFailure(
      @TestParameter boolean useSkymeld, @TestParameter boolean startsWithKeepGoing)
      throws Exception {
    addOptions("--experimental_merged_skyframe_analysis_execution=" + useSkymeld);
    // Run a simple test so that we can swipe its events
    write("a/a.sh", "exit 0").setExecutable(true);
    write(
        "a/BUILD",
        "load('//test_defs:foo_binary.bzl', 'foo_binary')",
        "foo_binary(name='a', srcs=['a.sh'])");
    write(
        "a/failingaspect.bzl",
        """
        def _failing_aspect_impl(target, ctx):
            aspect_out = ctx.actions.declare_file(ctx.label.name + ".aspect-bad")
            ctx.actions.run_shell(
                outputs = [aspect_out],
                command = "false",
            )
            return [OutputGroupInfo(**{
                "aspect-out": depset([aspect_out]),
            })]

        failing_aspect = aspect(implementation = _failing_aspect_impl)
        """);
    write(
        "a/simpleaspect.bzl",
        """
        def _simple_aspect_impl(target, ctx):
            aspect_out = ctx.actions.declare_file(ctx.label.name + ".aspect-good")
            ctx.actions.write(
                output = aspect_out,
                content = "Hello from aspect",
            )
            return [OutputGroupInfo(**{
                "aspect-out": depset([aspect_out]),
            })]

        simple_aspect = aspect(implementation = _simple_aspect_impl)
        """);
    write(
        "a/requiringaspect.bzl",
        """
        load(":simpleaspect.bzl", "simple_aspect")
        def _requiring_aspect_impl(target, ctx):
            aspect_out = ctx.actions.declare_file(ctx.label.name + ".requiring-aspect")
            ctx.actions.write(
                output = aspect_out,
                content = "Hello from requiring aspect",
            )
            return [OutputGroupInfo(**{
                "aspect-out": depset([aspect_out]),
            })]

        requiring_aspect = aspect(
            implementation=_requiring_aspect_impl,
            requires=[simple_aspect],
        )
        """);

    addOptions(
        "--aspects=//a:failingaspect.bzl%failing_aspect",
        "--aspects=//a:requiringaspect.bzl%requiring_aspect", "--output_groups=+aspect-out");

    // Run this test with and without -k
    boolean keepGoingValue = startsWithKeepGoing;
    for (int i = 0; i < 2; ++i, keepGoingValue = !keepGoingValue) {
      addOptions("--keep_going=" + keepGoingValue);
      RecordingEventListener listener = new RecordingEventListener();
      eventListener = listener;

      assertThrows(BuildFailedException.class, () -> buildTarget("//a:a"));

      assertThat(listener.targetSummaries).hasSize(1);
      assertThat(listener.aspectCompletions).isNotEmpty();
      // Sift through the AspectCompleteEvent to find the event for failing_aspect.
      AspectCompleteEvent failingAspectEvent =
          listener.aspectCompletions.stream()
              .filter(e -> e.getAspectName().contains("failing_aspect"))
              .findAny()
              .orElse(null);
      assertThat(failingAspectEvent).isNotNull();
      assertThat(failingAspectEvent.failed()).isTrue();

      TargetSummaryEvent targetSummary = listener.targetSummaries.get(0);
      assertThat(targetSummary.isOverallBuildSuccess()).isFalse();
      assertThat(targetSummary.getOverallTestStatus()).isNull();

      if (keepGoingValue) {
        // Make sure we got a successful target completion with -k. Aspects won't be reported as
        // they were already built in the first evaluation and don't need rebuilding. The toplevel
        // aspects key will have failed and also not report any aspects built.
        assertThat(listener.targetCompletions).hasSize(1);
        assertThat(listener.targetCompletions.get(0).failed()).isFalse();
        assertThat(listener.buildComplete.getResult().getSuccessfulTargets()).hasSize(1);
      }

      assertThat(listener.buildComplete.getResult().getSuccess()).isFalse();

      // To test TargetSummaryPublisher's handling of BuildCompleteEvent, replay just that event
      // with new publisher instance and mocked event bus.
      EventBus eventBus = mock(EventBus.class);
      ArgumentCaptor<TargetSummaryEvent> actual = ArgumentCaptor.forClass(TargetSummaryEvent.class);
      TargetSummaryPublisher replay =
          new TargetSummaryPublisher(
              eventBus, getCommandEnvironment()::withMergedAnalysisAndExecutionSourceOfTruth);
      replay.buildStarting(listener.buildStarting);
      replayAspectsIdentified(listener, replay);
      replayPopulateTargets(listener, replay);
      replay.buildCompleteEvent(listener.buildComplete);
      verify(eventBus).post(actual.capture());
      assertContentEquals(targetSummary, actual.getValue());
    }
  }

  @Test
  public void testWithFailingAspect_testsStillRunWithKeepGoing() throws Exception {
    RecordingEventListener listener = new RecordingEventListener();
    eventListener = listener;

    // Run a simple test so that we can swipe its events
    write("a/a.sh", "exit 0").setExecutable(true);
    write(
        "a/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name='a', srcs=['a.sh'])");
    write(
        "a/failingaspect.bzl",
        """
        def _failing_aspect_impl(target, ctx):
            aspect_out = ctx.actions.declare_file(ctx.label.name + ".aspect")
            ctx.actions.run_shell(
                outputs = [aspect_out],
                command = "false",
            )
            return [OutputGroupInfo(**{
                "aspect-out": depset([aspect_out]),
            })]

        failing_aspect = aspect(implementation = _failing_aspect_impl)
        """);

    assertThat(
            runTest(
                "--aspects=//a:failingaspect.bzl%failing_aspect",
                "--output_groups=+aspect-out", "-k", TARGET_SUMMARY_FLAG, "//a:a"))
        .isEqualTo(1);
    assertThat(listener.testSummaries).hasSize(1);
    assertThat(listener.targetSummaries).hasSize(1);
    assertThat(listener.aspectCompletions).hasSize(1);
    assertThat(listener.aspectCompletions.get(0).failed()).isTrue();

    TargetSummaryEvent targetSummary = listener.targetSummaries.get(0);
    assertThat(targetSummary.isOverallBuildSuccess()).isFalse();
    assertThat(targetSummary.getOverallTestStatus()).isNull(); // test status not recorded

    TestSummary testSummary = listener.testSummaries.get(0);
    assertThat(testSummary.getStatus()).isEqualTo(BlazeTestStatus.PASSED); // ... but test ran

    TargetCompleteEvent targetComplete = listener.targetCompletions.get(0);
    assertThat(targetSummary.postedAfter())
        .containsExactly(targetComplete.getEventId(), testSummary.getEventId());

    assertThat(listener.buildComplete.getResult().getSuccess()).isFalse();
    assertThat(listener.buildComplete.getResult().getSuccessfulTargets()).hasSize(1);
    assertThat(listener.buildComplete.getResult().getSuccessfulAspects()).isEmpty();

    // To test TargetSummaryPublisher's handling of BuildCompleteEvent, replay just that event
    // with new publisher instance and mocked event bus.
    EventBus eventBus = mock(EventBus.class);
    ArgumentCaptor<TargetSummaryEvent> actual = ArgumentCaptor.forClass(TargetSummaryEvent.class);
    TargetSummaryPublisher replay =
        new TargetSummaryPublisher(
            eventBus, getCommandEnvironment()::withMergedAnalysisAndExecutionSourceOfTruth);
    replay.buildStarting(listener.buildStarting);
    replayAspectsIdentified(listener, replay);
    replayPopulateTargets(listener, replay);
    replay.buildCompleteEvent(listener.buildComplete);
    // TestSummary event not needed since BuildCompleteEvent indicates build failure
    verify(eventBus).post(actual.capture());
    assertContentEquals(targetSummary, actual.getValue());
    replay.testSummaryEvent(testSummary);
    verify(eventBus).post(same(actual.getValue())); // still only match the first invocation *once*
  }

  @Test
  public void failingTargetWithSuccessfulAspect_targetSummaryIndicatesFailure(
      @TestParameter boolean startsWithKeepGoing) throws Exception {
    // Run a simple test so that we can swipe its events
    write("a/a.sh", "exit 0"); // should be executable
    write(
        "a/BUILD",
        "load('//test_defs:foo_binary.bzl', 'foo_binary')",
        "foo_binary(name='a', srcs=['a.sh'])");
    write(
        "a/simpleaspect.bzl",
        """
        def _simple_aspect_impl(target, ctx):
            aspect_out = ctx.actions.declare_file(ctx.label.name + ".aspect")
            ctx.actions.write(
                output = aspect_out,
                content = "Hello from aspect",
            )
            return [OutputGroupInfo(**{
                "aspect-out": depset([aspect_out]),
            })]

        simple_aspect = aspect(implementation = _simple_aspect_impl)
        """);

    addOptions("--aspects=//a:simpleaspect.bzl%simple_aspect", "--output_groups=+aspect-out");

    // Run this test with and without -k
    boolean keepGoingValue = startsWithKeepGoing;
    for (int i = 0; i < 2; ++i, keepGoingValue = !keepGoingValue) {
      addOptions("--keep_going=" + keepGoingValue);
      RecordingEventListener listener = new RecordingEventListener();
      eventListener = listener;

      assertThrows(BuildFailedException.class, () -> buildTarget("//a:a"));

      assertThat(listener.targetSummaries).hasSize(1);
      assertThat(listener.targetCompletions).hasSize(1);

      TargetCompleteEvent targetComplete = listener.targetCompletions.get(0);
      assertThat(targetComplete.failed()).isTrue();

      TargetSummaryEvent targetSummary = listener.targetSummaries.get(0);
      assertThat(targetSummary.isOverallBuildSuccess()).isFalse();
      assertThat(targetSummary.getOverallTestStatus()).isNull();
      assertThat(targetSummary.postedAfter()).containsExactly(targetComplete.getEventId());

      assertThat(listener.relevantEvents).containsAtLeast(targetComplete, targetSummary).inOrder();

      if (keepGoingValue) {
        // Make sure we got a successful aspect completion with -k
        assertThat(listener.aspectCompletions).hasSize(1);
        assertThat(listener.aspectCompletions.get(0).failed()).isFalse();
      }

      // To test TargetSummaryPublisher's handling of BuildCompleteEvent, replay just that event
      // with new publisher instance and mocked event bus.
      EventBus eventBus = mock(EventBus.class);
      ArgumentCaptor<TargetSummaryEvent> actual = ArgumentCaptor.forClass(TargetSummaryEvent.class);
      TargetSummaryPublisher replay =
          new TargetSummaryPublisher(
              eventBus, getCommandEnvironment()::withMergedAnalysisAndExecutionSourceOfTruth);
      replay.buildStarting(listener.buildStarting);
      replayAspectsIdentified(listener, replay);
      replayPopulateTargets(listener, replay);
      replay.buildCompleteEvent(listener.buildComplete);
      // TestSummary event not needed since BuildCompleteEvent indicates build failure
      verify(eventBus).post(actual.capture());
      assertContentEquals(targetSummary, actual.getValue());
    }
  }

  @Test
  public void failingTargetWithSuccessfulAspectWithRequiredAspect_targetSummaryIndicatesFailure(
      @TestParameter boolean useSkymeld, @TestParameter boolean startsWithKeepGoing)
      throws Exception {
    addOptions("--experimental_merged_skyframe_analysis_execution=" + useSkymeld);
    write("a/a.sh", "exit 0"); // a.sh is not executable, causing build failure
    write(
        "a/BUILD",
        "load('//test_defs:foo_binary.bzl', 'foo_binary')",
        "foo_binary(name='a', srcs=['a.sh'])");
    write(
        "a/simpleaspect.bzl",
        """
        def _simple_aspect_impl(target, ctx):
            aspect_out = ctx.actions.declare_file(ctx.label.name + ".aspect")
            ctx.actions.write(
                output = aspect_out,
                content = "Hello from aspect",
            )
            return [OutputGroupInfo(**{
                "aspect-out": depset([aspect_out]),
            })]

        simple_aspect = aspect(implementation = _simple_aspect_impl)
        """);
    write(
        "a/requiringaspect.bzl",
        """
        load(":simpleaspect.bzl", "simple_aspect")
        def _requiring_aspect_impl(target, ctx):
            aspect_out = ctx.actions.declare_file(ctx.label.name + ".requiring-aspect")
            ctx.actions.write(
                output = aspect_out,
                content = "Hello from requiring aspect",
            )
            return [OutputGroupInfo(**{
                "aspect-out": depset([aspect_out]),
            })]

        requiring_aspect = aspect(
            implementation=_requiring_aspect_impl,
            requires=[simple_aspect],
        )
        """);

    addOptions("--aspects=//a:requiringaspect.bzl%requiring_aspect", "--output_groups=+aspect-out");

    // Run this test with and without -k
    boolean keepGoingValue = startsWithKeepGoing;
    for (int i = 0; i < 2; ++i, keepGoingValue = !keepGoingValue) {
      addOptions("--keep_going=" + keepGoingValue);
      RecordingEventListener listener = new RecordingEventListener();
      eventListener = listener;

      assertThrows(BuildFailedException.class, () -> buildTarget("//a:a"));

      assertThat(listener.targetSummaries).hasSize(1);
      assertThat(listener.targetCompletions).hasSize(1);

      TargetCompleteEvent targetComplete = listener.targetCompletions.get(0);
      assertThat(targetComplete.failed()).isTrue();

      TargetSummaryEvent targetSummary = listener.targetSummaries.get(0);
      assertThat(targetSummary.isOverallBuildSuccess()).isFalse();
      assertThat(targetSummary.getOverallTestStatus()).isNull();
      assertThat(targetSummary.postedAfter()).containsExactly(targetComplete.getEventId());

      assertThat(listener.relevantEvents).containsAtLeast(targetComplete, targetSummary).inOrder();

      if (keepGoingValue) {
        // Make sure we got a successful aspect completion with -k
        assertThat(listener.aspectCompletions).hasSize(2);
        assertThat(listener.aspectCompletions.get(0).failed()).isFalse();
        assertThat(listener.aspectCompletions.get(1).failed()).isFalse();
      }

      // To test TargetSummaryPublisher's handling of BuildCompleteEvent, replay just that event
      // with new publisher instance and mocked event bus.
      EventBus eventBus = mock(EventBus.class);
      ArgumentCaptor<TargetSummaryEvent> actual = ArgumentCaptor.forClass(TargetSummaryEvent.class);
      TargetSummaryPublisher replay =
          new TargetSummaryPublisher(
              eventBus, getCommandEnvironment()::withMergedAnalysisAndExecutionSourceOfTruth);
      replay.buildStarting(listener.buildStarting);
      replayAspectsIdentified(listener, replay);
      replayPopulateTargets(listener, replay);
      replay.buildCompleteEvent(listener.buildComplete);
      // TestSummary event not needed since BuildCompleteEvent indicates build failure
      verify(eventBus).post(actual.capture());
      assertContentEquals(targetSummary, actual.getValue());
    }
  }

  @Test
  public void buildMultipleTargets_skymeld_multipleTopLevelTargetPendingExecutionRecorded()
      throws Exception {
    addOptions("--experimental_merged_skyframe_analysis_execution");
    RecordingEventListener listener = new RecordingEventListener();
    eventListener = listener;

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
    Set<TopLevelTargetPendingExecutionEvent> topLevelTargetPendingExecutions =
        listener.topLevelTargetPendingExecutions;
    assertThat(
            topLevelTargetPendingExecutions.stream()
                .map(e -> e.configuredTarget().getOriginalLabel().getCanonicalForm()))
        .containsExactly("//foo:foo", "//foo:bar");
  }

  @Test
  public void
      buildMultipleTargetsWithExecutionFailure_skymeld_multipleTopLevelTargetPendingExecutionRecorded()
          throws Exception {
    addOptions("--keep_going");
    addOptions("--experimental_merged_skyframe_analysis_execution");
    RecordingEventListener listener = new RecordingEventListener();
    eventListener = listener;

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

    Set<TopLevelTargetPendingExecutionEvent> topLevelTargetPendingExecutions =
        listener.topLevelTargetPendingExecutions;
    assertThat(
            topLevelTargetPendingExecutions.stream()
                .map(e -> e.configuredTarget().getOriginalLabel().getCanonicalForm()))
        .containsExactly("//foo:foo", "//foo:execution_failure");
  }

  @Test
  public void
      buildMultipleTargetsWithAnalysisFailure_skymeld_singleTopLevelTargetPendingExecutionRecorded()
          throws Exception {
    addOptions("--keep_going");
    addOptions("--experimental_merged_skyframe_analysis_execution");
    RecordingEventListener listener = new RecordingEventListener();
    eventListener = listener;

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

    assertThrows(
        BuildFailedException.class, () -> buildTarget("//foo:foo", "//foo:analysis_failure"));

    Set<TopLevelTargetPendingExecutionEvent> topLevelTargetPendingExecutions =
        listener.topLevelTargetPendingExecutions;
    assertThat(
            topLevelTargetPendingExecutions.stream()
                .map(e -> e.configuredTarget().getOriginalLabel().getCanonicalForm()))
        .containsExactly("//foo:foo");
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

  private void replayAspectsIdentified(
      RecordingEventListener listener, TargetSummaryPublisher replay) {
    for (ToplevelAspectsIdentifiedEvent event : listener.toplevelAspectsIdentifiedEvents) {
      replay.toplevelAspectsIdentified(event);
    }
  }

  private void replayPopulateTargets(
      RecordingEventListener listener, TargetSummaryPublisher replay) {
    if (getCommandEnvironment().withMergedAnalysisAndExecutionSourceOfTruth()) {
      for (TopLevelTargetPendingExecutionEvent event : listener.topLevelTargetPendingExecutions) {
        replay.populateTarget(event);
      }
    } else {
      replay.populateTargets(listener.testFilteringComplete);
    }
  }

  private static void assertContentEquals(TargetSummaryEvent expected, TargetSummaryEvent actual) {
    assertThat(actual.isOverallBuildSuccess()).isEqualTo(expected.isOverallBuildSuccess());
    assertThat(actual.getOverallTestStatus()).isEqualTo(expected.getOverallTestStatus());
    assertThat(actual.getEventId()).isEqualTo(expected.getEventId());
    assertThat(actual.postedAfter()).isEqualTo(expected.postedAfter());
  }
}
