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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.AllowConcurrentEvents;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.analysis.test.TestResult;
import com.google.devtools.build.lib.analysis.test.TestRunnerAction;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.TestFilteringCompleteEvent;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.runtime.commands.BuildCommand;
import com.google.devtools.build.lib.runtime.commands.TestCommand;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.TestCompletionValue.TestCompletionKey;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.TestAnalyzedEvent;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.TopLevelTargetAnalyzedEvent;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import com.google.devtools.build.lib.view.test.TestStatus.TestResultData;
import com.google.devtools.build.skyframe.NotifyingHelper;
import com.google.devtools.build.skyframe.NotifyingHelper.EventType;
import com.google.devtools.common.options.Options;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link AggregatingTestListener}. */
@RunWith(JUnit4.class)
public final class AggregatingTestListenerTest extends BuildIntegrationTestCase {

  private RecordingOutErr recordingOutErr;
  private BlazeCommandDispatcher dispatcher;
  private Object eventListener;
  private EventBus eventBus;

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder()
        .addBlazeModule(
            new BlazeModule() {
              @Override
              public void beforeCommand(CommandEnvironment env) {
                AggregatingTestListenerTest.this.eventBus = env.getEventBus();
                if (eventListener != null) {
                  eventBus.register(eventListener);
                }
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

  private static final class RecordingEventListener {
    private TestResult testResult;
    private TestFilteringCompleteEvent testFilteringComplete;
    private BuildCompleteEvent buildComplete;
    private final Set<TestAnalyzedEvent> testAnalysisCompleteEvents = new HashSet<>();
    private final List<TestSummary> testSummaries = new ArrayList<>();

    @Subscribe
    public void testResult(TestResult event) {
      this.testResult = event;
    }

    @Subscribe
    public void testFilteringComplete(TestFilteringCompleteEvent event) {
      testFilteringComplete = event;
    }

    @Subscribe
    public void testAnalysisComplete(TestAnalyzedEvent event) {
      testAnalysisCompleteEvents.add(event);
    }

    @Subscribe
    public void buildComplete(BuildCompleteEvent event) {
      buildComplete = event;
    }

    @Subscribe
    public void testSummary(TestSummary event) {
      testSummaries.add(event);
    }
  }

  @Test
  public void testInconsistentTestEventsAndBuildResult() throws Exception {
    RecordingEventListener listener = new RecordingEventListener();
    eventListener = listener;

    // Run a simple test so that we can swipe its events
    write("a/a.sh", "exit 0").setExecutable(true);
    write(
        "a/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name='a', srcs=['a.sh'])");
    assertThat(runTest("//a:a")).isEqualTo(0);
    assertThat(listener.testSummaries).hasSize(1);

    // Create an intact AggregatingTestListener and replay some of the events:
    AggregatingTestListener testListener =
        new AggregatingTestListener(
            Options.getDefaults(TestSummaryOptions.class),
            Options.getDefaults(ExecutionOptions.class),
            eventBus);
    // 1. Tell it about the test
    if (listener.testFilteringComplete != null) {
      testListener.populateTests(listener.testFilteringComplete);
    } else {
      // Skymeld only.
      testListener.populateTest(Iterables.getOnlyElement(listener.testAnalysisCompleteEvents));
    }
    // 2. Send the test result event
    testListener.testEvent(listener.testResult);
    // 3. Send a build complete event without any successful targets.
    testListener.buildComplete(
        listener.buildComplete.getResult().getActualTargets(),
        ImmutableList.of(),
        ImmutableList.of());
    // 4. Check that the listener received only one more TestSummary, even though the test case
    // had a TestSummary event but was not in successfulTargets
    assertThat(listener.testSummaries).hasSize(2);
  }

  /** Regression test for b/193902847. */
  @Test
  public void aliasAndActualWithValidationAspect() throws Exception {
    write(
        "foo/BUILD",
        """
        load("//test_defs:foo_test.bzl", "foo_test")

        alias(
            name = "alias",
            actual = ":actual",
        )

        foo_test(
            name = "actual",
            srcs = ["test.sh"],
        )
        """);
    write("foo/test.sh", "exit 0").setExecutable(true);
    assertThat(runTest("--experimental_use_validation_aspect", "//foo:alias", "//foo:actual"))
        .isEqualTo(0);
  }

  /** Regression test for b/241013089. */
  @Test
  public void aliasAndActualInDifferentConfigurations() throws Exception {
    write(
        "foo/defs.bzl",
        """
        def _my_transition_impl(settings, attr):
            return {"//command_line_option:compiler": "abc"}

        my_transition = transition(
            implementation = _my_transition_impl,
            inputs = [],
            outputs = ["//command_line_option:compiler"],
        )

        def _my_test_impl(ctx):
            f = ctx.actions.declare_file(ctx.attr.name + ".sh")
            ctx.actions.write(f, "#!/bin/sh\\nexit 0\\n", is_executable = True)
            return DefaultInfo(executable = f)

        my_test = rule(
            implementation = _my_test_impl,
            test = True,
            cfg = my_transition,
        )
        """);
    write(
        "foo/BUILD",
        """
        load(":defs.bzl", "my_test")

        alias(
            name = "alias",
            actual = ":actual",
        )

        my_test(name = "actual")
        """);

    assertThat(runTest("//foo:alias", "//foo:actual")).isEqualTo(0);
  }

  /** Regression test for b/419325593. */
  @Test
  public void aliasWinsRaceAgainstActualInSkymeldMode() throws Exception {
    write(
        "foo/BUILD",
        """
        load("//test_defs:foo_test.bzl", "foo_test")

        foo_test(name = "actual", srcs = ["test.sh"])
        alias(name = "alias", actual = ":actual")
        """);
    write("foo/test.sh", "exit 0").setExecutable(true);

    CountDownLatch testComplete = new CountDownLatch(1);
    eventListener =
        new Object() {
          @Subscribe
          @AllowConcurrentEvents
          void topLevelTargetAnalyzed(TopLevelTargetAnalyzedEvent event)
              throws InterruptedException {
            if (!AliasProvider.isAlias(event.configuredTarget())) {
              testComplete.await(); // Actual waits for alias to complete the test.
            }
          }
        };
    getSkyframeExecutor()
        .getEvaluator()
        .injectGraphTransformerForTesting(
            NotifyingHelper.makeNotifyingTransformer(
                (key, type, order, context) -> {
                  if (key instanceof TestCompletionKey && type == EventType.SET_VALUE) {
                    testComplete.countDown();
                  }
                }));

    assertThat(runTest("//foo:alias", "//foo:actual")).isEqualTo(0);
  }

  @Test
  public void incompleteRunsNotProcessedTwice() throws Exception {
    TestSummary summary =
        testDifferentialAnalyzeAndReport(
            "//foo:two_shards",
            _ -> null,
            "load('//test_defs:foo_test.bzl', 'foo_test')",
            "foo_test(name = 'two_shards', srcs = ['test.sh'], shard_count = 2)");
    assertThat(summary.getStatus()).isEqualTo(BlazeTestStatus.INCOMPLETE);
    assertThat(summary.totalRuns()).isEqualTo(1); // > 1 indicates duplicate processing.
  }

  @Test
  public void validationFailureReportsAsBuildFailure() throws Exception {
    TestSummary summary =
        testDifferentialAnalyzeAndReport(
            "//foo:my_test",
            _ -> ImmutableSet.of(),
            "load('//test_defs:foo_test.bzl', 'foo_test')",
            "foo_test(name = 'my_test', srcs = ['test.sh'])");
    assertThat(summary.getStatus()).isEqualTo(BlazeTestStatus.FAILED_TO_BUILD);
  }

  @Test
  public void validationSuccessReportsOriginalTestStatus() throws Exception {
    TestSummary summary =
        testDifferentialAnalyzeAndReport(
            "//foo:my_test",
            target -> ImmutableSet.of(ConfiguredTargetKey.fromConfiguredTarget(target)),
            "load('//test_defs:foo_test.bzl', 'foo_test')",
            "foo_test(name = 'my_test', srcs = ['test.sh'])");
    assertThat(summary.getStatus()).isEqualTo(BlazeTestStatus.PASSED);
    assertThat(summary.totalRuns()).isEqualTo(1); // > 1 indicates duplicate processing.
  }

  @Test
  public void skymeldMultiTestTargets_oneTestAnalysisCompleteEventPerTarget() throws Exception {
    RecordingEventListener listener = new RecordingEventListener();
    eventListener = listener;
    writeMyTestBzl();
    write(
        "foo/BUILD",
        """
        load("//foo:my_test.bzl", "my_test")

        my_test(
            name = "bar",
            srcs = ["bar.in"],
        )

        my_test(
            name = "foo",
            srcs = ["foo.in"],
        )
        """);
    write("foo/foo.in");
    write("foo/bar.in");

    assertThat(
            runTest(
                "--experimental_merged_skyframe_analysis_execution",
                "//foo:foo",
                "//foo:bar"))
        .isEqualTo(/*exitCode*/ 0);

    assertThat(listener.testAnalysisCompleteEvents).hasSize(2);
    assertThat(
            listener.testAnalysisCompleteEvents.stream()
                .map(event -> event.configuredTarget().getLabel().getCanonicalForm()))
        .containsExactly("//foo:foo", "//foo:bar");
  }

  @Test
  public void skymeldMultiTestTargetsAnalysisError_keepGoing_oneTestAnalysisCompleteEvent()
      throws Exception {
    RecordingEventListener listener = new RecordingEventListener();
    eventListener = listener;
    writeMyTestBzl();
    write(
        "private/BUILD",
        "load('//test_defs:foo_library.bzl', 'foo_library')",
        "foo_library(name = 'private', visibility = ['//visibility:private'])");
    write(
        "foo/BUILD",
        """
        load("//foo:my_test.bzl", "my_test")

        my_test(
            name = "analysis_error",
            srcs = ["foo.in"],
            deps = ["//private"],
        )

        my_test(
            name = "foo",
            srcs = ["foo.in"],
        )
        """);
    write("foo/foo.in");

    assertThat(
            runTest(
                "--experimental_merged_skyframe_analysis_execution",
                "--keep_going",
                "//foo:foo",
                "//foo:analysis_error"))
        .isEqualTo(/*exitCode*/ 1);

    assertThat(listener.testAnalysisCompleteEvents).hasSize(1);
    assertThat(
            Iterables.getOnlyElement(listener.testAnalysisCompleteEvents)
                .configuredTarget()
                .getLabel()
                .getCanonicalForm())
        .isEqualTo("//foo:foo");
  }

  private void writeMyTestBzl() throws Exception {
    write(
        "foo/my_test.bzl",
        """
        def _impl(ctx):
            executable = ctx.actions.declare_file(ctx.attr.name + ".exec")
            executable_content = "exit %d" % (ctx.attr.exit_code)
            ctx.actions.write(executable, executable_content, is_executable = True)
            return DefaultInfo(executable = executable)

        my_test = rule(
            implementation = _impl,
            test = True,
            attrs = {
                "srcs": attr.label_list(allow_files = True),
                "deps": attr.label_list(),
                "exit_code": attr.int(),
            },
        )
        """);
  }

  private TestSummary testDifferentialAnalyzeAndReport(
      String testName,
      Function<RuleConfiguredTarget, ImmutableSet<ConfiguredTargetKey>> fakeValidatedTargets,
      String... buildFile)
      throws Exception {
    // Run analysis just so we can get the ConfiguredTarget for the test.
    write("foo/BUILD", buildFile);
    addOptions("--nobuild");
    buildTarget(testName);
    RuleConfiguredTarget testTarget = (RuleConfiguredTarget) getExistingConfiguredTarget(testName);

    EventBus testBus = new EventBus();
    AggregatingTestListener listener =
        new AggregatingTestListener(
            Options.getDefaults(TestSummaryOptions.class),
            Options.getDefaults(ExecutionOptions.class),
            testBus);
    testBus.register(listener);
    testBus.post(
        new TestFilteringCompleteEvent(
            ImmutableList.of(testTarget),
            ImmutableList.of(testTarget),
            /*targetsToSkip=*/ ImmutableList.of(),
            ImmutableMap.of(testTarget.getConfigurationKey(), getTargetConfiguration())));
    // Simulate one passed test shard, resulting in incomplete test result for sharded test
    testBus.post(
        new TestResult(
            Iterables.filter(testTarget.getActions(), TestRunnerAction.class).iterator().next(),
            TestResultData.newBuilder().setStatus(BlazeTestStatus.PASSED).addTestTimes(999).build(),
            ImmutableMultimap.of(),
            /* cached= */ false,
            /* systemFailure= */ null));

    AtomicReference<TestSummary> summary = new AtomicReference<>();
    listener.differentialAnalyzeAndReport(
        ImmutableList.of(testTarget),
        /*skippedTargets=*/ ImmutableList.of(),
        fakeValidatedTargets.apply(testTarget),
        (summaries, n) ->
            assertThat(summary.getAndSet(Iterables.getOnlyElement(summaries))).isNull());

    assertThat(summary.get()).isNotNull();
    return summary.get();
  }
}
