// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.metrics.criticalpath;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;
import static org.mockito.Mockito.mock;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionChangePrunedEvent;
import com.google.devtools.build.lib.actions.ActionCompletionEvent;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.ActionStartedEvent;
import com.google.devtools.build.lib.actions.AggregatedSpawnMetrics;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.CachedActionEvent;
import com.google.devtools.build.lib.actions.DiscoveredInputsEvent;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SimpleSpawn;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnExecutedEvent;
import com.google.devtools.build.lib.actions.SpawnExecutedEvent.ChangePhase;
import com.google.devtools.build.lib.actions.SpawnMetrics;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.cache.OutputMetadataStore;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil.MockAction;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil.NullAction;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.clock.BlazeClock.NanosToMillisSinceEpochConverter;
import com.google.devtools.build.lib.exec.util.FakeActionInputFileCache;
import com.google.devtools.build.lib.skyframe.ActionTemplateExpansionValue;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.skyframe.rewinding.ActionRewoundEvent;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.testutil.TestFileOutErr;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.WalkableGraph;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.time.Duration;
import java.time.Instant;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Unit tests for {@link CriticalPathComputer}. */
@RunWith(TestParameterInjector.class)
public class CriticalPathComputerTest extends FoundationTestCase {

  private ManualClock clock;
  private CriticalPathComputer computer;
  private ArtifactRoot artifactRoot;
  private ArtifactRoot derivedArtifactRoot;

  @Before
  public final void initializeRoots() {
    Path workspaceRoot = scratch.resolve("/workspace");
    derivedArtifactRoot = ArtifactRoot.asDerivedRoot(workspaceRoot, RootType.OUTPUT, "test");
    artifactRoot = ArtifactRoot.asSourceRoot(Root.fromPath(workspaceRoot));
  }

  @Before
  public void createComputer() {
    clock = new ManualClock();
    computer = new CriticalPathComputer(new ActionKeyContext(), /* graph= */ null);
  }

  private static void assertActionMatches(Action action, CriticalPathComponent component) {
    if (!actionMatches(action, component)) {
      fail("Action " + action + " did not match one in " + component);
    }
  }

  private static boolean actionMatches(Action action, CriticalPathComponent component) {
    return component.getAction() == action;
  }

  @Test
  public void testNoSpawnMetrics() {
    CriticalPathComponent cp = new CriticalPathComponent(1, new NullAction(), 0);
    assertThat(cp.getSpawnMetrics()).isEqualTo(AggregatedSpawnMetrics.EMPTY);
    assertThat(cp.getLongestPhaseSpawnRunnerName()).isNull();
  }

  @Test
  public void testMultipleSpawnMetrics() {
    CriticalPathComponent cp = new CriticalPathComponent(1, new NullAction(), 0);
    cp.addSpawnResult(
        SpawnMetrics.Builder.forRemoteExec().setTotalTimeInMs(10 * 1000).build(),
        "first",
        "",
        false);
    cp.addSpawnResult(
        SpawnMetrics.Builder.forRemoteExec().setTotalTimeInMs(30 * 1000).build(),
        "second",
        "",
        false);
    cp.addSpawnResult(
        SpawnMetrics.Builder.forRemoteExec().setTotalTimeInMs(20 * 1000).build(),
        "third",
        "",
        false);
    cp.finishActionExecution(0, 40, "test");
    // The current implementation keeps the maximum spawn metrics because we do not differentiate
    // between sequential or parallel spawn invocations within a single Bazel action. So while it is
    // still 'incorrect', it is more fair than keeping the latest invocation data.
    assertThat(cp.getSpawnMetrics().getRemoteMetrics().totalTimeInMs()).isEqualTo(30 * 1000);
    assertThat(cp.getLongestPhaseSpawnRunnerName()).isEqualTo("second");
  }

  /**
   * Test that 'other' time is correctly computed as any time not measured by the rest of the stats.
   */
  @Test
  public void testSpawnMetricsOtherTimeComputed() {
    SpawnMetrics spawnMetrics =
        SpawnMetrics.Builder.forRemoteExec()
            .setTotalTimeInMs(100 * 1000)
            .setParseTimeInMs(1 * 1000)
            .setNetworkTimeInMs(2 * 1000)
            .setFetchTimeInMs(3 * 1000)
            .setQueueTimeInMs(4 * 1000)
            .setSetupTimeInMs(5 * 1000)
            .setUploadTimeInMs(6 * 1000)
            .setExecutionWallTimeInMs(7 * 1000)
            .setRetryTimeInMs(ImmutableMap.of(1, 8 * 1000))
            .setProcessOutputsTimeInMs(9 * 1000)
            .build();
    assertThat(spawnMetrics.otherTimeInMs()).isEqualTo(55 * 1000);
  }

  @Test
  public void testCriticalPathOneAction() throws Exception {
    simulateActionExec(new NullAction(), 2 * 1000, 1 * 1000, true);
    checkCriticalPath(
        Duration.ofSeconds(2),
        Duration.ofSeconds(1),
        Duration.ofSeconds(1),
        "2.00",
        "50.00",
        "50.00");
    checkTopComponentsTimes(computer, 2000L);
  }

  @Test
  public void testCriticalPathQueueTimeWithoutRetries() throws Exception {
    SpawnResult.Builder spawnResult =
        createSpawnResult()
            .setSpawnMetrics(
                SpawnMetrics.Builder.forRemoteExec()
                    .setTotalTimeInMs(4 * 1000)
                    .setExecutionWallTimeInMs(1 * 1000)
                    .setQueueTimeInMs(1 * 1000)
                    .build());
    simulateActionExec(new NullAction(), 8 * 1000, spawnResult.build());
    AggregatedCriticalPath stats =
        checkCriticalPath(
            Duration.ofSeconds(8),
            Duration.ofSeconds(4),
            Duration.ofSeconds(1),
            "8.00",
            "50.00",
            "12.50");
    assertThat(stats.getSpawnMetrics().getRemoteMetrics().queueTimeInMs()).isEqualTo(1 * 1000);
  }

  /**
   * Check that the timing stats are printed correctly, that the printed values correctly match
   * their label.
   */
  @Test
  public void testCriticalPathToString() throws Exception {
    MockAction actionA = new MockAction(ImmutableSet.of(), ImmutableSet.of(artifact("a.out")));
    SpawnResult.Builder spawnResult =
        createSpawnResult()
            .setSpawnMetrics(
                SpawnMetrics.Builder.forRemoteExec()
                    .setParseTimeInMs(5 * 1000)
                    .setNetworkTimeInMs(6 * 1000)
                    .setFetchTimeInMs(7 * 1000)
                    .setQueueTimeInMs(8 * 1000)
                    .setSetupTimeInMs(9 * 1000)
                    .setUploadTimeInMs(10 * 1000)
                    .setProcessOutputsTimeInMs(4 * 1000)
                    .setExecutionWallTimeInMs(40 * 1000)
                    .setTotalTimeInMs(100 * 1000)
                    .build());
    simulateActionExec(actionA, spawnResult);
    AggregatedCriticalPath stats = computer.aggregate();
    assertThat(stats).isNotNull();

    String toString = stats.toString();
    assertThat(toString).contains("parse: 5.00%");
    assertThat(toString).contains("network: 6.00%");
    assertThat(toString).contains("fetch: 7.00%");
    assertThat(toString).contains("queue: 8.00%");
    assertThat(toString).contains("setup: 9.00%");
    assertThat(toString).contains("upload: 10.00%");
    assertThat(toString).contains("processOutputs: 4.00%");
    assertThat(toString).contains("process: 40.00%");
    assertThat(toString).contains("other: 11.00%");
  }

  /**
   * Check that we only print certain critical parts of the timing stats when they are below a
   * certain threshold, to avoid spamming the user.
   */
  @Test
  public void testCriticalPathToStringSummary() throws Exception {
    MockAction actionA = new MockAction(ImmutableSet.of(), ImmutableSet.of(artifact("a.out")));
    MockAction actionB =
        new MockAction(
            Collections.singleton(artifact("a.out")), ImmutableSet.of(artifact("b.out")));

    SpawnResult.Builder spawnResult =
        createSpawnResult()
            .setSpawnMetrics(
                SpawnMetrics.Builder.forRemoteExec()
                    .setNetworkTimeInMs(10 * 1000)
                    .setParseTimeInMs(10 * 1000)
                    .setFetchTimeInMs(10 * 1000)
                    .setQueueTimeInMs(10 * 1000)
                    .setSetupTimeInMs(10 * 1000)
                    .setProcessOutputsTimeInMs(10 * 1000)
                    .setExecutionWallTimeInMs(20 * 1000)
                    .setUploadTimeInMs(10 * 1000)
                    .setTotalTimeInMs(100 * 1000)
                    .build());
    simulateActionExec(actionA, spawnResult);
    AggregatedCriticalPath stats = computer.aggregate();
    assertThat(stats).isNotNull();
    String summary = stats.toString();
    assertThat(summary).contains("network: 10.00%");
    assertThat(summary).contains("parse: 10.00%");
    assertThat(summary).contains("queue: 10.00%");
    assertThat(summary).contains("upload: 10.00%");
    assertThat(summary).contains("setup: 10.00%");
    assertThat(summary).contains("processOutputs: 10.00%");
    assertThat(summary).contains("process: 20.00%");
    assertThat(summary).contains("fetch: 10.00%");
    assertThat(summary).contains("other: 10.00%");

    // Add another action execution so that now the critical path is A + B, and the 10 second stats
    // each are bumped below 10%, bringing them below the "summary" threshold.
    spawnResult = createSpawnResult(10 * 1000);
    simulateActionExec(actionB, spawnResult);
    stats = computer.aggregate();
    assertThat(stats).isNotNull();
    summary = stats.toStringSummary();
    assertThat(summary).doesNotContain("network:");
    assertThat(summary).doesNotContain("parse:");
    assertThat(summary).contains("queue:");
    assertThat(summary).doesNotContain("upload:");
    assertThat(summary).contains("setup:");
    assertThat(summary).contains("process:");
    assertThat(summary).doesNotContain("fetch:");
    assertThat(summary).doesNotContain("processOutputs:");
    assertThat(summary).doesNotContain("other:");
  }

  // The real value of durations are not important for the test, using the same unit for all
  // declarations makes it easier to verify the aggregated values are correct.
  @SuppressWarnings("CanonicalDuration")
  @Test
  public void testAggregateMetrics() throws Exception {
    MockAction actionA = new MockAction(ImmutableList.of(), ImmutableSet.of(artifact("a.out")));
    MockAction actionB =
        new MockAction(ImmutableList.of(artifact("a.out")), ImmutableSet.of(artifact("b.out")));
    MockAction actionC =
        new MockAction(ImmutableList.of(artifact("b.out")), ImmutableSet.of(artifact("c.out")));
    MockAction actionD =
        new MockAction(ImmutableList.of(artifact("c.out")), ImmutableSet.of(artifact("d.out")));

    simulateActionExec(
        actionA,
        createSpawnResult()
            .setSpawnMetrics(
                SpawnMetrics.Builder.forRemoteExec()
                    .setNetworkTimeInMs(1 * 1000)
                    .setParseTimeInMs(2 * 1000)
                    .setFetchTimeInMs(3 * 1000)
                    .setQueueTimeInMs(4 * 1000)
                    .setSetupTimeInMs(5 * 1000)
                    .setProcessOutputsTimeInMs(6 * 1000)
                    .setExecutionWallTimeInMs(7 * 1000)
                    .setUploadTimeInMs(8 * 1000)
                    .setTotalTimeInMs(100 * 1000)
                    .build()));

    simulateActionExec(
        actionB,
        createSpawnResult()
            .setSpawnMetrics(
                SpawnMetrics.Builder.forRemoteExec()
                    .setNetworkTimeInMs(20 * 1000)
                    .setParseTimeInMs(30 * 1000)
                    .setFetchTimeInMs(40 * 1000)
                    .setQueueTimeInMs(50 * 1000)
                    .setSetupTimeInMs(60 * 1000)
                    .setProcessOutputsTimeInMs(70 * 1000)
                    .setExecutionWallTimeInMs(80 * 1000)
                    .setUploadTimeInMs(90 * 1000)
                    .setTotalTimeInMs(1000 * 1000)
                    .build()));

    simulateActionExec(
        actionC,
        createSpawnResult()
            .setSpawnMetrics(
                SpawnMetrics.Builder.forWorkerExec()
                    .setNetworkTimeInMs(10 * 1000)
                    .setParseTimeInMs(20 * 1000)
                    .setFetchTimeInMs(30 * 1000)
                    .setQueueTimeInMs(40 * 1000)
                    .setSetupTimeInMs(50 * 1000)
                    .setProcessOutputsTimeInMs(60 * 1000)
                    .setExecutionWallTimeInMs(70 * 1000)
                    .setUploadTimeInMs(80 * 1000)
                    .setTotalTimeInMs(1000 * 1000)
                    .build()));

    simulateActionExec(
        actionD,
        createSpawnResult()
            .setSpawnMetrics(
                SpawnMetrics.Builder.forWorkerExec()
                    .setNetworkTimeInMs(200 * 1000)
                    .setParseTimeInMs(300 * 1000)
                    .setFetchTimeInMs(400 * 1000)
                    .setQueueTimeInMs(500 * 1000)
                    .setSetupTimeInMs(600 * 1000)
                    .setProcessOutputsTimeInMs(700 * 1000)
                    .setExecutionWallTimeInMs(800 * 1000)
                    .setUploadTimeInMs(900 * 1000)
                    .setTotalTimeInMs(10000 * 1000)
                    .build()));

    AggregatedSpawnMetrics aggregated = computer.aggregate().getSpawnMetrics();
    SpawnMetrics remoteMetrics = aggregated.getMetrics(SpawnMetrics.ExecKind.REMOTE);
    assertThat(remoteMetrics.networkTimeInMs()).isEqualTo(21 * 1000);
    assertThat(remoteMetrics.parseTimeInMs()).isEqualTo(32 * 1000);
    assertThat(remoteMetrics.fetchTimeInMs()).isEqualTo(43 * 1000);
    assertThat(remoteMetrics.queueTimeInMs()).isEqualTo(54 * 1000);
    assertThat(remoteMetrics.setupTimeInMs()).isEqualTo(65 * 1000);
    assertThat(remoteMetrics.processOutputsTimeInMs()).isEqualTo(76 * 1000);
    assertThat(remoteMetrics.executionWallTimeInMs()).isEqualTo(87 * 1000);
    assertThat(remoteMetrics.uploadTimeInMs()).isEqualTo(98 * 1000);
    assertThat(remoteMetrics.totalTimeInMs()).isEqualTo(1100 * 1000);

    SpawnMetrics workerMetrics = aggregated.getMetrics(SpawnMetrics.ExecKind.WORKER);
    assertThat(workerMetrics.networkTimeInMs()).isEqualTo(210 * 1000);
    assertThat(workerMetrics.parseTimeInMs()).isEqualTo(320 * 1000);
    assertThat(workerMetrics.fetchTimeInMs()).isEqualTo(430 * 1000);
    assertThat(workerMetrics.queueTimeInMs()).isEqualTo(540 * 1000);
    assertThat(workerMetrics.setupTimeInMs()).isEqualTo(650 * 1000);
    assertThat(workerMetrics.processOutputsTimeInMs()).isEqualTo(760 * 1000);
    assertThat(workerMetrics.executionWallTimeInMs()).isEqualTo(870 * 1000);
    assertThat(workerMetrics.uploadTimeInMs()).isEqualTo(980 * 1000);
    assertThat(workerMetrics.totalTimeInMs()).isEqualTo(11000 * 1000);
  }

  @Test
  public void testEmptyCriticalPath() {
    AggregatedCriticalPath empty = computer.aggregate();
    assertThat(empty.components()).isEmpty();
    assertThat(empty.getAggregatedElapsedTime().toMillis()).isEqualTo(0);
    checkTopComponentsTimes(computer);
  }

  /** Tests that we only record the top slowest components and that we drop the rest. */
  @Test
  public void testTopComponentsOverflow() throws Exception {
    for (int i = 0; i <= 1000; i++) {
      MockAction action = new MockAction(ImmutableSet.of(), ImmutableSet.of(artifact(i + ".out")));
      simulateActionExec(action, i);
    }
    long[] topTimes = new long[CriticalPathComputer.SLOWEST_COMPONENTS_SIZE];
    for (int i = 0; i < CriticalPathComputer.SLOWEST_COMPONENTS_SIZE; i++) {
      topTimes[i] = 1000L - i;
    }
    checkTopComponentsTimes(computer, topTimes);
  }

  @Test
  public void testLargestMemoryComponentsOverflow() throws Exception {
    for (int i = 0; i < 1000; i++) {
      MockAction action = new MockAction(ImmutableSet.of(), ImmutableSet.of(artifact(i + ".out")));
      // the largest actions are in the middle
      simulateActionExec(
          action,
          createSpawnResult()
              .setSpawnMetrics(
                  SpawnMetrics.Builder.forRemoteExec()
                      .setMemoryEstimateBytes(500 < i && i < 600 ? i : 0)
                      .setExecutionWallTimeInMs(1 * 1000)
                      .setTotalTimeInMs(i * 1000)
                      .build()));
    }

    List<CriticalPathComponent> result = computer.getLargestMemoryComponents();

    assertThat(result).hasSize(20);
    assertThat(result.get(0).getSpawnMetrics().getRemoteMetrics().memoryEstimate()).isEqualTo(599);
  }

  @Test
  public void testLargestInputSizeComponentsOverflow() throws Exception {
    for (int i = 0; i < 1000; i++) {
      MockAction action = new MockAction(ImmutableSet.of(), ImmutableSet.of(artifact(i + ".out")));
      simulateActionExec(
          action,
          createSpawnResult()
              .setSpawnMetrics(
                  SpawnMetrics.Builder.forRemoteExec()
                      .setInputBytes(500 < i && i < 600 ? i : 0)
                      .setExecutionWallTimeInMs(1 * 1000)
                      .setTotalTimeInMs(i * 1000)
                      .build()));
    }

    List<CriticalPathComponent> result = computer.getLargestInputSizeComponents();

    assertThat(result).hasSize(20);
    assertThat(result.get(0).getSpawnMetrics().getRemoteMetrics().inputBytes()).isEqualTo(599);
  }

  @Test
  public void testLargestInputCountComponentsOverflow() throws Exception {
    for (int i = 0; i < 1000; i++) {
      MockAction action = new MockAction(ImmutableSet.of(), ImmutableSet.of(artifact(i + ".out")));
      simulateActionExec(
          action,
          createSpawnResult()
              .setSpawnMetrics(
                  SpawnMetrics.Builder.forRemoteExec()
                      .setInputFiles(500 < i && i < 600 ? i : 0)
                      .setExecutionWallTimeInMs(1 * 1000)
                      .setTotalTimeInMs(i * 1000)
                      .build()));
    }

    List<CriticalPathComponent> result = computer.getLargestInputCountComponents();

    assertThat(result).hasSize(20);
    assertThat(result.get(0).getSpawnMetrics().getRemoteMetrics().inputFiles()).isEqualTo(599);
  }

  @Test
  public void testActionCached() throws Exception {
    MockAction cachedAction =
        new MockAction(ImmutableSet.of(), ImmutableSet.of(artifact("cached.out")));

    MockAction topLevelAction =
        new MockAction(
            Collections.singleton(artifact("cached.out")), ImmutableSet.of(artifact("top.out")));

    computer.actionCached(
        new CachedActionEvent(
            cachedAction, new FakeActionInputFileCache(), clock.nanoTime(), clock.nanoTime()));
    simulateActionExec(topLevelAction, 1000);

    AggregatedCriticalPath aggregated = computer.aggregate();

    assertThat(aggregated.components()).hasSize(2);
    assertActionMatches(topLevelAction, aggregated.components().get(0));
    assertActionMatches(cachedAction, aggregated.components().get(1));
    assertThat(aggregated.components().get(0).getElapsedTime()).isEqualTo(Duration.ofSeconds(1));
    assertThat(aggregated.components().get(1).getElapsedTime()).isEqualTo(Duration.ZERO);

    checkTopComponentsTimes(computer, 1000, 0L);
  }

  /** Test that wall time is not computed using nanotime. */
  @Test
  public void testWallTime() throws Exception {
    simulateActionExec(new NullAction(), 2000);
    checkCriticalPath(2000, "2.00");
    checkTopComponentsTimes(computer, 2000L);
    NanosToMillisSinceEpochConverter converter =
        BlazeClock.createNanosToMillisSinceEpochConverter(clock);
    assertThat(computer.getMaxCriticalPath().getStartTimeMillisSinceEpoch(converter)).isEqualTo(0L);
  }

  /**
   * When running shared actions concurrently we might end up receiving multiple events, one per
   * shared action. In that case we record a single component and we update the time of the maximum
   * elapsed time.
   */
  @Test
  public void testConcurrentSharedActions() throws Exception {
    MockAction shared1 = new MockAction(ImmutableSet.of(), ImmutableSet.of(artifact("shared.out")));
    MockAction shared2 = new MockAction(ImmutableSet.of(), ImmutableSet.of(artifact("shared.out")));

    MockAction action1 =
        new MockAction(
            Collections.singleton(artifact("shared.out")),
            ImmutableSet.of(artifact("action1.out")));

    MockAction action2 =
        new MockAction(
            Collections.singleton(artifact("shared.out")),
            ImmutableSet.of(artifact("action2.out")));

    long shared1Start = clock.nanoTime();
    computer.actionStarted(new ActionStartedEvent(shared1, shared1Start));
    clock.advanceMillis(1000);
    long shared2Start = clock.nanoTime();
    // We concurrently execute shared2 before shared1 could finish. But we record it as a cache hit.
    computer.actionCached(
        new CachedActionEvent(
            shared2, new FakeActionInputFileCache(), clock.nanoTime(), shared2Start));
    clock.advanceMillis(1);
    // Action2 depends on shared2, so it can start executing without waiting to shared1. This will
    // prevent us from identifying the critical path in some circumstance, but we are OK with that.
    simulateActionExec(action2, 11);

    computer.actionComplete(
        new ActionCompletionEvent(
            shared1Start,
            clock.nanoTime(),
            shared1,
            new FakeActionInputFileCache(),
            mock(OutputMetadataStore.class),
            mock(ActionLookupData.class)));
    simulateActionExec(action1, 10);
    AggregatedCriticalPath criticalPath = computer.aggregate();

    // Yes, this is not correct but expected. While action2.time > action1.time, because
    // action2 executed before shared1 finishes it incorrectly gets the time set by shared2.
    assertActionMatches(action1, criticalPath.components().get(0));
    // We expect that the component used for any critical path is shared1, as it is the first that
    // was started.
    assertActionMatches(shared1, criticalPath.components().get(1));
    assertThat(criticalPath.components().get(1).getElapsedTime())
        .isEqualTo(Duration.ofMillis(1012));

    List<CriticalPathComponent> slowest = computer.getSlowestComponents();
    assertThat(slowest).hasSize(3);
    for (CriticalPathComponent cpath : slowest) {
      if (actionMatches(shared1, cpath)) {
        assertThat(cpath.getElapsedTime()).isEqualTo(Duration.ofMillis(1012));
      }
      // While shared2 was a cache hit, because it was executed concurrently with shared1 we
      // keep one component with the maximum time.
      if (actionMatches(shared2, cpath)) {
        assertThat(cpath.getElapsedTime()).isEqualTo(Duration.ofMillis(1012));
      }
      if (actionMatches(action1, cpath)) {
        assertThat(cpath.getElapsedTime()).isEqualTo(Duration.ofMillis(10));
      }
      if (actionMatches(action2, cpath)) {
        assertThat(cpath.getElapsedTime()).isEqualTo(Duration.ofMillis(11));
        assertThat(cpath.getChild().getElapsedTime()).isEqualTo(Duration.ofMillis(1012));
      }
    }
  }

  @Test
  public void testTotalAggregateRunTimeWithGaps() throws Exception {
    MockAction action1 =
        new MockAction(ImmutableSet.of(), ImmutableSet.of(artifact("action1.out")));
    MockAction action2 =
        new MockAction(
            ImmutableSet.of(artifact("action1.out")), ImmutableSet.of(artifact("action2.out")));
    MockAction action3 =
        new MockAction(
            ImmutableSet.of(artifact("action2.out")), ImmutableSet.of(artifact("action3.out")));

    long action1Start = clock.nanoTime();
    computer.actionStarted(new ActionStartedEvent(action1, action1Start));
    clock.advanceMillis(1000);
    computer.actionComplete(
        new ActionCompletionEvent(
            action1Start,
            clock.nanoTime(),
            action1,
            new FakeActionInputFileCache(),
            mock(OutputMetadataStore.class),
            mock(ActionLookupData.class)));

    clock.advanceMillis(2000);
    long action2Start = clock.nanoTime();
    computer.actionStarted(new ActionStartedEvent(action2, action2Start));
    clock.advanceMillis(3000);
    computer.actionComplete(
        new ActionCompletionEvent(
            action2Start,
            clock.nanoTime(),
            action2,
            new FakeActionInputFileCache(),
            mock(OutputMetadataStore.class),
            mock(ActionLookupData.class)));

    clock.advanceMillis(2000);
    long action3Start = clock.nanoTime();
    computer.actionStarted(new ActionStartedEvent(action3, action3Start));
    clock.advanceMillis(4000);
    computer.actionComplete(
        new ActionCompletionEvent(
            action3Start,
            clock.nanoTime(),
            action3,
            new FakeActionInputFileCache(),
            mock(OutputMetadataStore.class),
            mock(ActionLookupData.class)));

    // The runtime of the critical path ignoring gaps is 8 seconds.
    assertThat(computer.getMaxCriticalPath().getAggregatedElapsedTime())
        .isEqualTo(Duration.ofSeconds(8));
    assertThat(Duration.ofNanos(clock.nanoTime() - action1Start)).isEqualTo(Duration.ofSeconds(12));
  }

  @Test
  public void testTotalAggregateRunTimeWithOverlappingTimes() throws Exception {
    MockAction action1 =
        new MockAction(ImmutableSet.of(), ImmutableSet.of(artifact("action1.out")));
    MockAction action2 =
        new MockAction(
            ImmutableSet.of(artifact("action1.out")), ImmutableSet.of(artifact("action2.out")));

    long action1Start = clock.nanoTime();
    computer.actionStarted(new ActionStartedEvent(action1, action1Start));
    clock.advanceMillis(1000);
    long action2Start = clock.nanoTime();
    computer.actionStarted(new ActionStartedEvent(action2, action2Start));
    clock.advanceMillis(2000);
    computer.actionComplete(
        new ActionCompletionEvent(
            action1Start,
            clock.nanoTime(),
            action1,
            new FakeActionInputFileCache(),
            mock(OutputMetadataStore.class),
            mock(ActionLookupData.class)));
    clock.advanceMillis(2000);
    computer.actionComplete(
        new ActionCompletionEvent(
            action2Start,
            clock.nanoTime(),
            action2,
            new FakeActionInputFileCache(),
            mock(OutputMetadataStore.class),
            mock(ActionLookupData.class)));

    // The total run time of all actions in the critical path is 5 seconds.
    assertThat(computer.getMaxCriticalPath().getAggregatedElapsedTime())
        .isEqualTo(Duration.ofSeconds(5));
    AggregatedCriticalPath criticalPath = computer.aggregate();
    assertThat(criticalPath.components()).hasSize(2);
    // Action 2  has a run time of 4 seconds
    assertThat(criticalPath.components().get(0).getElapsedTime()).isEqualTo(Duration.ofSeconds(4));
    // Action 1 has a run time of 3 seconds
    assertThat(criticalPath.components().get(1).getElapsedTime()).isEqualTo(Duration.ofSeconds(3));
  }

  @Test
  public void testTotalAggregateRunTimeWithParallelRuns() throws Exception {
    MockAction action1 =
        new MockAction(ImmutableSet.of(), ImmutableSet.of(artifact("action1.out")));
    MockAction action2 =
        new MockAction(
            ImmutableSet.of(artifact("action1.out")), ImmutableSet.of(artifact("action2.out")));

    long action2Start = clock.nanoTime();
    computer.actionStarted(new ActionStartedEvent(action2, action2Start));
    clock.advanceMillis(1000);
    long action1Start = clock.nanoTime();
    computer.actionStarted(new ActionStartedEvent(action1, action1Start));
    clock.advanceMillis(2000);
    computer.actionComplete(
        new ActionCompletionEvent(
            action1Start,
            clock.nanoTime(),
            action1,
            new FakeActionInputFileCache(),
            mock(OutputMetadataStore.class),
            mock(ActionLookupData.class)));
    clock.advanceMillis(2000);
    computer.actionComplete(
        new ActionCompletionEvent(
            action2Start,
            clock.nanoTime(),
            action2,
            new FakeActionInputFileCache(),
            mock(OutputMetadataStore.class),
            mock(ActionLookupData.class)));

    // The total run time of all actions in the critical path is 5 seconds.
    assertThat(computer.getMaxCriticalPath().getAggregatedElapsedTime())
        .isEqualTo(Duration.ofSeconds(5));
    AggregatedCriticalPath criticalPath = computer.aggregate();
    assertThat(criticalPath.components()).hasSize(2);
    // Action 2 has a run time of 5 seconds
    assertThat(criticalPath.components().get(0).getElapsedTime()).isEqualTo(Duration.ofSeconds(5));
    // Action 1 has a run time of 2 seconds
    assertThat(criticalPath.components().get(1).getElapsedTime()).isEqualTo(Duration.ofSeconds(2));
  }

  @Test
  public void testLongestTotalTime() throws Exception {
    MockAction action1 =
        new MockAction(ImmutableSet.of(), ImmutableSet.of(artifact("action1.out")));
    MockAction action2 =
        new MockAction(ImmutableSet.of(), ImmutableSet.of(artifact("action2.out")));
    MockAction action3 =
        new MockAction(
            ImmutableList.of(artifact("action1.out"), artifact("action2.out")),
            ImmutableSet.of(artifact("action3.out")));

    // Action 1 - 0s - 3s
    long action1Start = clock.nanoTime();
    computer.actionStarted(new ActionStartedEvent(action1, action1Start));
    clock.advanceMillis(3000);
    computer.actionComplete(
        new ActionCompletionEvent(
            action1Start,
            clock.nanoTime(),
            action1,
            new FakeActionInputFileCache(),
            mock(OutputMetadataStore.class),
            mock(ActionLookupData.class)));
    // Action 2 - 3s - 7s
    long action2Start = clock.nanoTime();
    computer.actionStarted(new ActionStartedEvent(action2, action2Start));
    clock.advanceMillis(1000);
    // Action 3 - 4s - 7s
    long action3Start = clock.nanoTime();
    computer.actionStarted(new ActionStartedEvent(action3, action3Start));
    clock.advanceMillis(3000);
    computer.actionComplete(
        new ActionCompletionEvent(
            action2Start,
            clock.nanoTime(),
            action2,
            new FakeActionInputFileCache(),
            mock(OutputMetadataStore.class),
            mock(ActionLookupData.class)));
    computer.actionComplete(
        new ActionCompletionEvent(
            action3Start,
            clock.nanoTime(),
            action3,
            new FakeActionInputFileCache(),
            mock(OutputMetadataStore.class),
            mock(ActionLookupData.class)));

    // The total run time should be 6s (Action 1 + Action 3) since Action 2 overlaps with
    // action 3, they will not be aggregated.
    assertThat(computer.getMaxCriticalPath().getAggregatedElapsedTime())
        .isEqualTo(Duration.ofSeconds(6));
    AggregatedCriticalPath criticalPath = computer.aggregate();
    assertThat(criticalPath.components()).hasSize(2);
    // Action 3 has a run time of 3 seconds
    assertThat(criticalPath.components().get(0).getElapsedTime()).isEqualTo(Duration.ofSeconds(3));
    // Action 1 has a run time of 3 seconds
    assertThat(criticalPath.components().get(1).getElapsedTime()).isEqualTo(Duration.ofSeconds(3));
  }

  @Test
  public void rewoundActionMayStartTwice() throws Exception {
    // This test demonstrates that a rewound action can cause two ActionStartedEvents to be emitted,
    // one paired with an ActionRewoundEvent and the other with an ActionCompletedEvent, and the
    // CriticalPathComputer handles it.
    MockAction producer =
        new MockAction(ImmutableSet.of(), ImmutableSet.of(artifact("shared.out")));
    MockAction consumer =
        new MockAction(
            Collections.singleton(artifact("shared.out")),
            ImmutableSet.of(artifact("consumer.out")));

    simulateActionExec(producer, 10);
    long consumerFirstStart = clock.nanoTime();
    computer.actionStarted(new ActionStartedEvent(consumer, consumerFirstStart));
    clock.advanceMillis(5);
    computer.actionRewound(new ActionRewoundEvent(consumerFirstStart, clock.nanoTime(), consumer));

    // In a real rewinding case, "producer" would be re-evaluated, and the events for that
    // re-evaluation would be suppressed. This statement simulates that process by advancing the
    // clock without any associated events.
    clock.advanceMillis(10);
    simulateActionExec(consumer, 20);

    AggregatedCriticalPath criticalPath = computer.aggregate();

    assertActionMatches(consumer, criticalPath.components().get(0));
    assertActionMatches(producer, criticalPath.components().get(1));

    assertThat(criticalPath.components().get(0).getElapsedTime()).isEqualTo(Duration.ofMillis(20));
    assertThat(criticalPath.components().get(1).getElapsedTime()).isEqualTo(Duration.ofMillis(10));

    List<CriticalPathComponent> slowest = computer.getSlowestComponents();
    assertThat(slowest).hasSize(2);
    for (CriticalPathComponent cpath : slowest) {
      if (actionMatches(producer, cpath)) {
        assertThat(cpath.getElapsedTime()).isEqualTo(Duration.ofMillis(10));
      }
      if (actionMatches(consumer, cpath)) {
        assertThat(cpath.getElapsedTime()).isEqualTo(Duration.ofMillis(20));
      }
    }
  }

  /**
   * Check that the slowest components list does not duplicate entries when an action has multiple
   * outputs.
   */
  @Test
  public void testSlowestComponentsNoDuplicates() throws Exception {
    MockAction action =
        new MockAction(ImmutableList.of(), ImmutableSet.of(artifact("a.out"), artifact("b.out")));
    simulateActionExec(action, 123);

    List<CriticalPathComponent> slowest = computer.getSlowestComponents();
    assertThat(slowest).hasSize(1);
  }

  @Test
  public void testSequentialActionExec() throws Exception {
    simulateSequentialAndParallelActionExec(
        new MockAction(ImmutableList.of(), ImmutableSet.of(artifact("a.out"))),
        ImmutableList.of(
            ImmutableList.of(2 * 1000), ImmutableList.of(3 * 1000), ImmutableList.of(4 * 1000)));
    SpawnMetrics metrics = computer.getMaxCriticalPath().getSpawnMetrics().getRemoteMetrics();
    assertThat(metrics.totalTimeInMs()).isEqualTo(9 * 1000);
  }

  @Test
  public void testMaximumSequentialAndParallelActionMetrics() throws Exception {
    MockAction action = new MockAction(ImmutableList.of(), ImmutableSet.of(artifact("a.out")));

    ImmutableList<ImmutableList<Integer>> seqAndParallelSeries =
        ImmutableList.of(
            ImmutableList.of(5 * 1000), // +5
            ImmutableList.of(1 * 1000, 3 * 1000), // +3
            ImmutableList.of(7 * 1000) // +7
            );

    simulateSequentialAndParallelActionExec(action, seqAndParallelSeries);
    SpawnMetrics metrics = computer.getMaxCriticalPath().getSpawnMetrics().getRemoteMetrics();
    assertThat(metrics.totalTimeInMs()).isEqualTo(15 * 1000);
  }

  @Test
  public void testInputDiscoveryAndAction() throws Exception {
    Action action = new MockAction(ImmutableList.of(), ImmutableSet.of(artifact("a.out")));
    simulateActionExec(action, 2 * 1000, 2 * 1000, true, 5 * 1000);
    SpawnMetrics metrics = computer.getMaxCriticalPath().getSpawnMetrics().getRemoteMetrics();
    assertThat(metrics.parseTimeInMs()).isEqualTo(5 * 1000);
    assertThat(metrics.executionWallTimeInMs()).isEqualTo(2 * 1000);
    assertThat(metrics.totalTimeInMs()).isEqualTo(7 * 1000);
  }

  @Test
  public void testInputDiscoveryBeforeActionStarted() throws Exception {
    Artifact artifact = artifact("a.out");
    Action action = new MockAction(ImmutableList.of(), ImmutableSet.of(artifact));
    computer.discoverInputs(
        new DiscoveredInputsEvent(
            SpawnMetrics.Builder.forRemoteExec()
                .setParseTimeInMs(5 * 1000)
                .setTotalTimeInMs(5 * 1000)
                .build(),
            action,
            /* startTimeNanos= */ 0));

    computer.actionComplete(
        new ActionCompletionEvent(
            0,
            clock.nanoTime(),
            action,
            new FakeActionInputFileCache(),
            mock(OutputMetadataStore.class),
            mock(ActionLookupData.class)));
    SpawnMetrics metrics = computer.getMaxCriticalPath().getSpawnMetrics().getRemoteMetrics();
    assertThat(metrics.parseTimeInMs()).isEqualTo(5 * 1000);
    assertThat(metrics.totalTimeInMs()).isEqualTo(5 * 1000);
  }

  @Test
  public void testTryAddComponentShouldAddNonSharedActions() throws Exception {
    Artifact artifact = artifact("a.out");
    MockAction sharedAction = new MockAction(ImmutableList.of(), ImmutableSet.of(artifact));
    MockAction nonSharedAction =
        new MockAction(ImmutableList.of(), ImmutableSet.of(artifact), /* isShareable= */ false);
    computer.actionStarted(new ActionStartedEvent(sharedAction, clock.nanoTime()));
    IllegalStateException exception =
        assertThrows(
            IllegalStateException.class,
            () ->
                computer.actionStarted(new ActionStartedEvent(nonSharedAction, clock.nanoTime())));
    assertThat(exception)
        .hasMessageThat()
        .contains("Duplicate output artifact found for unsharable actions.");
  }

  @Test
  public void toleratesCriticalPathInconsistency() throws Exception {
    Artifact depArtifact = derivedArtifact("test/a.out");
    Artifact parentArtifact = derivedArtifact("test/b.out");
    MockAction depAction = new MockAction(ImmutableList.of(), ImmutableSet.of(depArtifact));
    MockAction parentAction =
        new MockAction(ImmutableList.of(depArtifact), ImmutableSet.of(parentArtifact));

    computer.actionStarted(new ActionStartedEvent(depAction, clock.nanoTime()));
    clock.advanceMillis(1000);
    computer.actionStarted(new ActionStartedEvent(parentAction, clock.nanoTime()));

    // Complete the parent action while the dep action is still running and check that the resulting
    // critical path ignores the still-running dep.
    computer.actionComplete(
        new ActionCompletionEvent(
            clock.nanoTime(),
            clock.nanoTime(),
            parentAction,
            new FakeActionInputFileCache(),
            mock(OutputMetadataStore.class),
            mock(ActionLookupData.class)));
    assertThat(Iterables.getOnlyElement(computer.aggregate().components()).getAction())
        .isEqualTo(parentAction);
  }

  @Test
  public void testChangePruning() throws Exception {
    MockAction action1 =
        new MockAction(ImmutableSet.of(), ImmutableSet.of(derivedArtifact("test/action1.out")));
    MockAction action2 =
        new MockAction(
            ImmutableSet.of(derivedArtifact("test/action1.out")),
            ImmutableSet.of(derivedArtifact("test/action2.out")));
    MockAction action3 =
        new MockAction(
            ImmutableList.of(derivedArtifact("test/action2.out")),
            ImmutableSet.of(derivedArtifact("test/action3.out")));
    MockAction action4 =
        new MockAction(
            ImmutableList.of(
                derivedArtifact("test/action1.out"), derivedArtifact("test/action3.out")),
            ImmutableSet.of(derivedArtifact("test/action4.out")));

    computer =
        new CriticalPathComputer(
            new ActionKeyContext(),
            new WalkableGraph() {
              @Nullable
              @Override
              public SkyValue getValue(SkyKey key) throws InterruptedException {
                if (key instanceof ActionLookupKey) {
                  return new ActionLookupValue() {
                    @Override
                    public ImmutableList<ActionAnalysisMetadata> getActions() {
                      return ImmutableList.of(action3);
                    }
                  };
                }
                throw new UnsupportedOperationException();
              }

              @Override
              public Map<SkyKey, SkyValue> getSuccessfulValues(Iterable<? extends SkyKey> keys)
                  throws InterruptedException {
                throw new UnsupportedOperationException();
              }

              @Override
              public Map<SkyKey, Exception> getMissingAndExceptions(Iterable<SkyKey> keys)
                  throws InterruptedException {
                throw new UnsupportedOperationException();
              }

              @Nullable
              @Override
              public Exception getException(SkyKey key) throws InterruptedException {
                throw new UnsupportedOperationException();
              }

              @Override
              public boolean isCycle(SkyKey key) throws InterruptedException {
                throw new UnsupportedOperationException();
              }

              @Override
              public Map<SkyKey, Iterable<SkyKey>> getDirectDeps(Iterable<SkyKey> keys)
                  throws InterruptedException {
                throw new UnsupportedOperationException();
              }

              @Override
              public Iterable<SkyKey> getDirectDeps(SkyKey key) throws InterruptedException {
                throw new UnsupportedOperationException();
              }

              @Override
              public Map<SkyKey, Iterable<SkyKey>> getReverseDeps(Iterable<? extends SkyKey> keys)
                  throws InterruptedException {
                throw new UnsupportedOperationException();
              }

              @Override
              public Map<SkyKey, Pair<SkyValue, Iterable<SkyKey>>> getValueAndRdeps(
                  Iterable<SkyKey> keys) throws InterruptedException {
                throw new UnsupportedOperationException();
              }
            });

    // Action 1 - 0s - 1s
    long action1Start = clock.nanoTime();
    computer.actionStarted(new ActionStartedEvent(action1, action1Start));
    clock.advanceMillis(1000);
    computer.actionComplete(
        new ActionCompletionEvent(
            action1Start,
            clock.nanoTime(),
            action1,
            new FakeActionInputFileCache(),
            mock(OutputMetadataStore.class),
            mock(ActionLookupData.class)));
    // Action 2 - 1s - 3s
    long action2Start = clock.nanoTime();
    computer.actionStarted(new ActionStartedEvent(action2, action2Start));
    clock.advanceMillis(2000);
    computer.actionComplete(
        new ActionCompletionEvent(
            action2Start,
            clock.nanoTime(),
            action2,
            new FakeActionInputFileCache(),
            mock(OutputMetadataStore.class),
            mock(ActionLookupData.class)));
    // Action 3 - 3s - 3s, change pruned, no events
    computer.actionChangePruned(
        new ActionChangePrunedEvent(ActionsTestUtil.NULL_ACTION_LOOKUP_DATA, clock.nanoTime()));
    // Action 4 - 3s - 6s
    long action4Start = clock.nanoTime();
    computer.actionStarted(new ActionStartedEvent(action4, action4Start));
    clock.advanceMillis(3000);
    computer.actionComplete(
        new ActionCompletionEvent(
            action4Start,
            clock.nanoTime(),
            action4,
            new FakeActionInputFileCache(),
            mock(OutputMetadataStore.class),
            mock(ActionLookupData.class)));

    // The total run time should be 6s (Action 1 + Action 2 + Action 4) since Action 3 is
    // change-pruned.
    assertThat(computer.getMaxCriticalPath().getAggregatedElapsedTime())
        .isEqualTo(Duration.ofSeconds(6));
    AggregatedCriticalPath criticalPath = computer.aggregate();
    assertThat(criticalPath.components()).hasSize(4);
    // Action 4 has a run time of 3 seconds
    assertThat(criticalPath.components().get(0).prettyPrintAction()).contains("action4.out");
    assertThat(criticalPath.components().get(0).getElapsedTime()).isEqualTo(Duration.ofSeconds(3));
    // Action 3 has a run time of 0 seconds
    assertThat(criticalPath.components().get(1).prettyPrintAction()).contains("action3.out");
    assertThat(criticalPath.components().get(1).getElapsedTime()).isEqualTo(Duration.ZERO);
    // Action 2 has a run time of 2 seconds
    assertThat(criticalPath.components().get(2).prettyPrintAction()).contains("action2.out");
    assertThat(criticalPath.components().get(2).getElapsedTime()).isEqualTo(Duration.ofSeconds(2));
    // Action 1 has a run time of 1 seconds
    assertThat(criticalPath.components().get(3).prettyPrintAction()).contains("action1.out");
    assertThat(criticalPath.components().get(3).getElapsedTime()).isEqualTo(Duration.ofSeconds(1));
  }

  private void simulateActionExec(Action action, int totalTime) throws InterruptedException {
    long nanoTimeStart = clock.nanoTime();
    computer.actionStarted(new ActionStartedEvent(action, nanoTimeStart));
    clock.advanceMillis(totalTime);
    computer.actionComplete(
        new ActionCompletionEvent(
            nanoTimeStart,
            clock.nanoTime(),
            action,
            new FakeActionInputFileCache(),
            mock(OutputMetadataStore.class),
            mock(ActionLookupData.class)));
  }

  private void simulateActionExec(
      Action action,
      int totalTimeInMs,
      int processTimeInMs,
      boolean completeAction,
      int discoverInputsDurationInMs)
      throws InterruptedException {
    computer.discoverInputs(
        new DiscoveredInputsEvent(
            SpawnMetrics.Builder.forRemoteExec()
                .setParseTimeInMs(discoverInputsDurationInMs)
                .setTotalTimeInMs(discoverInputsDurationInMs)
                .build(),
            action,
            /* startTimeNanos= */ 0));
    simulateActionExec(action, totalTimeInMs, processTimeInMs, completeAction);
  }

  private void simulateActionExec(
      Action action, int totalTimeInMs, int processTimeInMs, boolean completeAction)
      throws InterruptedException {
    SpawnResult spawnResult =
        createSpawnResult()
            .setSpawnMetrics(
                SpawnMetrics.Builder.forRemoteExec()
                    .setTotalTimeInMs(processTimeInMs)
                    .setExecutionWallTimeInMs(processTimeInMs)
                    .build())
            .build();
    simulateActionExec(action, totalTimeInMs, spawnResult, completeAction);
  }

  private void simulateActionExec(Action action, SpawnResult.Builder spawnResult)
      throws InterruptedException {
    simulateActionExec(
        action, spawnResult.build().getMetrics().totalTimeInMs(), spawnResult.build());
  }

  private void simulateActionExec(Action action, int totalTimeInMs, SpawnResult spawnResult)
      throws InterruptedException {
    simulateActionExec(action, totalTimeInMs, spawnResult, true);
  }

  private void simulateActionExec(
      Action action, int totalTimeInMs, SpawnResult spawnResult, boolean completeAction)
      throws InterruptedException {
    long startTime = clock.nanoTime();
    computer.actionStarted(new ActionStartedEvent(action, startTime));
    clock.advanceMillis(totalTimeInMs);
    Spawn spawn =
        new SimpleSpawn(
            action,
            /* arguments= */ ImmutableList.of(),
            /* environment= */ ImmutableMap.of(),
            /* executionInfo= */ ImmutableMap.of(),
            action.getInputs(),
            action.getOutputs(),
            ResourceSet.ZERO);
    computer.spawnExecuted(
        new SpawnExecutedEvent(
            spawn,
            new FakeActionInputFileCache(),
            null,
            new TestFileOutErr(),
            spawnResult,
            Instant.now(),
            /* spawnIdentifier= */ "1"));
    if (completeAction) {
      computer.actionComplete(
          new ActionCompletionEvent(
              startTime,
              clock.nanoTime(),
              action,
              new FakeActionInputFileCache(),
              mock(OutputMetadataStore.class),
              mock(ActionLookupData.class)));
    }
  }

  private void simulateSequentialAndParallelActionExec(
      Action action, ImmutableList<ImmutableList<Integer>> totalTimesInMs)
      throws InterruptedException {
    long startTime = clock.nanoTime();
    for (ImmutableList<Integer> parallelDuration : totalTimesInMs) {
      for (Integer phaseDuration : parallelDuration) {
        simulateActionExec(action, phaseDuration, phaseDuration, false);
      }
      computer.nextCriticalPathPhase(new ChangePhase(action));
    }
    computer.actionComplete(
        new ActionCompletionEvent(
            startTime,
            clock.nanoTime(),
            action,
            new FakeActionInputFileCache(),
            mock(OutputMetadataStore.class),
            mock(ActionLookupData.class)));
  }

  private Artifact derivedArtifact(String path) {
    DerivedArtifact artifact =
        (DerivedArtifact)
            ActionsTestUtil.createArtifactWithExecPath(
                derivedArtifactRoot, PathFragment.create(path));
    artifact.setGeneratingActionKey(ActionsTestUtil.NULL_ACTION_LOOKUP_DATA);
    return artifact;
  }

  private Artifact artifact(String path) {
    return ActionsTestUtil.createArtifactWithExecPath(artifactRoot, PathFragment.create(path));
  }

  private void checkCriticalPath(int totalWallTimeInMillis, String totalWallTimeStr) {
    AggregatedCriticalPath criticalPath = computer.aggregate();

    assertThat(criticalPath).isNotNull();
    assertThat(criticalPath.getAggregatedElapsedTime())
        .isEqualTo(Duration.ofMillis(totalWallTimeInMillis));

    String summary = criticalPath.toStringSummary();
    assertThat(summary).contains("Critical Path: " + totalWallTimeStr + "s");
  }

  @CanIgnoreReturnValue
  private AggregatedCriticalPath checkCriticalPath(
      Duration totalWallTime,
      Duration totalTime,
      Duration totalProcessTime,
      final String totalWallTimeStr,
      final String totalTimePercent,
      final String totalProcessPercent) {
    AggregatedCriticalPath criticalPath = computer.aggregate();

    assertThat(criticalPath).isNotNull();
    assertThat(criticalPath.getAggregatedElapsedTime())
        .isEqualTo(Duration.ofMillis(totalWallTime.toMillis()));
    assertThat(criticalPath.getSpawnMetrics().getRemoteMetrics().totalTimeInMs())
        .isEqualTo(totalTime.toMillis());
    assertThat(criticalPath.getSpawnMetrics().getRemoteMetrics().executionWallTimeInMs())
        .isEqualTo(totalProcessTime.toMillis());

    String summary = criticalPath.toStringSummary();
    assertThat(summary).contains("Critical Path: " + totalWallTimeStr + "s");
    assertThat(summary).contains("Remote (" + totalTimePercent + "% of the time)");
    assertThat(summary).contains("process: " + totalProcessPercent + "%");

    return criticalPath;
  }

  private static void checkTopComponentsTimes(CriticalPathComputer computer, long... times) {
    List<CriticalPathComponent> topComponents = computer.getSlowestComponents();
    assertThat(topComponents).hasSize(times.length);

    for (int i = 0; i < times.length; i++) {
      assertThat(topComponents.get(i).getElapsedTime()).isEqualTo(Duration.ofMillis(times[i]));
    }
  }

  private static SpawnResult.Builder createSpawnResult(int processTimeInMs) {
    SpawnResult.Builder spawnResult = new SpawnResult.Builder();
    spawnResult.setStatus(SpawnResult.Status.SUCCESS);
    spawnResult.setExitCode(0);
    spawnResult.setWallTimeInMs(processTimeInMs);
    spawnResult.setRunnerName("test");
    return spawnResult;
  }

  @Test
  public void testTreeFileDependency() throws Exception {
    SpecialArtifact tree =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(derivedArtifactRoot, "tree");

    // Action A produces the TreeArtifact
    MockAction actionA = new MockAction(ImmutableList.of(), ImmutableSet.of(tree));

    // Action B depends on a file INSIDE the TreeArtifact
    TreeFileArtifact child = TreeFileArtifact.createTreeOutput(tree, "file.txt");
    MockAction actionB =
        new MockAction(
            ImmutableList.of(child),
            ImmutableSet.of(ActionsTestUtil.createArtifact(derivedArtifactRoot, "b.out")));

    // Set up metadata
    child.getPath().getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(child.getPath(), "content");
    FileArtifactValue childMetadata = FileArtifactValue.createForTesting(child);
    TreeArtifactValue treeMetadata =
        TreeArtifactValue.newBuilder(tree).putChild(child, childMetadata).build();
    FakeActionInputFileCache cache = new FakeActionInputFileCache();
    cache.putTreeArtifact(tree, treeMetadata);

    // Simulate execution
    long startTimeA = clock.nanoTime();
    computer.actionStarted(new ActionStartedEvent(actionA, startTimeA));
    clock.advanceMillis(1000);
    computer.actionComplete(
        new ActionCompletionEvent(
            startTimeA,
            clock.nanoTime(),
            actionA,
            cache,
            null,
            ActionsTestUtil.NULL_ACTION_LOOKUP_DATA));

    long startTimeB = clock.nanoTime();
    computer.actionStarted(new ActionStartedEvent(actionB, startTimeB));
    clock.advanceMillis(1000);
    computer.actionComplete(
        new ActionCompletionEvent(
            startTimeB,
            clock.nanoTime(),
            actionB,
            cache,
            null,
            ActionsTestUtil.NULL_ACTION_LOOKUP_DATA));

    checkCriticalPath(2000, "2.00");
  }

  // Test scenario:
  //
  //   [actionA] (outputs TreeArtifact tree)
  //      │
  //      ▼
  //   [actionTemplateA] (inputs: tree, outputs: treeB)
  //      │
  //      ▼
  //   ┌───────────────────────────────────────────────────────────┐
  //   │ map_directory() Expansion Boundary                        │
  //   │                                                           │
  //   │  ├─► [actionA1] (outputs TreeFileArtifact treeB/file1.txt)│
  //   │  │      │                                                 │
  //   │  │      ▼                                                 │
  //   │  │   (treeB/file1.txt)                                    │
  //   │  │                                                        │
  //   └──┼────────────────────────────────────────────────────────┘
  //      │
  //      ▼
  //   [actionB] (inputs: treeB)
  //
  // Verify that the critical path of actionB correctly links to the child actionA1 (slowest
  // component of treeB)
  // even though actionB takes the top-level tree artifact treeB as input, and actionA1 depends on
  // childA1 inside tree.
  @Test
  public void testTreeFileDependency_fromTemplateExpansion() throws Exception {
    SpecialArtifact tree =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(derivedArtifactRoot, "tree");
    MockAction actionA = new MockAction(ImmutableList.of(), ImmutableSet.of(tree));

    SpecialArtifact treeB =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(derivedArtifactRoot, "treeB");

    ActionLookupKey templateKey =
        ActionTemplateExpansionValue.key(ActionsTestUtil.NULL_ARTIFACT_OWNER, 0);
    TreeFileArtifact childA1 = TreeFileArtifact.createTreeOutput(tree, "input_file.txt");
    TreeFileArtifact child1 = createTemplateExpansionOutput(treeB, "file1.txt", templateKey, 0);

    MockAction actionA1 = new MockAction(ImmutableList.of(childA1), ImmutableSet.of(child1));

    MockAction actionB =
        new MockAction(
            ImmutableList.of(treeB),
            ImmutableSet.of(ActionsTestUtil.createArtifact(derivedArtifactRoot, "b.out")));

    FakeActionInputFileCache cache = new FakeActionInputFileCache();
    TreeArtifactValue.Builder treeBuilder = TreeArtifactValue.newBuilder(tree);
    writeChildAndRegister(treeBuilder, childA1, "input");
    cache.putTreeArtifact(tree, treeBuilder.build());

    TreeArtifactValue.Builder treeBBuilder = TreeArtifactValue.newBuilder(treeB);
    writeChildAndRegister(treeBBuilder, child1, "content");
    cache.putTreeArtifact(treeB, treeBBuilder.build());

    runAction(actionA, 100, cache);
    runAction(actionA1, 1000, cache, ActionLookupData.create(templateKey, 0));
    runAction(actionB, 500, cache);

    // Critical path: actionA (100) -> actionA1 (1000) -> actionB (500) = 1600.
    checkCriticalPath(1600, "1.60");
  }

  // Test scenario:
  //
  //   [actionA] (outputs TreeArtifact tree)
  //      │
  //      ▼
  //   [actionTemplateA] (inputs: tree, outputs: treeB)
  //      │
  //      ▼
  //   ┌────────────────────────────────────────────────────────────────┐
  //   │ map_directory() Expansion Boundary                             │
  //   │                                                                │
  //   │  ├─► [actionA1] (outputs SubTreeArtifact treeB/subtree)        │
  //   │  │      │                                                      │
  //   │  │      ▼                                                      │
  //   │  │   (treeB/subtree/file)                                      │
  //   │  │                                                             │
  //   └──┼─────────────────────────────────────────────────────────────┘
  //      │
  //      ▼
  //   [actionB] (inputs: treeB)
  //
  // Verify that the critical path of actionB correctly links to the child actionA1 (slowest
  // component of treeB)
  // when the file is nested inside a subtree artifact, bubbling up through the subtree parent to
  // the top-level treeB.
  @Test
  public void testTreeFileDependency_nestedSubtree() throws Exception {
    SpecialArtifact tree =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(derivedArtifactRoot, "tree");
    MockAction actionA = new MockAction(ImmutableList.of(), ImmutableSet.of(tree));

    SpecialArtifact treeB =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(derivedArtifactRoot, "treeB");

    ActionLookupKey templateKey =
        ActionTemplateExpansionValue.key(ActionsTestUtil.NULL_ARTIFACT_OWNER, 0);
    SpecialArtifact subtree = createSubTreeOutput(treeB, "subtree", templateKey, 0);

    TreeFileArtifact childA1 = TreeFileArtifact.createTreeOutput(tree, "input_file.txt");
    TreeFileArtifact child =
        TreeFileArtifact.createTemplateExpansionOutput(subtree, "file.txt", templateKey);

    MockAction actionA1 = new MockAction(ImmutableList.of(childA1), ImmutableSet.of(subtree));

    MockAction actionB =
        new MockAction(
            ImmutableList.of(treeB),
            ImmutableSet.of(ActionsTestUtil.createArtifact(derivedArtifactRoot, "b.out")));

    FakeActionInputFileCache cache = new FakeActionInputFileCache();
    TreeArtifactValue.Builder treeBuilder = TreeArtifactValue.newBuilder(tree);
    writeChildAndRegister(treeBuilder, childA1, "input");
    cache.putTreeArtifact(tree, treeBuilder.build());

    TreeArtifactValue.Builder treeBBuilder = TreeArtifactValue.newBuilder(treeB);
    writeChildAndRegister(treeBBuilder, child, "content");
    cache.putTreeArtifact(treeB, treeBBuilder.build());

    runAction(actionA, 100, cache);
    runAction(actionA1, 1000, cache, ActionLookupData.create(templateKey, 0));
    runAction(actionB, 500, cache);

    // Critical path: actionA (100) -> actionA1 (1000) -> actionB (500) = 1600.
    checkCriticalPath(1600, "1.60");
  }

  // Test scenario:
  //
  //   [actionA] (outputs TreeArtifact tree)
  //      │
  //      ▼
  //   [actionTemplateA] (inputs: tree, outputs: treeB)
  //      │
  //      ▼
  //   ┌────────────────────────────────────────────────────────────────────────┐
  //   │ first map_directory() Expansion Boundary                               │
  //   │                                                                        │
  //   │  ├─► [actionA1] (outputs TreeFileArtifact treeB/file1.txt)             │
  //   │  │                                                                     │
  //   │  └─► [actionA2] (outputs SubTreeArtifact treeB/subtree)                │
  //   │                 │                                                      │
  //   └─────────────────┼──────────────────────────────────────────────────────┘
  //   │                 │
  //   │                 ▼
  //   │          (treeB/subtree)
  //   │                 │
  //   │                 ├─► (treeB/subtree/file2.txt) (implicit member)
  //   │                 │
  //   ▼                 │
  // [actionTemplateB]   │
  //   │                 │
  //   ▼                 │
  //   ┌─────────────────┼──────────────────────────────────────────────────────┐
  //   │ second map_directory() Expansion Boundary                              │
  //   │                 │                                                      │
  //   │  ├─► [actionB2] (inputs: treeB/subtree/file2.txt,                      │
  //   │  │                 outputs: TreeFileArtifact treeC/b2.out)             │
  //   │  │                                                                     │
  //   └────────────────────────────────────────────────────────────────────────┘
  //
  // Verify that actionB2 (which depends on a file inside the subtree) correctly links to actionA2
  // (which
  // generated the subtree as a whole), by traversing up the parent chain when the input has no
  // direct generating action.
  @Test
  public void testSubTreeFileDependency_fromTemplateExpansion() throws Exception {
    SpecialArtifact tree =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(derivedArtifactRoot, "tree");
    MockAction actionA = new MockAction(ImmutableList.of(), ImmutableSet.of(tree));

    SpecialArtifact treeB =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(derivedArtifactRoot, "treeB");

    ActionLookupKey templateKey1 =
        ActionTemplateExpansionValue.key(ActionsTestUtil.NULL_ARTIFACT_OWNER, 0);

    TreeFileArtifact childA1 = TreeFileArtifact.createTreeOutput(tree, "input_file.txt");

    // actionA1 -> output treeB/file1.txt
    TreeFileArtifact child1 = createTemplateExpansionOutput(treeB, "file1.txt", templateKey1, 0);
    MockAction actionA1 = new MockAction(ImmutableList.of(childA1), ImmutableSet.of(child1));

    // actionA2 -> output treeB/subtree (SubTreeArtifact)
    SpecialArtifact subtree = createSubTreeOutput(treeB, "subtree", templateKey1, 1);
    MockAction actionA2 = new MockAction(ImmutableList.of(childA1), ImmutableSet.of(subtree));

    // child2 (treeB/subtree/file2.txt) is inside the subtree, but not explicitly outputted by
    // actionA2.
    TreeFileArtifact child2 =
        TreeFileArtifact.createTemplateExpansionOutput(subtree, "file2.txt", templateKey1);

    SpecialArtifact treeC =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(derivedArtifactRoot, "treeC");

    ActionLookupKey templateKey2 =
        ActionTemplateExpansionValue.key(ActionsTestUtil.NULL_ARTIFACT_OWNER, 1);
    TreeFileArtifact childC2 = createTemplateExpansionOutput(treeC, "b2.out", templateKey2, 0);

    // actionB2 -> input child2 (treeB/subtree/file2.txt), output childC2 (treeC/b2.out)
    MockAction actionB2 = new MockAction(ImmutableList.of(child2), ImmutableSet.of(childC2));

    FakeActionInputFileCache cache = new FakeActionInputFileCache();
    TreeArtifactValue.Builder treeBuilder = TreeArtifactValue.newBuilder(tree);
    writeChildAndRegister(treeBuilder, childA1, "input");
    cache.putTreeArtifact(tree, treeBuilder.build());

    TreeArtifactValue.Builder treeBBuilder = TreeArtifactValue.newBuilder(treeB);
    writeChildAndRegister(treeBBuilder, child1, "content1");
    writeChildAndRegister(treeBBuilder, child2, "content2");
    cache.putTreeArtifact(treeB, treeBBuilder.build());

    // Caching treeC isn't required for actionB2 execution in this test, but childC2 has to exist on
    // disk.
    childC2.getPath().getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(childC2.getPath(), "contentC2");

    runAction(actionA, 100, cache);
    runAction(actionA1, 1000, cache, ActionLookupData.create(templateKey1, 0));
    runAction(actionA2, 2000, cache, ActionLookupData.create(templateKey1, 1));
    runAction(actionB2, 500, cache, ActionLookupData.create(templateKey2, 0));

    // Critical path of actionB2 should be:
    // actionA (100) -> actionA2 (2000) -> actionB2 (500) = 2600.
    checkCriticalPath(2600, "2.60");
  }

  // Test scenario:
  //
  //   [actionA] (outputs TreeArtifact tree)
  //      │
  //      ▼
  //   [actionTemplateA] (inputs: tree, outputs: treeB)
  //      │
  //      ▼
  //   ┌────────────────────────────────────────────────────────────────────────┐
  //   │ first map_directory() Expansion Boundary                               │
  //   │                                                                        │
  //   │  └──► [actionA1] (outputs TreeFileArtifact treeB/file1.txt)            │
  //   │                 │                                                      │
  //   └─────────────────┼──────────────────────────────────────────────────────┘
  //   │                 │
  //   │                 ▼
  //   │          (treeB/file1.txt)
  //   │                 │
  //   ▼                 │
  // [actionTemplateB]   │
  //   │                 │
  //   ▼                 │
  //   ┌─────────────────┼──────────────────────────────────────────────────────┐
  //   │ second map_directory() Expansion Boundary                              │
  //   │                 │                                                      │
  //   │  └──► [actionB1] (inputs: treeB/file1.txt,                             │
  //   │                   outputs: TreeFileArtifact treeC/b1.out)              │
  //   └────────────────────────────────────────────────────────────────────────┘
  //
  // Verify that actionB1 (in the second expansion) correctly links to actionA1 (in the first
  // expansion)
  // which generated treeB/file1.txt directly.
  @Test
  public void testTreeFileDependency_nestedExpansion() throws Exception {
    SpecialArtifact tree =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(derivedArtifactRoot, "tree");
    MockAction actionA = new MockAction(ImmutableList.of(), ImmutableSet.of(tree));

    SpecialArtifact treeB =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(derivedArtifactRoot, "treeB");

    ActionLookupKey templateKey1 =
        ActionTemplateExpansionValue.key(ActionsTestUtil.NULL_ARTIFACT_OWNER, 0);

    TreeFileArtifact childA1 = TreeFileArtifact.createTreeOutput(tree, "input_file.txt");

    // actionA1 -> output treeB/file1.txt
    TreeFileArtifact child1 = createTemplateExpansionOutput(treeB, "file1.txt", templateKey1, 0);
    MockAction actionA1 = new MockAction(ImmutableList.of(childA1), ImmutableSet.of(child1));

    SpecialArtifact treeC =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(derivedArtifactRoot, "treeC");

    ActionLookupKey templateKey2 =
        ActionTemplateExpansionValue.key(ActionsTestUtil.NULL_ARTIFACT_OWNER, 1);
    TreeFileArtifact childC1 = createTemplateExpansionOutput(treeC, "b1.out", templateKey2, 0);

    // actionB1 -> input child1 (treeB/file1.txt), output childC1 (treeC/b1.out)
    MockAction actionB1 = new MockAction(ImmutableList.of(child1), ImmutableSet.of(childC1));

    FakeActionInputFileCache cache = new FakeActionInputFileCache();
    TreeArtifactValue.Builder treeBuilder = TreeArtifactValue.newBuilder(tree);
    writeChildAndRegister(treeBuilder, childA1, "input");
    cache.putTreeArtifact(tree, treeBuilder.build());

    TreeArtifactValue.Builder treeBBuilder = TreeArtifactValue.newBuilder(treeB);
    writeChildAndRegister(treeBBuilder, child1, "content1");
    cache.putTreeArtifact(treeB, treeBBuilder.build());

    // Caching treeC isn't required for actionB1 execution in this test, but childC1 has to exist on
    // disk.
    childC1.getPath().getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(childC1.getPath(), "contentC1");

    runAction(actionA, 100, cache);
    runAction(actionA1, 1000, cache, ActionLookupData.create(templateKey1, 0));
    runAction(actionB1, 500, cache, ActionLookupData.create(templateKey2, 0));

    // Critical path of actionB1 should be:
    // actionA (100) -> actionA1 (1000) -> actionB1 (500) = 1600.
    checkCriticalPath(1600, "1.60");
  }

  @Test
  public void testTreeFileDependency_slowSiblingTemplateAction() throws Exception {
    SpecialArtifact tree =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(derivedArtifactRoot, "tree");
    MockAction actionA = new MockAction(ImmutableList.of(), ImmutableSet.of(tree));

    SpecialArtifact treeB =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(derivedArtifactRoot, "treeB");

    ActionLookupKey templateKey1 =
        ActionTemplateExpansionValue.key(ActionsTestUtil.NULL_ARTIFACT_OWNER, 0);

    TreeFileArtifact childA1 = TreeFileArtifact.createTreeOutput(tree, "input_file1.txt");
    TreeFileArtifact childA2 = TreeFileArtifact.createTreeOutput(tree, "input_file2.txt");

    // actionA1 -> output treeB/file1.txt (fast: 100ms)
    TreeFileArtifact childB1 = createTemplateExpansionOutput(treeB, "file1.txt", templateKey1, 0);
    MockAction actionA1 = new MockAction(ImmutableList.of(childA1), ImmutableSet.of(childB1));

    // actionA2 -> output treeB/file2.txt (slow: 1000ms)
    TreeFileArtifact childB2 = createTemplateExpansionOutput(treeB, "file2.txt", templateKey1, 1);
    MockAction actionA2 = new MockAction(ImmutableList.of(childA2), ImmutableSet.of(childB2));

    SpecialArtifact treeC =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(derivedArtifactRoot, "treeC");

    ActionLookupKey templateKey2 =
        ActionTemplateExpansionValue.key(ActionsTestUtil.NULL_ARTIFACT_OWNER, 1);

    // actionB1 (inputs treeB/file1.txt from actionA1, outputs treeC/out1.txt, duration: 500ms)
    TreeFileArtifact childC1 = createTemplateExpansionOutput(treeC, "out1.txt", templateKey2, 0);
    MockAction actionB1 = new MockAction(ImmutableList.of(childB1), ImmutableSet.of(childC1));

    FakeActionInputFileCache cache = new FakeActionInputFileCache();
    TreeArtifactValue.Builder treeBuilder = TreeArtifactValue.newBuilder(tree);
    writeChildAndRegister(treeBuilder, childA1, "input1");
    writeChildAndRegister(treeBuilder, childA2, "input2");
    cache.putTreeArtifact(tree, treeBuilder.build());

    TreeArtifactValue.Builder treeBBuilder = TreeArtifactValue.newBuilder(treeB);
    writeChildAndRegister(treeBBuilder, childB1, "content1");
    writeChildAndRegister(treeBBuilder, childB2, "content2");
    cache.putTreeArtifact(treeB, treeBBuilder.build());

    childC1.getPath().getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(childC1.getPath(), "out1");

    runAction(actionA, 100, cache);
    runAction(actionA1, 100, cache, ActionLookupData.create(templateKey1, 0));
    runAction(actionA2, 1000, cache, ActionLookupData.create(templateKey1, 1));
    runAction(actionB1, 500, cache, ActionLookupData.create(templateKey2, 0));

    // Critical path of actionB1 should be:
    // actionA (100) -> actionA2 (1000) -> actionB1 (500) = 1600.
    checkCriticalPath(1600, "1.60");
  }

  // Test scenario:
  //
  //   [actionA] (outputs TreeArtifact tree)
  //      │
  //      ▼
  //   [actionTemplateA] (inputs: tree, outputs: treeB)
  //      │
  //      ▼
  //   ┌────────────────────────────────────────────────────────────────────────┐
  //   │ map_directory() Expansion Boundary                                     │
  //   │                                                                        │
  //   │  ├─► [actionA1] (outputs TreeFileArtifact treeB/file1.txt)             │
  //   │  │              │                                                      │
  //   │  │              ▼                                                      │
  //   │  │       (treeB/file1.txt)                                             │
  //   │  │              │                                                      │
  //   │  │              ▼                                                      │
  //   │  ├─► [actionA2] (inputs: treeB/file1.txt,                              │
  //   │  │               outputs: SubTreeArtifact treeB/subtree)               │
  //   │  │              │                                                      │
  //   │  │              ▼                                                      │
  //   │  │       (treeB/subtree)                                               │
  //   │  │              │                                                      │
  //   │  │              ├─► (treeB/subtree/file2.txt) (implicit member)        │
  //   │  │              │                                                      │
  //   │  │              ▼                                                      │
  //   │  └─► [actionA3] (inputs: treeB/subtree/file2.txt,                      │
  //   │                  outputs: TreeFileArtifact treeB/file3.txt)            │
  //   └────────────────────────────────────────────────────────────────────────┘
  //
  // Verify that inside a single map_directory() expansion, a chain of dependent actions
  // (A1 -> A2 -> A3) is correctly linked in the critical path, even when the dependencies
  // go through intermediate tree file artifacts (file1.txt) and subdirectories (subtree/file2.txt).
  @Test
  public void testTreeFileDependency_chainWithinExpansion() throws Exception {
    SpecialArtifact tree =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(derivedArtifactRoot, "tree");
    MockAction actionA = new MockAction(ImmutableList.of(), ImmutableSet.of(tree));

    SpecialArtifact treeB =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(derivedArtifactRoot, "treeB");

    ActionLookupKey templateKey =
        ActionTemplateExpansionValue.key(ActionsTestUtil.NULL_ARTIFACT_OWNER, 0);

    TreeFileArtifact childA1 = TreeFileArtifact.createTreeOutput(tree, "input_file.txt");

    // actionA1 -> output treeB/file1.txt
    TreeFileArtifact child1 = createTemplateExpansionOutput(treeB, "file1.txt", templateKey, 0);
    MockAction actionA1 = new MockAction(ImmutableList.of(childA1), ImmutableSet.of(child1));

    // actionA2 -> input treeB/file1.txt, output treeB/subtree (SubTreeArtifact)
    SpecialArtifact subtree = createSubTreeOutput(treeB, "subtree", templateKey, 1);
    MockAction actionA2 = new MockAction(ImmutableList.of(child1), ImmutableSet.of(subtree));

    // child2 (treeB/subtree/file2.txt) is inside the subtree, but not explicitly outputted by
    // actionA2.
    TreeFileArtifact child2 =
        TreeFileArtifact.createTemplateExpansionOutput(subtree, "file2.txt", templateKey);

    // actionA3 -> input child2 (treeB/subtree/file2.txt), output treeB/file3.txt
    TreeFileArtifact child3 = createTemplateExpansionOutput(treeB, "file3.txt", templateKey, 2);
    MockAction actionA3 = new MockAction(ImmutableList.of(child2), ImmutableSet.of(child3));

    FakeActionInputFileCache cache = new FakeActionInputFileCache();
    TreeArtifactValue.Builder treeBuilder = TreeArtifactValue.newBuilder(tree);
    writeChildAndRegister(treeBuilder, childA1, "input");
    cache.putTreeArtifact(tree, treeBuilder.build());

    TreeArtifactValue.Builder treeBBuilder = TreeArtifactValue.newBuilder(treeB);
    writeChildAndRegister(treeBBuilder, child1, "content1");
    writeChildAndRegister(treeBBuilder, child2, "content2");
    writeChildAndRegister(treeBBuilder, child3, "content3");
    cache.putTreeArtifact(treeB, treeBBuilder.build());

    runAction(actionA, 100, cache);
    runAction(actionA1, 1000, cache, ActionLookupData.create(templateKey, 0));
    runAction(actionA2, 2000, cache, ActionLookupData.create(templateKey, 1));
    runAction(actionA3, 3000, cache, ActionLookupData.create(templateKey, 2));

    // Critical path of actionA3 should be:
    // actionA (100) -> actionA1 (1000) -> actionA2 (2000) -> actionA3 (3000) = 6100.
    checkCriticalPath(6100, "6.10");
  }

  private TreeFileArtifact createTemplateExpansionOutput(
      SpecialArtifact parent, String relativePath, ActionLookupKey templateKey, int actionIndex) {
    TreeFileArtifact child =
        TreeFileArtifact.createTemplateExpansionOutput(parent, relativePath, templateKey);
    child.setGeneratingActionKey(ActionLookupData.create(templateKey, actionIndex));
    return child;
  }

  private SpecialArtifact createSubTreeOutput(
      SpecialArtifact parent, String relativePath, ActionLookupKey templateKey, int actionIndex) {
    SpecialArtifact subtree =
        SpecialArtifact.createSubTreeArtifact(
            parent, PathFragment.create(relativePath), templateKey);
    subtree.setGeneratingActionKey(ActionLookupData.create(templateKey, actionIndex));
    return subtree;
  }

  private void writeChildAndRegister(
      TreeArtifactValue.Builder builder, TreeFileArtifact child, String content) throws Exception {
    child.getPath().getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(child.getPath(), content);
    builder.putChild(child, FileArtifactValue.createForTesting(child));
  }

  private void runAction(
      MockAction action, int durationMillis, InputMetadataProvider metadataProvider)
      throws InterruptedException {
    runAction(action, durationMillis, metadataProvider, ActionsTestUtil.NULL_ACTION_LOOKUP_DATA);
  }

  private void runAction(
      MockAction action,
      int durationMillis,
      InputMetadataProvider metadataProvider,
      ActionLookupData lookupData)
      throws InterruptedException {
    long startTime = clock.nanoTime();
    computer.actionStarted(new ActionStartedEvent(action, startTime));
    clock.advanceMillis(durationMillis);
    computer.actionComplete(
        new ActionCompletionEvent(
            startTime, clock.nanoTime(), action, metadataProvider, null, lookupData));
  }

  private static SpawnResult.Builder createSpawnResult() {
    return createSpawnResult(0);
  }
}
