// Copyright 2022 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.extensions.proto.ProtoTruth.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;

import com.github.luben.zstd.ZstdInputStream;
import com.github.luben.zstd.ZstdOutputStream;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionCompletionEvent;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.DiscoveredInputsEvent;
import com.google.devtools.build.lib.actions.ExecutionGraph;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SimpleSpawn;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnExecutedEvent;
import com.google.devtools.build.lib.actions.SpawnMetrics;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.buildtool.BuildResult.BuildToolLogCollection;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.exec.util.FakeOwner;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.runtime.ExecutionGraphModule.ActionDumpWriter;
import com.google.devtools.build.lib.runtime.ExecutionGraphModule.DependencyInfo;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.time.Instant;
import java.util.UUID;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.ArgumentCaptor;

/** Unit tests for {@link ExecutionGraphModule}. */
@RunWith(TestParameterInjector.class)
public class ExecutionGraphModuleTest extends FoundationTestCase {
  private ExecutionGraphModule module;
  private ArtifactRoot artifactRoot;

  @Before
  public void createModule() {
    module = new ExecutionGraphModule();
  }

  @Before
  public final void initializeRoots() throws Exception {
    artifactRoot = ArtifactRoot.asDerivedRoot(scratch.resolve("/"), RootType.Output, "output");
  }

  private static ImmutableList<ExecutionGraph.Node> parse(ByteArrayOutputStream buffer)
      throws IOException {
    byte[] data = buffer.toByteArray();
    try (InputStream in = new ZstdInputStream(new ByteArrayInputStream(data))) {
      ImmutableList.Builder<ExecutionGraph.Node> nodeListBuilder = new ImmutableList.Builder<>();
      ExecutionGraph.Node node;
      while ((node = ExecutionGraph.Node.parseDelimitedFrom(in)) != null) {
        nodeListBuilder.add(node);
      }
      return nodeListBuilder.build();
    }
  }

  @Test
  public void testOneSpawn() throws IOException {
    UUID uuid = UUID.randomUUID();
    ByteArrayOutputStream buffer = new ByteArrayOutputStream();
    Spawn spawn =
        new SimpleSpawn(
            new FakeOwnerWithPrimaryOutput(
                "Mnemonic", "Progress message", "//foo", "output/foo/out"),
            ImmutableList.of("cmd"),
            ImmutableMap.of("env", "value"),
            ImmutableMap.of("exec", "value"),
            /* inputs= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            /* outputs= */ ImmutableSet.of(ActionInputHelper.fromPath("output/foo/out")),
            ResourceSet.ZERO);
    SpawnResult result =
        new SpawnResult.Builder()
            .setRunnerName("local")
            .setStatus(Status.SUCCESS)
            .setExitCode(0)
            .setSpawnMetrics(
                SpawnMetrics.Builder.forLocalExec()
                    .setTotalTimeInMs(1234)
                    .setExecutionWallTimeInMs(2345)
                    .setProcessOutputsTimeInMs(3456)
                    .build())
            .build();
    startLogging(eventBus, uuid, buffer, DependencyInfo.NONE);
    Instant startTimeInstant = Instant.now();
    module.spawnExecuted(new SpawnExecutedEvent(spawn, result, startTimeInstant));
    module.buildComplete(
        new BuildCompleteEvent(new BuildResult(startTimeInstant.toEpochMilli() + 1000)));

    ImmutableList<ExecutionGraph.Node> nodes = parse(buffer);
    assertThat(nodes).hasSize(1);
    assertThat(nodes.get(0).getTargetLabel()).isEqualTo("//foo:foo");
    assertThat(nodes.get(0).getMnemonic()).isEqualTo("Mnemonic");
    assertThat(nodes.get(0).getMetrics().getDurationMillis()).isEqualTo(1234L);
    assertThat(nodes.get(0).getMetrics().getFetchMillis()).isEqualTo(0);
    assertThat(nodes.get(0).getMetrics().getProcessOutputsMillis()).isEqualTo(3456);
    assertThat(nodes.get(0).getMetrics().getStartTimestampMillis())
        .isEqualTo(startTimeInstant.toEpochMilli());
    assertThat(nodes.get(0).getIndex()).isEqualTo(0);
    assertThat(nodes.get(0).getDependentIndexList()).isEmpty();
  }

  @Test
  public void testSpawnWithDiscoverInputs() throws IOException {
    UUID uuid = UUID.randomUUID();
    ByteArrayOutputStream buffer = new ByteArrayOutputStream();
    Spawn spawn =
        new SimpleSpawn(
            new FakeOwnerWithPrimaryOutput(
                "Mnemonic", "Progress message", "//foo", "output/foo/out"),
            ImmutableList.of("cmd"),
            ImmutableMap.of("env", "value"),
            ImmutableMap.of("exec", "value"),
            /* inputs= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            /* outputs= */ ImmutableSet.of(createOutputArtifact("output/foo/out")),
            ResourceSet.ZERO);
    SpawnResult result =
        new SpawnResult.Builder()
            .setRunnerName("local")
            .setStatus(Status.SUCCESS)
            .setExitCode(0)
            .setSpawnMetrics(
                SpawnMetrics.Builder.forLocalExec()
                    .setTotalTimeInMs(1234)
                    .setExecutionWallTimeInMs(2345)
                    .setProcessOutputsTimeInMs(3456)
                    .setParseTimeInMs(2000)
                    .build())
            .build();
    startLogging(eventBus, uuid, buffer, DependencyInfo.NONE);
    Instant startTimeInstant = Instant.ofEpochMilli(999888777L);
    module.discoverInputs(
        new DiscoveredInputsEvent(
            SpawnMetrics.Builder.forOtherExec().setParseTimeInMs(987).setTotalTimeInMs(987).build(),
            new ActionsTestUtil.NullAction(createOutputArtifact("output/foo/out")),
            0));
    module.spawnExecuted(new SpawnExecutedEvent(spawn, result, startTimeInstant));
    module.buildComplete(
        new BuildCompleteEvent(new BuildResult(startTimeInstant.toEpochMilli() + 1000)));

    ImmutableList<ExecutionGraph.Node> nodes = parse(buffer);
    ExecutionGraph.Metrics metrics = nodes.get(0).getMetrics();
    assertThat(metrics.getDurationMillis()).isEqualTo(2221);
    assertThat(metrics.getFetchMillis()).isEqualTo(0);
    assertThat(metrics.getProcessMillis()).isEqualTo(2345);
    assertThat(metrics.getProcessOutputsMillis()).isEqualTo(3456);
    assertThat(metrics.getParseMillis()).isEqualTo(2000);
    assertThat(metrics.getDiscoverInputsMillis()).isEqualTo(987);
  }

  @Test
  public void actionDepsWithThreeSpawns() throws IOException {
    UUID uuid = UUID.randomUUID();
    ByteArrayOutputStream buffer = new ByteArrayOutputStream();

    ActionInput out1 = ActionInputHelper.fromPath("output/foo/out1");
    ActionInput out2 = ActionInputHelper.fromPath("output/foo/out2");
    ActionInput outTop = ActionInputHelper.fromPath("output/foo/out.top");

    Spawn spawnOut1 =
        new SimpleSpawn(
            new FakeOwnerWithPrimaryOutput(
                "Mnemonic", "Progress message", "//foo", out1.getExecPathString()),
            ImmutableList.of("cmd"),
            ImmutableMap.of("env", "value"),
            ImmutableMap.of("exec", "value"),
            /* inputs= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            /* outputs= */ ImmutableSet.of(out1),
            ResourceSet.ZERO);
    Spawn spawnOut2 =
        new SimpleSpawn(
            new FakeOwnerWithPrimaryOutput(
                "Mnemonic", "Progress message", "//foo", out2.getExecPathString()),
            ImmutableList.of("cmd"),
            ImmutableMap.of("env", "value"),
            ImmutableMap.of("exec", "value"),
            /* inputs= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            /* outputs= */ ImmutableSet.of(out2),
            ResourceSet.ZERO);
    Spawn spawnTop =
        new SimpleSpawn(
            new FakeOwnerWithPrimaryOutput(
                "Mnemonic", "Progress message", "//foo", outTop.getExecPathString()),
            ImmutableList.of("cmd"),
            ImmutableMap.of("env", "value"),
            ImmutableMap.of("exec", "value"),
            /* inputs= */ NestedSetBuilder.create(Order.COMPILE_ORDER, out1, out2),
            /* outputs= */ ImmutableSet.of(outTop),
            ResourceSet.ZERO);
    SpawnResult result =
        new SpawnResult.Builder()
            .setRunnerName("local")
            .setStatus(Status.SUCCESS)
            .setExitCode(0)
            .setSpawnMetrics(
                SpawnMetrics.Builder.forLocalExec()
                    .setTotalTimeInMs(1234)
                    .setExecutionWallTimeInMs(2345)
                    .setProcessOutputsTimeInMs(3456)
                    .build())
            .build();
    startLogging(eventBus, uuid, buffer, DependencyInfo.ALL);
    Instant startTimeInstant = Instant.now();
    module.spawnExecuted(new SpawnExecutedEvent(spawnOut1, result, startTimeInstant));
    module.spawnExecuted(new SpawnExecutedEvent(spawnOut2, result, startTimeInstant));
    module.spawnExecuted(new SpawnExecutedEvent(spawnTop, result, startTimeInstant));
    module.buildComplete(
        new BuildCompleteEvent(new BuildResult(startTimeInstant.plusMillis(1000).toEpochMilli())));

    ImmutableList<ExecutionGraph.Node> nodes = parse(buffer);
    assertThat(nodes).hasSize(3);

    assertThat(nodes.get(0).getIndex()).isEqualTo(0);
    assertThat(nodes.get(0).getDependentIndexList()).isEmpty();

    assertThat(nodes.get(1).getIndex()).isEqualTo(1);
    assertThat(nodes.get(1).getDependentIndexList()).isEmpty();

    assertThat(nodes.get(2).getIndex()).isEqualTo(2);
    assertThat(nodes.get(2).getDependentIndexList()).containsExactly(0, 1);
  }

  private enum FailingOutputStreamFactory {
    CLOSE {
      @Override
      public ZstdOutputStream get() throws IOException {
        return new ZstdOutputStream(OutputStream.nullOutputStream()) {
          @Override
          public synchronized void close() throws IOException {
            throw new IOException("Simulated close failure");
          }
        };
      }
    },
    /** Called from {@link com.google.protobuf.CodedOutputStream#flush}. */
    WRITE {
      @Override
      public ZstdOutputStream get() throws IOException {
        return new ZstdOutputStream(OutputStream.nullOutputStream()) {
          @Override
          public synchronized void write(byte[] b, int off, int len) throws IOException {
            throw new IOException("oh no!");
          }
        };
      }
    };

    abstract ZstdOutputStream get() throws IOException;
  }

  /** Regression test for b/218721483. */
  @Test(timeout = 30_000)
  public void failureInOutputDoesNotHang(
      @TestParameter FailingOutputStreamFactory failingOutputStream) {
    UUID uuid = UUID.randomUUID();
    ActionDumpWriter writer =
        new ActionDumpWriter(
            BugReporter.defaultInstance(),
            /* localLockFreeOutputEnabled= */ false,
            /* logFileWriteEdges= */ false,
            OutputStream.nullOutputStream(),
            uuid,
            DependencyInfo.NONE,
            -1) {
          @Override
          protected void updateLogs(BuildToolLogCollection logs) {}

          @Override
          protected ZstdOutputStream createCompressingOutputStream() throws IOException {
            return failingOutputStream.get();
          }
        };
    module.setWriter(writer);
    eventBus.register(module);

    Instant startTimeInstant = Instant.now();
    eventBus.post(new BuildCompleteEvent(new BuildResult(startTimeInstant.toEpochMilli() + 1000)));
  }

  private void startLogging(
      EventBus eventBus, UUID uuid, OutputStream buffer, DependencyInfo depType) {
    startLogging(
        eventBus,
        BugReporter.defaultInstance(),
        /* localLockFreeOutputEnabled= */ false,
        /* logFileWriteEdges= */ false,
        uuid,
        buffer,
        depType);
  }

  private void startLogging(
      EventBus eventBus,
      BugReporter bugReporter,
      boolean localLockFreeOutputEnabled,
      boolean logFileWriteEdges,
      UUID uuid,
      OutputStream buffer,
      DependencyInfo depType) {
    ActionDumpWriter writer =
        new ActionDumpWriter(
            bugReporter, localLockFreeOutputEnabled, logFileWriteEdges, buffer, uuid, depType, -1) {
          @Override
          protected void updateLogs(BuildToolLogCollection logs) {}
        };
    module.setWriter(writer);
    eventBus.register(module);
  }

  @Test
  public void shutDownWithoutStartTolerated() {
    eventBus.register(module);
    Instant startTimeInstant = Instant.now();
    // Doesn't crash.
    eventBus.post(new BuildCompleteEvent(new BuildResult(startTimeInstant.toEpochMilli() + 1000)));
  }

  @Test
  public void testSpawnWithNullOwnerLabel() throws IOException {
    UUID uuid = UUID.randomUUID();
    ByteArrayOutputStream buffer = new ByteArrayOutputStream();
    Spawn spawn =
        new SimpleSpawn(
            new FakeOwnerWithPrimaryOutput(
                "Mnemonic", "Progress message", "//unused:label", "output/foo/out") {
              @Override
              public ActionOwner getOwner() {
                return ActionOwner.SYSTEM_ACTION_OWNER;
              }
            },
            ImmutableList.of("cmd"),
            ImmutableMap.of("env", "value"),
            ImmutableMap.of("exec", "value"),
            /* inputs= */ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            /* outputs= */ ImmutableSet.of(ActionInputHelper.fromPath("output/foo/out")),
            ResourceSet.ZERO);
    SpawnResult result =
        new SpawnResult.Builder()
            .setRunnerName("local")
            .setStatus(Status.SUCCESS)
            .setExitCode(0)
            .setSpawnMetrics(
                SpawnMetrics.Builder.forLocalExec()
                    .setTotalTimeInMs(1234)
                    .setExecutionWallTimeInMs(2345)
                    .setProcessOutputsTimeInMs(3456)
                    .build())
            .build();
    startLogging(eventBus, uuid, buffer, DependencyInfo.NONE);
    Instant startTimeInstant = Instant.now();
    module.spawnExecuted(new SpawnExecutedEvent(spawn, result, startTimeInstant));
    module.buildComplete(
        new BuildCompleteEvent(new BuildResult(startTimeInstant.toEpochMilli() + 1000)));

    ImmutableList<ExecutionGraph.Node> nodes = parse(buffer);
    assertThat(nodes).hasSize(1);
    assertThat(nodes.get(0).getTargetLabel()).isEmpty();
  }

  @Test
  public void spawnAndAction_withSameOutputs() throws Exception {
    var buffer = new ByteArrayOutputStream();
    startLogging(eventBus, UUID.randomUUID(), buffer, DependencyInfo.ALL);
    var options = new ExecutionGraphModule.ExecutionGraphOptions();
    options.logMissedActions = true;
    module.setOptions(options);

    module.spawnExecuted(
        new SpawnExecutedEvent(
            new SpawnBuilder().withOwnerPrimaryOutput(createOutputArtifact("foo/out")).build(),
            createRemoteSpawnResult(200),
            Instant.ofEpochMilli(100)));
    module.actionComplete(
        new ActionCompletionEvent(
            0, 0, new ActionsTestUtil.NullAction(createOutputArtifact("foo/out")), null));
    module.buildComplete(new BuildCompleteEvent(new BuildResult(1000)));

    assertThat(parse(buffer))
        .containsExactly(
            executionGraphNodeBuilderForSpawnBuilderSpawn()
                .setIndex(0)
                .setMetrics(
                    ExecutionGraph.Metrics.newBuilder()
                        .setStartTimestampMillis(100)
                        .setDurationMillis(200)
                        .setOtherMillis(200))
                .setRunner("remote")
                .build());
  }

  @Test
  public void spawnAndAction_withDifferentOutputs() throws Exception {
    var buffer = new ByteArrayOutputStream();
    startLogging(eventBus, UUID.randomUUID(), buffer, DependencyInfo.ALL);
    var options = new ExecutionGraphModule.ExecutionGraphOptions();
    options.logMissedActions = true;
    module.setOptions(options);
    var nanosToMillis = BlazeClock.createNanosToMillisSinceEpochConverter();
    module.setNanosToMillis(nanosToMillis);

    module.spawnExecuted(
        new SpawnExecutedEvent(
            new SpawnBuilder().withOwnerPrimaryOutput(createOutputArtifact("foo/out")).build(),
            createRemoteSpawnResult(200),
            Instant.ofEpochMilli(100)));
    var action = new ActionsTestUtil.NullAction(createOutputArtifact("bar/out"));
    module.actionComplete(new ActionCompletionEvent(0, 0, action, null));
    module.buildComplete(new BuildCompleteEvent(new BuildResult(1000)));

    assertThat(parse(buffer))
        .containsExactly(
            executionGraphNodeBuilderForSpawnBuilderSpawn()
                .setIndex(0)
                .setMetrics(
                    ExecutionGraph.Metrics.newBuilder()
                        .setStartTimestampMillis(100)
                        .setDurationMillis(200)
                        .setOtherMillis(200))
                .setRunner("remote")
                .build(),
            executionGraphNodeBuilderForAction(action)
                .setIndex(1)
                .setMetrics(
                    ExecutionGraph.Metrics.newBuilder()
                        .setStartTimestampMillis(nanosToMillis.toEpochMillis(0)))
                .build());
  }

  @Test
  public void multipleSpawnsWithSameOutput_recordsBothSpawnsWithRetry() throws Exception {
    var buffer = new ByteArrayOutputStream();
    startLogging(eventBus, UUID.randomUUID(), buffer, DependencyInfo.ALL);
    SpawnResult localResult = createLocalSpawnResult(100);
    SpawnResult remoteResult = createRemoteSpawnResult(200);
    Spawn spawn =
        new SpawnBuilder().withOwnerPrimaryOutput(createOutputArtifact("foo/out")).build();

    module.spawnExecuted(new SpawnExecutedEvent(spawn, localResult, Instant.EPOCH));
    module.spawnExecuted(new SpawnExecutedEvent(spawn, remoteResult, Instant.ofEpochMilli(100)));
    module.buildComplete(new BuildCompleteEvent(new BuildResult(1000)));

    ImmutableList<ExecutionGraph.Node> nodes = parse(buffer);
    assertThat(nodes)
        .containsExactly(
            executionGraphNodeBuilderForSpawnBuilderSpawn()
                .setIndex(0)
                .setMetrics(
                    ExecutionGraph.Metrics.newBuilder()
                        .setStartTimestampMillis(0)
                        .setDurationMillis(100)
                        .setOtherMillis(100))
                .setRunner("local")
                .build(),
            executionGraphNodeBuilderForSpawnBuilderSpawn()
                .setIndex(1)
                .setMetrics(
                    ExecutionGraph.Metrics.newBuilder()
                        .setStartTimestampMillis(100)
                        .setDurationMillis(200)
                        .setOtherMillis(200))
                .setRunner("remote")
                .setRetryOf(0)
                .build())
        .inOrder();
  }

  enum LocalLockFreeOutput {
    LOCAL_LOCK_FREE_OUTPUT_ENABLED(/*optionValue=*/ true) {
      @Override
      void assertBugReport(BugReporter bugReporter) {
        verify(bugReporter, never()).sendNonFatalBugReport(any());
      }
    },
    LOCAL_LOCK_FREE_OUTPUT_DISABLED(/*optionValue=*/ false) {
      @Override
      void assertBugReport(BugReporter bugReporter) {
        var captor = ArgumentCaptor.forClass(Exception.class);
        verify(bugReporter).sendNonFatalBugReport(captor.capture());
        assertThat(captor.getValue())
            .hasMessageThat()
            .contains("Multiple spawns produced 'output/foo/out' with overlapping execution time.");
      }
    };

    LocalLockFreeOutput(boolean optionValue) {
      this.optionValue = optionValue;
    }

    private final boolean optionValue;

    abstract void assertBugReport(BugReporter bugReporter);
  }

  @Test
  public void multipleSpawnsWithSameOutput_overlapping_recordsBothSpawnsWithoutRetry(
      @TestParameter LocalLockFreeOutput localLockFreeOutput) throws Exception {
    var buffer = new ByteArrayOutputStream();
    BugReporter bugReporter = mock(BugReporter.class);
    startLogging(
        eventBus,
        bugReporter,
        localLockFreeOutput.optionValue,
        /* logFileWriteEdges= */ false,
        UUID.randomUUID(),
        buffer,
        DependencyInfo.ALL);
    SpawnResult localResult = createLocalSpawnResult(100);
    SpawnResult remoteResult = createRemoteSpawnResult(200);
    Spawn spawn =
        new SpawnBuilder().withOwnerPrimaryOutput(createOutputArtifact("foo/out")).build();

    module.spawnExecuted(new SpawnExecutedEvent(spawn, localResult, Instant.EPOCH));
    module.spawnExecuted(new SpawnExecutedEvent(spawn, remoteResult, Instant.ofEpochMilli(10)));
    module.buildComplete(new BuildCompleteEvent(new BuildResult(1000)));

    ImmutableList<ExecutionGraph.Node> nodes = parse(buffer);
    assertThat(nodes)
        .containsExactly(
            executionGraphNodeBuilderForSpawnBuilderSpawn()
                .setIndex(0)
                .setMetrics(
                    ExecutionGraph.Metrics.newBuilder()
                        .setStartTimestampMillis(0)
                        .setDurationMillis(100)
                        .setOtherMillis(100))
                .setRunner("local")
                .build(),
            executionGraphNodeBuilderForSpawnBuilderSpawn()
                .setIndex(1)
                .setMetrics(
                    ExecutionGraph.Metrics.newBuilder()
                        .setStartTimestampMillis(10)
                        .setDurationMillis(200)
                        .setOtherMillis(200))
                .setRunner("remote")
                .build())
        .inOrder();
    localLockFreeOutput.assertBugReport(bugReporter);
  }

  @Test
  public void multipleSpawnsWithSameOutput_overlapping_ignoresSecondSpawnForDependencies()
      throws Exception {
    var buffer = new ByteArrayOutputStream();
    startLogging(
        eventBus,
        BugReporter.defaultInstance(),
        /* localLockFreeOutputEnabled= */ true,
        /* logFileWriteEdges= */ false,
        UUID.randomUUID(),
        buffer,
        DependencyInfo.ALL);
    SpawnResult localResult = createLocalSpawnResult(100);
    SpawnResult remoteResult = createRemoteSpawnResult(200);
    Artifact input = createOutputArtifact("foo/input");
    Spawn spawn = new SpawnBuilder().withOwnerPrimaryOutput(input).build();
    Spawn dependentSpawn =
        new SpawnBuilder()
            .withOwnerPrimaryOutput(createOutputArtifact("foo/output"))
            .withInput(input)
            .build();
    SpawnResult dependentResult = createRemoteSpawnResult(300);

    module.spawnExecuted(new SpawnExecutedEvent(spawn, localResult, Instant.EPOCH));
    module.spawnExecuted(new SpawnExecutedEvent(spawn, remoteResult, Instant.ofEpochMilli(10)));
    module.spawnExecuted(
        new SpawnExecutedEvent(dependentSpawn, dependentResult, Instant.ofEpochMilli(300)));
    module.buildComplete(new BuildCompleteEvent(new BuildResult(1000)));

    ImmutableList<ExecutionGraph.Node> nodes = parse(buffer);
    assertThat(nodes)
        .containsExactly(
            executionGraphNodeBuilderForSpawnBuilderSpawn()
                .setIndex(0)
                .setMetrics(
                    ExecutionGraph.Metrics.newBuilder()
                        .setStartTimestampMillis(0)
                        .setDurationMillis(100)
                        .setOtherMillis(100))
                .setRunner("local")
                .build(),
            executionGraphNodeBuilderForSpawnBuilderSpawn()
                .setIndex(1)
                .setMetrics(
                    ExecutionGraph.Metrics.newBuilder()
                        .setStartTimestampMillis(10)
                        .setDurationMillis(200)
                        .setOtherMillis(200))
                .setRunner("remote")
                .build(),
            executionGraphNodeBuilderForSpawnBuilderSpawn()
                .setIndex(2)
                .setMetrics(
                    ExecutionGraph.Metrics.newBuilder()
                        .setStartTimestampMillis(300)
                        .setDurationMillis(300)
                        .setOtherMillis(300))
                .setRunner("remote")
                .addDependentIndex(0)
                .build())
        .inOrder();
  }

  private class FakeOwnerWithPrimaryOutput extends FakeOwner {

    private final String primaryOutput;

    public FakeOwnerWithPrimaryOutput(
        String mnemonic, String progressMessage, String ownerLabel, String primaryOutput) {
      super(mnemonic, progressMessage, ownerLabel);
      this.primaryOutput = primaryOutput;
    }

    @Override
    public Artifact getPrimaryOutput() {
      return ActionsTestUtil.createArtifactWithExecPath(
          artifactRoot, PathFragment.create(primaryOutput));
    }
  }

  private Artifact createOutputArtifact(String rootRelativePath) {
    return ActionsTestUtil.createArtifactWithExecPath(
        artifactRoot, artifactRoot.getExecPath().getRelative(rootRelativePath));
  }

  private SpawnResult createLocalSpawnResult(int totalTimeInMs) {
    return new SpawnResult.Builder()
        .setRunnerName("local")
        .setStatus(Status.SUCCESS)
        .setExitCode(0)
        .setSpawnMetrics(
            SpawnMetrics.Builder.forLocalExec().setTotalTimeInMs(totalTimeInMs).build())
        .build();
  }

  private SpawnResult createRemoteSpawnResult(int totalTimeInMs) {
    return new SpawnResult.Builder()
        .setRunnerName("remote")
        .setStatus(Status.SUCCESS)
        .setExitCode(0)
        .setSpawnMetrics(
            SpawnMetrics.Builder.forRemoteExec().setTotalTimeInMs(totalTimeInMs).build())
        .build();
  }

  /**
   * Creates a {@link ExecutionGraph.Node.Builder} with pre-populated defaults for spawns created
   * using {@link SpawnBuilder}.
   */
  private ExecutionGraph.Node.Builder executionGraphNodeBuilderForSpawnBuilderSpawn() {
    return ExecutionGraph.Node.newBuilder()
        .setDescription("action 'progress message'")
        .setTargetLabel("//dummy:label")
        .setMnemonic("Mnemonic")
        // This comes from SpawnResult.Builder, which defaults to an empty string.
        .setRunnerSubtype("");
  }

  /**
   * Creates a {@link ExecutionGraph.Node.Builder} with pre-populated defaults for action events.
   */
  private ExecutionGraph.Node.Builder executionGraphNodeBuilderForAction(Action action) {
    return ExecutionGraph.Node.newBuilder()
        .setDescription(action.prettyPrint())
        .setTargetLabel(action.getOwner().getLabel().toString())
        .setMnemonic(action.getMnemonic());
  }
}
