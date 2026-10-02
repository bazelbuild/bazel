// Copyright 2026 The Bazel Authors. All rights reserved.
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

import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.SpawnExecutedEvent;
import com.google.devtools.build.lib.actions.SpawnMetrics;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.buildtool.BuildResult.BuildToolLogCollection;
import com.google.devtools.build.lib.exec.util.FakeActionInputFileCache;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.runtime.ExecutionGraphModule.ActionDumpWriter;
import com.google.devtools.build.lib.runtime.ExecutionGraphModule.DependencyInfo;
import com.google.devtools.build.lib.testutil.TestFileOutErr;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.sun.management.ThreadMXBean;
import java.io.OutputStream;
import java.lang.management.ManagementFactory;
import java.time.Instant;
import java.util.Arrays;

/**
 * Measures the per-spawn wall time and allocation of {@link ActionDumpWriter#enqueue(
 * SpawnExecutedEvent)}, to quantify the cost of deferring {@code outputToNode} registration until
 * after {@code enqueueBytes()}
 * 
 * <pre>{@code
 * bazel build //src/test/java/com/google/devtools/build/lib/runtime:ExecutionGraphModuleBenchmark
 * for i in $(seq 5); do
 *   ./bazel-bin/src/test/java/com/google/devtools/build/lib/runtime/ExecutionGraphModuleBenchmark
 * done
 * }</pre>
 */
public final class ExecutionGraphModuleBenchmark {

  /** Spawns enqueued per trial. */
  private static final int SPAWNS_PER_TRIAL = 50_000;

  /** Inputs per spawn, drawn from previously generated outputs so dep edges get resolved. */
  private static final int INPUTS_PER_SPAWN = 5;

  private static final int TRIALS = 25;

  /**
   * Trials discarded as JIT warmup. Needs to be generous: whether the JIT has settled shifts the
   * measured allocation by tens of bytes per spawn and then persists for the rest of the JVM's
   * life, which otherwise shows up as large variance between invocations rather than within one.
   */
  private static final int WARMUP_TRIALS = 15;

  private static final ThreadMXBean THREAD_MX = (ThreadMXBean) ManagementFactory.getThreadMXBean();

  private final ArtifactRoot artifactRoot;
  private final SpawnResult spawnResult;

  private ExecutionGraphModuleBenchmark() {
    Path execRoot =
        new InMemoryFileSystem(DigestHashFunction.SHA256).getPath("/benchmark").getRelative("..");
    this.artifactRoot = ArtifactRoot.asDerivedRoot(execRoot, RootType.OUTPUT, "output");
    this.spawnResult =
        new SpawnResult.Builder()
            .setRunnerName("remote")
            .setStatus(Status.SUCCESS)
            .setExitCode(0)
            .setSpawnMetrics(SpawnMetrics.Builder.forRemoteExec().setTotalTimeInMs(100).build())
            .build();
  }

  public static void main(String[] args) throws Exception {
    new ExecutionGraphModuleBenchmark().run();
  }

  private void run() throws Exception {
    long[] nanosPerSpawn = new long[TRIALS - WARMUP_TRIALS];
    long[] bytesPerSpawn = new long[TRIALS - WARMUP_TRIALS];

    for (int trial = 0; trial < TRIALS; trial++) {
      // Fresh artifact names per trial so that each writer's outputToNode sees only new outputs,
      // keeping every spawn on the common code path rather than the retry/bug-report branches.
      SpawnExecutedEvent[] events = buildEvents(trial);
      ActionDumpWriter writer = newWriter();

      long nanos0 = System.nanoTime();
      long alloc0 = allocatedBytes();
      for (SpawnExecutedEvent event : events) {
        writer.enqueue(event);
      }
      long alloc = allocatedBytes() - alloc0;
      long nanos = System.nanoTime() - nanos0;

      writer.shutdown(/* logs= */ null);

      String marker;
      if (trial < WARMUP_TRIALS) {
        marker = " (warmup, discarded)";
      } else {
        marker = "";
        nanosPerSpawn[trial - WARMUP_TRIALS] = nanos / SPAWNS_PER_TRIAL;
        bytesPerSpawn[trial - WARMUP_TRIALS] = alloc / SPAWNS_PER_TRIAL;
      }
      System.out.printf(
          "trial %2d: %6d ns/spawn  %6d bytes/spawn%s%n",
          trial, nanos / SPAWNS_PER_TRIAL, alloc / SPAWNS_PER_TRIAL, marker);
    }

    Arrays.sort(nanosPerSpawn);
    Arrays.sort(bytesPerSpawn);
    int n = bytesPerSpawn.length;
    System.out.printf(
        "%nover %d measured trials of %d spawns:%n"
            + "  %6d ns/spawn    (min %d, max %d)%n"
            + "  %6d bytes/spawn (min %d, max %d)%n",
        n,
        SPAWNS_PER_TRIAL,
        nanosPerSpawn[n / 2],
        nanosPerSpawn[0],
        nanosPerSpawn[n - 1],
        bytesPerSpawn[n / 2],
        bytesPerSpawn[0],
        bytesPerSpawn[n - 1]);
  }

  private SpawnExecutedEvent[] buildEvents(int trial) {
    SpawnExecutedEvent[] events = new SpawnExecutedEvent[SPAWNS_PER_TRIAL];
    Artifact[] outputs = new Artifact[SPAWNS_PER_TRIAL];
    for (int i = 0; i < SPAWNS_PER_TRIAL; i++) {
      outputs[i] = createOutputArtifact("t" + trial + "/out" + i);
    }

    for (int i = 0; i < SPAWNS_PER_TRIAL; i++) {
      SpawnBuilder spawnBuilder =
          new SpawnBuilder().withOwnerPrimaryOutput(outputs[i]).withOutput(outputs[i]);
      // Depend on the immediately preceding outputs, so most dep lookups hit outputToNode.
      for (int j = 1; j <= INPUTS_PER_SPAWN && i - j >= 0; j++) {
        spawnBuilder.withInput((ActionInput) outputs[i - j]);
      }
      events[i] =
          new SpawnExecutedEvent(
              spawnBuilder.build(),
              new FakeActionInputFileCache(),
              null,
              new TestFileOutErr(),
              spawnResult,
              Instant.ofEpochMilli(i),
              /* spawnIdentifier= */ "s" + i);
    }
    return events;
  }

  private Artifact createOutputArtifact(String rootRelativePath) {
    DerivedArtifact artifact =
        (DerivedArtifact)
            ActionsTestUtil.createArtifactWithExecPath(
                artifactRoot, artifactRoot.getExecPath().getRelative(rootRelativePath));
    artifact.setGeneratingActionKey(ActionsTestUtil.NULL_ACTION_LOOKUP_DATA);
    return artifact;
  }

  private static ActionDumpWriter newWriter() {
    return new ActionDumpWriter(
        BugReporter.defaultInstance(),
        new EventBus(),
        /* localLockFreeOutputEnabled= */ false,
        /* logFileWriteEdges= */ false,
        OutputStream.nullOutputStream(),
        DependencyInfo.ALL,
        /* queueSize= */ -1,
        /* queuedBytesLimit= */ -1) {
      @Override
      protected void updateLogs(BuildToolLogCollection logs) {}
    };
  }

  /**
   * Bytes allocated by the calling thread. Reading this on the enqueueing thread excludes the
   * writer thread's compression work, isolating the path under measurement.
   */
  private static long allocatedBytes() {
    return THREAD_MX.getThreadAllocatedBytes(Thread.currentThread().getId());
  }
}
