// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.skyframe;

import static com.google.devtools.build.skyframe.AbstractParallelEvaluator.maybeMarkRebuilding;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.concurrent.ForkJoinQuiescingExecutor;
import com.google.devtools.build.lib.concurrent.NamedForkJoinPool;
import com.google.devtools.build.lib.concurrent.QuiescingExecutor;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.skyframe.NodeEntry.LifecycleState;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * {@link CycleDetector} that does not try to actually identify cycles, instead just marking every
 * incomplete but reachable node entry as a cycle node. This is an optimization for consumers of
 * Skyframe evaluations that don't actually need to know what the precise cycles are.
 */
public class ShortCircuitingCycleDetector implements CycleDetector {
  private static final SkyFunctionName DUMMY_CYCLE_MARKER =
      SkyFunctionName.createHermetic("DUMMY_CYCLE_MARKER");

  @SerializationConstant public static final SkyKey DUMMY_CYCLE_KEY = () -> DUMMY_CYCLE_MARKER;

  @SerializationConstant
  public static final ErrorInfo CYCLE_ERROR_INFO =
      ErrorInfo.fromCycle(
          CycleInfo.createCycleInfo(ImmutableList.of(), ImmutableList.of(DUMMY_CYCLE_KEY)));

  private final Set<SkyKey> seenNodes;
  private final int numThreads;

  public ShortCircuitingCycleDetector(int numThreads) {
    this.numThreads = numThreads;
    seenNodes =
        Collections.newSetFromMap(
            new ConcurrentHashMap<>(/* initialCapacity= */ numThreads, /* loadFactor= */ 0.75f));
  }

  @Override
  public void checkForCycles(
      Iterable<SkyKey> badRoots,
      EvaluationResult.Builder<?> result,
      ParallelEvaluatorContext evaluatorContext)
      throws InterruptedException {
    Map<SkyKey, ? extends NodeEntry> roots =
        evaluatorContext
            .getGraph()
            .getBatchMap(null, QueryableGraph.Reason.CYCLE_CHECKING, badRoots);
    QuiescingExecutor quiescingExecutor =
        ForkJoinQuiescingExecutor.newBuilder()
            .withOwnershipOf(
                NamedForkJoinPool.newNamedPool("short-circuiting-cycle-detector", numThreads))
            .setErrorClassifier(ParallelEvaluatorErrorClassifier.instance())
            .build();
    seenNodes.addAll(roots.keySet());
    for (Map.Entry<SkyKey, ? extends NodeEntry> rootEntry : roots.entrySet()) {
      quiescingExecutor.execute(new MarkCycle(rootEntry, evaluatorContext, quiescingExecutor));
      result.addError(rootEntry.getKey(), CYCLE_ERROR_INFO);
    }
    try {
      quiescingExecutor.awaitQuiescence(/* interruptWorkers= */ true);
    } catch (SchedulerException e) {
      Preconditions.checkState(e.getCause() instanceof InterruptedException, e);
      throw (InterruptedException) e.getCause();
    }
  }

  private class MarkCycle implements Runnable {
    private final SkyKey key;
    private final NodeEntry entry;
    private final ParallelEvaluatorContext evaluatorContext;
    private final QuiescingExecutor executor;

    MarkCycle(
        Map.Entry<SkyKey, ? extends NodeEntry> mapEntry,
        ParallelEvaluatorContext evaluatorContext,
        QuiescingExecutor executor) {
      this.key = mapEntry.getKey();
      this.entry = mapEntry.getValue();
      this.evaluatorContext = evaluatorContext;
      this.executor = executor;
    }

    @Override
    public void run() {
      try {
        List<SkyKey> dirtyDeps = ImmutableList.of();
        while (entry.hasUnsignaledDeps()) {
          entry.signalDep(evaluatorContext.getGraphVersion(), null);
        }
        if (entry.isDirty() && entry.getLifecycleState() == LifecycleState.CHECK_DEPENDENCIES) {
          // The entry was checking dependencies, but had no dependencies outstanding (otherwise
          // the signaling loop above would have put it into the NEEDS_REBUILDING state). The only
          // reason it didn't get another chance to build must have been that we were in a
          // nokeep_going build. Tolerate that situation, even though it currently only occurs in
          // tests.
          // Tell entry we're about to check some of its deps so it lets us build it.
          dirtyDeps = entry.getNextDirtyDirectDeps();
          entry.addTemporaryDirectDepGroup(dirtyDeps);
          for (SkyKey dep : dirtyDeps) {
            entry.signalDep(evaluatorContext.getGraphVersion(), dep);
          }
          Preconditions.checkState(
              !entry.hasUnsignaledDeps(), "Entry has unsignaled deps: %s %s", entry, dirtyDeps);
          Preconditions.checkState(
              entry.getLifecycleState() == LifecycleState.NEEDS_REBUILDING,
              "Not NEEDS_REBUILDING: %s",
              entry);
        }
        Map<SkyKey, ? extends NodeEntry> deps =
            evaluatorContext
                .getGraph()
                .getBatchMap(
                    key,
                    QueryableGraph.Reason.CYCLE_CHECKING,
                    Iterables.concat(dirtyDeps, Iterables.concat(entry.getTemporaryDirectDeps())));
        for (Map.Entry<SkyKey, ? extends NodeEntry> depEntry : deps.entrySet()) {
          if (!depEntry.getValue().isDone() && seenNodes.add(depEntry.getKey())) {
            executor.execute(new MarkCycle(depEntry, evaluatorContext, executor));
          }
        }
        maybeMarkRebuilding(entry);
        SkyFunctionEnvironment env =
            SkyFunctionEnvironment.createForError(
                key,
                entry.getTemporaryDirectDeps(),
                /* bubbleErrorInfo= */ ImmutableMap.of(),
                entry.getAllRemainingDirtyDirectDeps(),
                evaluatorContext);

        env.setError(entry, CYCLE_ERROR_INFO);
        // We aren't committing cycles in graph order (in fact we visit parents before children), so
        // it's completely possible that one of our not-yet-visited children is not done because it
        // too transitively depends on a cycle.
        var unusedReverseDeps = env.commitAndGetParents(entry, /* expectDoneDeps= */ false);
      } catch (InterruptedException e) {
        throw SchedulerException.ofInterruption(e, key);
      }
    }
  }
}
