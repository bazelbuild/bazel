// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.collect.nestedset.NestedSetVisitor;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.profiler.GoogleAutoProfilerUtils;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.skyframe.Differencer.Diff;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.DeletingInvalidationState;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.DirtyingInvalidationState;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.InvalidationState;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import java.time.Duration;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/**
 * An in-memory {@link MemoizingEvaluator} that uses the eager invalidation strategy. This class is,
 * by itself, not thread-safe. Neither is it thread-safe to use this class in parallel with any of
 * the returned graphs. However, it is allowed to access the graph from multiple threads as long as
 * that does not happen in parallel with an {@link #evaluate} call.
 *
 * <p>This memoizing evaluator uses a monotonically increasing {@link IntVersion}.
 */
public final class InMemoryMemoizingEvaluator extends AbstractInMemoryMemoizingEvaluator {

  private final ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions;
  private final DirtyTrackingProgressReceiver progressReceiver;
  // Not final only for testing.
  private InMemoryGraph graph;
  private IntVersion lastGraphVersion = null;

  // State related to invalidation and deletion.
  private Set<SkyKey> valuesToDelete = new LinkedHashSet<>();
  private Set<SkyKey> valuesToDirty = new LinkedHashSet<>();
  private Map<SkyKey, SkyValue> valuesToInject = new HashMap<>();
  private final DeletingInvalidationState deleterState = new DeletingInvalidationState();
  private final Differencer differencer;
  private final GraphInconsistencyReceiver graphInconsistencyReceiver;
  private final EventFilter eventFilter;

  // Keep edges in graph. Can be false to save memory, in which case incremental builds are
  // not possible.
  private final boolean keepEdges;

  // Values that the caller explicitly specified are assumed to be changed -- they will be
  // re-evaluated even if none of their children are changed.
  private final InvalidationState invalidatorState = new DirtyingInvalidationState();

  private final NestedSetVisitor.VisitedState emittedEventState;

  private final AtomicBoolean evaluating = new AtomicBoolean(false);

  public InMemoryMemoizingEvaluator(
      Map<SkyFunctionName, SkyFunction> skyFunctions, Differencer differencer) {
    this(skyFunctions, differencer, /*progressReceiver=*/ null);
  }

  public InMemoryMemoizingEvaluator(
      Map<SkyFunctionName, SkyFunction> skyFunctions,
      Differencer differencer,
      @Nullable EvaluationProgressReceiver progressReceiver) {
    this(
        skyFunctions,
        differencer,
        progressReceiver,
        GraphInconsistencyReceiver.THROWING,
        EventFilter.FULL_STORAGE,
        new NestedSetVisitor.VisitedState(),
        /* keepEdges= */ true,
        /* usePooledSkyKeyInterning= */ true);
  }

  public InMemoryMemoizingEvaluator(
      Map<SkyFunctionName, SkyFunction> skyFunctions,
      Differencer differencer,
      @Nullable EvaluationProgressReceiver progressReceiver,
      GraphInconsistencyReceiver graphInconsistencyReceiver,
      EventFilter eventFilter,
      NestedSetVisitor.VisitedState emittedEventState,
      boolean keepEdges,
      boolean usePooledSkyKeyInterning) {
    this.skyFunctions = ImmutableMap.copyOf(skyFunctions);
    this.differencer = Preconditions.checkNotNull(differencer);
    this.progressReceiver = new DirtyTrackingProgressReceiver(progressReceiver);
    this.graphInconsistencyReceiver = Preconditions.checkNotNull(graphInconsistencyReceiver);
    this.eventFilter = eventFilter;
    this.graph =
        keepEdges
            ? InMemoryGraph.create(usePooledSkyKeyInterning)
            : InMemoryGraph.createEdgeless(usePooledSkyKeyInterning);
    this.emittedEventState = emittedEventState;
    this.keepEdges = keepEdges;
  }

  private void invalidate(Iterable<SkyKey> diff) {
    Iterables.addAll(valuesToDirty, diff);
  }

  private static final Duration MIN_TIME_TO_LOG_DELETION = Duration.ofMillis(10);

  @Override
  public void delete(Predicate<SkyKey> deletePredicate) {
    try (AutoProfiler ignored =
        GoogleAutoProfilerUtils.logged("deletion marking", MIN_TIME_TO_LOG_DELETION)) {
      Set<SkyKey> toDelete = Sets.newConcurrentHashSet();
      graph.parallelForEach(
          e -> {
            if (e.isDirty() || deletePredicate.test(e.getKey())) {
              toDelete.add(e.getKey());
            }
          });
      valuesToDelete.addAll(toDelete);
    }
  }

  @Override
  public void deleteDirty(long versionAgeLimit) {
    Preconditions.checkArgument(versionAgeLimit >= 0, versionAgeLimit);
    Version threshold = IntVersion.of(lastGraphVersion.getVal() - versionAgeLimit);
    valuesToDelete.addAll(
        Sets.filter(
            progressReceiver.getUnenqueuedDirtyKeys(),
            skyKey -> {
              NodeEntry entry = graph.get(null, Reason.OTHER, skyKey);
              Preconditions.checkNotNull(entry, skyKey);
              Preconditions.checkState(entry.isDirty(), skyKey);
              return entry.getVersion().atMost(threshold);
            }));
  }

  @Override
  public <T extends SkyValue> EvaluationResult<T> evaluate(
      Iterable<? extends SkyKey> roots, EvaluationContext evaluationContext)
      throws InterruptedException {
    // NOTE: Performance critical code. See bug "Null build performance parity".
    IntVersion graphVersion = lastGraphVersion == null ? IntVersion.of(0) : lastGraphVersion.next();
    setAndCheckEvaluateState(true, roots);
    try {
      // Mark for removal any inflight nodes from the previous evaluation.
      valuesToDelete.addAll(progressReceiver.getAndClearInflightKeys());

      // The RecordingDifferencer implementation is not quite working as it should be at this point.
      // It clears the internal data structures after getDiff is called and will not return
      // diffs for historical versions. This makes the following code sensitive to interrupts.
      // Ideally we would simply not update lastGraphVersion if an interrupt occurs.
      Diff diff =
          differencer.getDiff(new DelegatingWalkableGraph(graph), lastGraphVersion, graphVersion);
      if (!diff.isEmpty() || !valuesToInject.isEmpty() || !valuesToDelete.isEmpty()) {
        valuesToInject.putAll(diff.changedKeysWithNewValues());
        invalidate(diff.changedKeysWithoutNewValues());
        pruneInjectedValues(valuesToInject);
        invalidate(valuesToInject.keySet());

        performInvalidation();
        injectValues(graphVersion);
      }

      EvaluationResult<T> result;
      try (SilentCloseable c = Profiler.instance().profile("ParallelEvaluator.eval")) {
        ParallelEvaluator evaluator =
            new ParallelEvaluator(
                graph,
                graphVersion,
                Version.minimal(),
                skyFunctions,
                evaluationContext.getEventHandler(),
                emittedEventState,
                eventFilter,
                ErrorInfoManager.UseChildErrorInfoIfNecessary.INSTANCE,
                evaluationContext.getKeepGoing(),
                progressReceiver,
                graphInconsistencyReceiver,
                evaluationContext
                    .getExecutor()
                    .orElseGet(
                        () ->
                            AbstractQueueVisitor.create(
                                "skyframe-evaluator",
                                evaluationContext.getParallelism(),
                                ParallelEvaluatorErrorClassifier.instance())),
                new SimpleCycleDetector(),
                evaluationContext.mergingSkyframeAnalysisExecutionPhases(),
                evaluationContext.getUnnecessaryTemporaryStateDropperReceiver());
        result = evaluator.eval(roots);
      }
      return EvaluationResult.<T>builder()
          .mergeFrom(result)
          .setWalkableGraph(new DelegatingWalkableGraph(graph))
          .build();
    } finally {
      lastGraphVersion = graphVersion;
      setAndCheckEvaluateState(false, roots);
    }
  }

  /**
   * Removes entries in {@code valuesToInject} whose values are equal to the present values in the
   * graph.
   */
  private void pruneInjectedValues(Map<SkyKey, SkyValue> valuesToInject) {
    for (Iterator<Map.Entry<SkyKey, SkyValue>> it = valuesToInject.entrySet().iterator();
        it.hasNext(); ) {
      Map.Entry<SkyKey, SkyValue> entry = it.next();
      SkyKey key = entry.getKey();
      SkyValue newValue = entry.getValue();
      NodeEntry prevEntry = graph.get(null, Reason.OTHER, key);
      if (prevEntry != null && prevEntry.isDone()) {
        if (keepEdges) {
          try {
            if (!prevEntry.hasAtLeastOneDep()) {
              if (newValue.equals(prevEntry.getValue())
                  && !valuesToDirty.contains(key)
                  && !valuesToDelete.contains(key)) {
                it.remove();
              }
            } else {
              // Rare situation of an injected dep that depends on another node. Usually the dep is
              // the error transience node. When working with external repositories, it can also be
              // an external workspace file. Don't bother injecting it, just invalidate it.
              // We'll wastefully evaluate the node freshly during evaluation, but this happens very
              // rarely.
              valuesToDirty.add(key);
              it.remove();
            }
          } catch (InterruptedException e) {
            throw new IllegalStateException(
                "InMemoryGraph does not throw: " + entry + ", " + prevEntry, e);
          }
        } else {
          // No incrementality. Just delete the old value from the graph. The new value is about to
          // be injected.
          graph.remove(key);
        }
      }
    }
  }

  /**
   * Injects values in {@code valuesToInject} into the graph.
   */
  private void injectValues(IntVersion version) {
    if (valuesToInject.isEmpty()) {
      return;
    }
    try {
      ParallelEvaluator.injectValues(valuesToInject, version, graph, progressReceiver);
    } catch (InterruptedException e) {
      throw new IllegalStateException("InMemoryGraph doesn't throw interrupts", e);
    }
    // Start with a new map to avoid bloat since clear() does not downsize the map.
    valuesToInject = new HashMap<>();
  }

  private void performInvalidation() throws InterruptedException {
    EagerInvalidator.delete(graph, valuesToDelete, progressReceiver, deleterState, keepEdges);
    // Note that clearing the valuesToDelete would not do an internal resizing. Therefore, if any
    // build has a large set of dirty values, subsequent operations (even clearing) will be slower.
    // Instead, just start afresh with a new LinkedHashSet.
    valuesToDelete = new LinkedHashSet<>();

    EagerInvalidator.invalidate(graph, valuesToDirty, progressReceiver, invalidatorState);
    // Ditto.
    valuesToDirty = new LinkedHashSet<>();
  }

  private void setAndCheckEvaluateState(boolean newValue, Object requestInfo) {
    Preconditions.checkState(evaluating.getAndSet(newValue) != newValue,
        "Re-entrant evaluation for request: %s", requestInfo);
  }

  @Override
  public void postLoggingStats(ExtendedEventHandler eventHandler) {
    eventHandler.post(new SkyframeGraphStatsEvent(graph.valuesSize()));
  }

  @Override
  public void injectGraphTransformerForTesting(GraphTransformerForTesting transformer) {
    this.graph = transformer.transform(this.graph);
  }

  @Override
  public InMemoryGraph getInMemoryGraph() {
    return graph;
  }

  public ImmutableMap<SkyFunctionName, SkyFunction> getSkyFunctionsForTesting() {
    return skyFunctions;
  }
}
