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

import static com.google.common.base.Preconditions.checkState;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.util.TestType;
import com.google.devtools.build.skyframe.NodeEntry.DirtyType;
import com.google.devtools.build.skyframe.SkyframeGraphStatsEvent.EvaluationStats;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;

/**
 * An in-memory {@link MemoizingEvaluator} that uses the eager invalidation strategy. This class is,
 * by itself, not thread-safe. Neither is it thread-safe to use this class in parallel with any of
 * the returned graphs. However, it is allowed to access the graph from multiple threads as long as
 * that does not happen in parallel with an {@link #evaluate} call.
 *
 * <p>This memoizing evaluator uses a monotonically increasing {@link IntVersion} for incremental
 * evaluations and {@link Version#constant} for non-incremental evaluations.
 */
public final class InMemoryMemoizingEvaluator extends AbstractInMemoryMemoizingEvaluator {

  /**
   * A progress receiver that, in addition to tracking dirty and inflight notes, also collects
   * stats.
   */
  private static final class ProgressReceiver extends DirtyAndInflightTrackingProgressReceiver {
    // Nodes that were dirtied because one of their transitive dependencies changed
    private final ConcurrentHashMultiset<SkyFunctionName> dirtied;

    // Nodes that were dirtied because they themselves changed (for example, a leaf node that
    // represents a file and that changed between builds)
    private final ConcurrentHashMultiset<SkyFunctionName> changed;

    // Nodes that were built and found different from the previous version
    private final ConcurrentHashMultiset<SkyFunctionName> built;

    // Nodes that were built and found to be same as the previous version
    private final ConcurrentHashMultiset<SkyFunctionName> cleaned;

    // Nodes that were computed during the build
    private final ConcurrentHashMultiset<SkyFunctionName> evaluated;

    private static ConcurrentHashMultiset<SkyFunctionName> createMultiset() {
      return ConcurrentHashMultiset.create(
          new ConcurrentHashMap<>(Runtime.getRuntime().availableProcessors(), 0.75f));
    }

    private ProgressReceiver(EvaluationProgressReceiver progressReceiver) {
      super(progressReceiver);

      dirtied = createMultiset();
      changed = createMultiset();
      built = createMultiset();
      cleaned = createMultiset();
      evaluated = createMultiset();
    }

    private static ImmutableMap<SkyFunctionName, Integer> fromMultiset(
        ConcurrentHashMultiset<SkyFunctionName> s) {
      return s.entrySet().stream()
          .collect(ImmutableMap.toImmutableMap(e -> e.getElement(), e -> e.getCount()));
    }

    private EvaluationStats aggregateAndReset() {
      EvaluationStats result =
          new EvaluationStats(
              fromMultiset(dirtied),
              fromMultiset(changed),
              fromMultiset(built),
              fromMultiset(cleaned),
              fromMultiset(evaluated));
      dirtied.clear();
      changed.clear();
      built.clear();
      cleaned.clear();
      evaluated.clear();
      return result;
    }

    @Override
    public void dirtied(SkyKey skyKey, DirtyType dirtyType) {
      super.dirtied(skyKey, dirtyType);

      switch (dirtyType) {
        case DIRTY -> dirtied.add(skyKey.functionName());
        case CHANGE -> changed.add(skyKey.functionName());
        case REWIND -> {} // Should not happen but let's not crash the server due to logging
      }
    }

    @Override
    public void evaluated(
        SkyKey skyKey,
        EvaluationState state,
        @Nullable SkyValue newValue,
        @Nullable ErrorInfo newError,
        @Nullable GroupedDeps directDeps) {
      super.evaluated(skyKey, state, newValue, newError, directDeps);

      if (directDeps == null) {
        // In this case, no actual evaluation work was done so let's not record it.
      } else if (state.versionChanged()) {
        built.add(skyKey.functionName(), 1);
      } else {
        cleaned.add(skyKey.functionName(), 1);
      }
    }

    @Override
    public void stateEnding(SkyKey skyKey, NodeState nodeState) {
      super.stateEnding(skyKey, nodeState);
      if (nodeState == NodeState.COMPUTE) {
        evaluated.add(skyKey.functionName());
      }
    }
  }

  // Not final only for testing.
  private InMemoryGraph graph;

  public InMemoryMemoizingEvaluator(
      Map<SkyFunctionName, SkyFunction> skyFunctions, Differencer differencer) {
    this(skyFunctions, differencer, EvaluationProgressReceiver.NULL);
  }

  public InMemoryMemoizingEvaluator(
      Map<SkyFunctionName, SkyFunction> skyFunctions,
      Differencer differencer,
      EvaluationProgressReceiver progressReceiver) {
    this(
        skyFunctions,
        differencer,
        progressReceiver,
        GraphInconsistencyReceiver.THROWING,
        EventFilter.FULL_STORAGE,
        new EmittedEventState(),
        /* keepEdges= */ true,
        /* usePooledInterning= */ true);
  }

  public InMemoryMemoizingEvaluator(
      Map<SkyFunctionName, SkyFunction> skyFunctions,
      Differencer differencer,
      EvaluationProgressReceiver progressReceiver,
      GraphInconsistencyReceiver graphInconsistencyReceiver,
      EventFilter eventFilter,
      EmittedEventState emittedEventState,
      boolean keepEdges,
      boolean usePooledInterning) {
    super(
        ImmutableMap.copyOf(skyFunctions),
        differencer,
        new ProgressReceiver(progressReceiver),
        eventFilter,
        emittedEventState,
        graphInconsistencyReceiver,
        keepEdges,
        Version.minimal());
    this.graph =
        keepEdges
            ? InMemoryGraph.create(usePooledInterning)
            : InMemoryGraph.createEdgeless(usePooledInterning);
  }

  @Override
  public void postLoggingStats(ExtendedEventHandler eventHandler) {
    EvaluationStats evaluationStats = ((ProgressReceiver) progressReceiver).aggregateAndReset();
    eventHandler.post(new SkyframeGraphStatsEvent(graph.valuesSize(), evaluationStats));
  }

  @Override
  public void injectGraphTransformerForTesting(GraphTransformerForTesting transformer) {
    checkState(TestType.isInTest());
    this.graph = transformer.transform(this.graph);
  }

  @Override
  public InMemoryGraph getInMemoryGraph() {
    return graph;
  }

  @VisibleForTesting
  public ImmutableMap<SkyFunctionName, SkyFunction> getSkyFunctionsForTesting() {
    return skyFunctions;
  }
}
