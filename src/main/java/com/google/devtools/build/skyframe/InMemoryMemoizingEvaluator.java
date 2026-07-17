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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.util.TestType;
import java.util.Map;

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
        new DirtyAndInflightTrackingProgressReceiver(progressReceiver),
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
