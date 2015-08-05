// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.base.Function;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor.ThreadPoolExecutorParams;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.DeletingNodeVisitor;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.DirtyingNodeVisitor;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.InvalidationState;

import java.util.concurrent.ThreadPoolExecutor;

import javax.annotation.Nullable;

/**
 * Utility class for performing eager invalidation on Skyframe graphs.
 *
 * <p>This is intended only for use in alternative {@code MemoizingEvaluator} implementations.
 */
public final class EagerInvalidator {

  private EagerInvalidator() {}

  /**
   * Deletes given values. The {@code traverseGraph} parameter controls whether this method deletes
   * (transitive) dependents of these nodes and relevant graph edges, or just the nodes themselves.
   * Deleting just the nodes is inconsistent unless the graph will not be used for incremental
   * builds in the future, but unfortunately there is a case where we delete nodes intra-build. As
   * long as the full upward transitive closure of the nodes is specified for deletion, the graph
   * remains consistent.
   */
  public static void delete(DirtiableGraph graph, Iterable<SkyKey> diff,
      EvaluationProgressReceiver invalidationReceiver, InvalidationState state,
      boolean traverseGraph, DirtyKeyTracker dirtyKeyTracker) throws InterruptedException {
    InvalidatingNodeVisitor visitor =
        createDeletingVisitorIfNeeded(graph, diff, invalidationReceiver, state, traverseGraph,
            dirtyKeyTracker);
    if (visitor != null) {
      visitor.run();
    }
  }

  @Nullable
  static InvalidatingNodeVisitor createDeletingVisitorIfNeeded(DirtiableGraph graph,
      Iterable<SkyKey> diff, EvaluationProgressReceiver invalidationReceiver,
      InvalidationState state, boolean traverseGraph, DirtyKeyTracker dirtyKeyTracker) {
    state.update(diff);
    return state.isEmpty() ? null
        : new DeletingNodeVisitor(graph, invalidationReceiver, state, traverseGraph,
            dirtyKeyTracker);
  }

  @Nullable
  static InvalidatingNodeVisitor createInvalidatingVisitorIfNeeded(DirtiableGraph graph,
      Iterable<SkyKey> diff, EvaluationProgressReceiver invalidationReceiver,
      InvalidationState state, DirtyKeyTracker dirtyKeyTracker,
      Function<ThreadPoolExecutorParams, ThreadPoolExecutor> executorFactory) {
    state.update(diff);
    return state.isEmpty() ? null
        : new DirtyingNodeVisitor(graph, invalidationReceiver, state, dirtyKeyTracker,
            executorFactory);
  }

  @Nullable
  static InvalidatingNodeVisitor createInvalidatingVisitorIfNeeded(DirtiableGraph graph,
      Iterable<SkyKey> diff, EvaluationProgressReceiver invalidationReceiver,
      InvalidationState state, DirtyKeyTracker dirtyKeyTracker) {
    return createInvalidatingVisitorIfNeeded(graph, diff, invalidationReceiver, state,
        dirtyKeyTracker, AbstractQueueVisitor.EXECUTOR_FACTORY);
  }

  /**
   * Invalidates given values and their upward transitive closure in the graph, using an executor
   * constructed with the provided factory, if necessary.
   */
  public static void invalidate(DirtiableGraph graph, Iterable<SkyKey> diff,
      EvaluationProgressReceiver invalidationReceiver, InvalidationState state,
      DirtyKeyTracker dirtyKeyTracker,
      Function<ThreadPoolExecutorParams, ThreadPoolExecutor> executorFactory)
          throws InterruptedException {
    // If we are invalidating, we must be in an incremental build by definition, so we must
    // maintain a consistent graph state by traversing the graph and invalidating transitive
    // dependencies. If edges aren't present, it would be impossible to check the dependencies of
    // a dirty node in any case.
    InvalidatingNodeVisitor visitor =
        createInvalidatingVisitorIfNeeded(graph, diff, invalidationReceiver, state,
            dirtyKeyTracker, executorFactory);
    if (visitor != null) {
      visitor.run();
    }
  }

  /**
   * Invalidates given values and their upward transitive closure in the graph.
   */
  public static void invalidate(DirtiableGraph graph, Iterable<SkyKey> diff,
      EvaluationProgressReceiver invalidationReceiver, InvalidationState state,
      DirtyKeyTracker dirtyKeyTracker)
      throws InterruptedException {
    invalidate(graph, diff, invalidationReceiver, state, dirtyKeyTracker,
        AbstractQueueVisitor.EXECUTOR_FACTORY);
  }

}
