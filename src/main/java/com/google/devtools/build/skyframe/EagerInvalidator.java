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

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.DeletingNodeVisitor;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.DirtyingNodeVisitor;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.InvalidationState;

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
        createVisitor(/*delete=*/true, graph, diff, invalidationReceiver, state, traverseGraph,
            dirtyKeyTracker);
    if (visitor != null) {
      visitor.run();
    }
  }

  /**
   * Creates an invalidation visitor that is ready to run. Caller should call #run() on the visitor.
   * Allows test classes to keep a reference to the visitor, and await exceptions/interrupts.
   */
  @VisibleForTesting
  static InvalidatingNodeVisitor createVisitor(boolean delete, DirtiableGraph graph,
      Iterable<SkyKey> diff, EvaluationProgressReceiver invalidationReceiver,
      InvalidationState state, boolean traverseGraph, DirtyKeyTracker dirtyKeyTracker) {
    state.update(diff);
    if (state.isEmpty()) {
      return null;
    }
    return delete
        ? new DeletingNodeVisitor(graph, invalidationReceiver, state, traverseGraph,
          dirtyKeyTracker)
        : new DirtyingNodeVisitor(graph, invalidationReceiver, state, dirtyKeyTracker);
  }

  /**
   * Invalidates given values and their upward transitive closure in the graph.
   */
  public static void invalidate(DirtiableGraph graph, Iterable<SkyKey> diff,
      EvaluationProgressReceiver invalidationReceiver, InvalidationState state,
      DirtyKeyTracker dirtyKeyTracker)
          throws InterruptedException {
    // If we are invalidating, we must be in an incremental build by definition, so we must
    // maintain a consistent graph state by traversing the graph and invalidating transitive
    // dependencies. If edges aren't present, it would be impossible to check the dependencies of
    // a dirty node in any case.
    InvalidatingNodeVisitor visitor =
        createVisitor(/*delete=*/false, graph, diff, invalidationReceiver, state,
            /*traverseGraph=*/true, dirtyKeyTracker);
    if (visitor != null) {
      visitor.run();
    }
  }
}
