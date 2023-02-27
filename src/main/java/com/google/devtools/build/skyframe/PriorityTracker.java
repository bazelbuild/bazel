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
package com.google.devtools.build.skyframe;

/**
 * Tracks the priority of node evaluations.
 *
 * <p>Prioritization has two goals: decrease the total walltime and minimize the number of inflight
 * nodes. The build graph has uneven depth. When the build is less CPU-bound due to available
 * parallelism, long sequential dependencies should be prioritized first so that they do not become
 * long poles in walltime.
 *
 * <p>Minimizing inflight nodes is desirable because they contribute to memory pressure. A high
 * inflight node count increases work done by the garbage collector and triggers cache clearing.
 * Assigning higher priority to nodes with a higher likelihood of completing, without further
 * fanout, helps to avoid these causes of poor performance.
 *
 * <p>Prioritization uses two runtime signals.
 *
 * <ul>
 *   <li><i>depth</i>: the deeper a node, the more likely it is to be one that could be on a long
 *       critical path of computations and the closer to leaf-level and the less likely it is to
 *       cause further fanout.
 *   <li><i>evaluationCount</i>: {@link SkyFunction}s are written to minimize restarts and they are
 *       usually bounded for any node. The higher the evaluation count, the more likely the next
 *       restart completes the node, reducing memory pressure.
 * </ul>
 *
 * <p>Prioritization also uses statically determined information. Certain types of nodes have high
 * fanout by design. These may be annotated using the {@link SkyKey#hasLowFanout} method.
 */
interface PriorityTracker {
  /** The priority with higher meaning more urgent. */
  int getPriority();

  /**
   * Current estimated depth.
   *
   * <p>Depth is initialized by adding one to parent depth. For heavy computations to prioritize
   * correctly, their reverse dependencies under transitive closure (excluding the root) should also
   * track depth.
   *
   * <p>8-bits is likely enough for this. It's an {@code int} because Java doesn't support
   * arithmetic operations on narrower types.
   */
  int depth();

  /**
   * Attempts to update the depth.
   *
   * <p>During evaluation, parent nodes initialize priority as one more than their own priority.
   * Since a node may have multiple parents, depth may increase during evaluation.
   */
  void updateDepthIfGreater(int proposedDepth);

  /** Adds one to the evaluation count component of priority. */
  void incrementEvaluationCount();

  default int getChildDepth() {
    return depth() + 1;
  }
}
