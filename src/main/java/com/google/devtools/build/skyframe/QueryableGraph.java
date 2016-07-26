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

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;

import java.util.EnumSet;
import java.util.Map;

import javax.annotation.Nullable;

/** A graph that exposes its entries and structure, for use by classes that must traverse it. */
@ThreadSafe
public interface QueryableGraph {
  /**
   * Returns the node with the given {@code key}, or {@code null} if the node does not exist.
   *
   * @param requestor if non-{@code null}, the node on behalf of which {@code key} is being
   *     requested.
   * @param reason the reason the node is being requested.
   */
  @Nullable
  NodeEntry get(@Nullable SkyKey requestor, Reason reason, SkyKey key);

  /**
   * Fetches all the given nodes. Returns a map {@code m} such that, for all {@code k} in {@code
   * keys}, {@code m.get(k).equals(e)} iff {@code get(k) == e} and {@code e != null}, and {@code
   * !m.containsKey(k)} iff {@code get(k) == null}. The {@code fields} parameter is a hint to the
   * QueryableGraph implementation that allows it to possibly construct certain fields of the
   * returned node entries more lazily. Hints may only be applied to nodes in a certain state, like
   * done nodes.
   * 
   * @param requestor if non-{@code null}, the node on behalf of which the given {@code keys} are
   *     being requested.
   * @param reason the reason the nodes are being requested.
   */
  Map<SkyKey, NodeEntry> getBatchWithFieldHints(
      @Nullable SkyKey requestor,
      Reason reason,
      Iterable<SkyKey> keys,
      EnumSet<NodeEntryField> fields);

  /**
   * The reason that a node is being looked up in the Skyframe graph.
   *
   * <p>Alternate graph implementations may wish to make use of this information.
   */
  enum Reason {
    /**
     * The node is being looked up as part of the prefetch step before evaluation of a SkyFunction.
     */
    PREFETCH,

    /**
     * The node is being fetched because it is about to be evaluated or it has already been
     * evaluated, but *not* because it was just requested during evaluation of a SkyFunction (see
     * DEP_REQUESTED).
     */
    EVALUATION,

    /** The node is being looked up because it was requested during evaluation of a SkyFunction. */
    DEP_REQUESTED,

    /** The node is being looked up during the invalidation phase of Skyframe evaluation. */
    INVALIDATION,

    /** The node is being looked up during the cycle checking phase of Skyframe evaluation. */
    CYCLE_CHECKING,

    /** The node is being looked up so that an rdep can be added to it. */
    RDEP_ADDITION,

    /** The node is being looked up so that an rdep can be removed from it. */
    RDEP_REMOVAL,

    /** The node is being looked up so it can be enqueued for evaluation or change pruning. */
    ENQUEUING_CHILD,

    /**
     * The node is being looked up so that it can be signaled that a dependency is now complete.
     */
    SIGNAL_DEP,

    /**
     * The node is being looking up as part of the error bubbling phase of fail-fast Skyframe
     * evaluation.
     */
    ERROR_BUBBLING,

    /** Some other reason than one of the above. */
    OTHER,
  }
}
