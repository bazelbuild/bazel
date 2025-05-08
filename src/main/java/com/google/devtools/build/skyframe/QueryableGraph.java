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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.supplier.InterruptibleSupplier;
import com.google.devtools.build.lib.supplier.MemoizingInterruptibleSupplier;
import com.google.devtools.build.skyframe.NodeEntry.DirtyType;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * A graph that exposes its entries and structure, for use by classes that must traverse it.
 *
 * <p>Certain graph implementations can throw {@link InterruptedException} when trying to retrieve
 * node entries. Such exceptions should not be caught locally -- they should be allowed to propagate
 * up.
 */
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
  NodeEntry get(@Nullable SkyKey requestor, Reason reason, SkyKey key) throws InterruptedException;

  /**
   * Fetches all the given nodes. Returns a {@link NodeBatch} {@code b} such that, for all {@code k}
   * in {@code keys}, {@code b.get(k) == get(k)}.
   *
   * <p>Prefer calling this method over {@link #getBatchMap} if it is not necessary to represent the
   * result as a {@link Map}, as it may be significantly more efficient.
   *
   * @param requestor if non-{@code null}, the node on behalf of which the given {@code keys} are
   *     being requested.
   * @param reason the reason the nodes are being requested.
   */
  default NodeBatch getBatch(
      @Nullable SkyKey requestor, Reason reason, Iterable<? extends SkyKey> keys)
      throws InterruptedException {
    return getBatchMap(requestor, reason, keys)::get;
  }

  /** A hint about the most efficient way to look up a key in the graph. */
  enum LookupHint {
    INDIVIDUAL,
    BATCH
  }

  /**
   * Hints to the caller about the most efficient way to look up a key in this graph.
   *
   * <p>A return of {@link LookupHint#INDIVIDUAL} indicates that the given key can efficiently be
   * looked up by calling {@link #get}. In such a case, it is not worth the effort to aggregate the
   * key into a collection with other keys for a {@link #getBatch} call.
   *
   * <p>A return of {@link LookupHint#BATCH} indicates that the given key should ideally be
   * requested with other keys as part of a call to {@link #getBatch}. This may be the case if, for
   * example, the corresponding node is stored remotely, and requesting keys in a single batch
   * reduces trips to remote storage.
   */
  LookupHint getLookupHint(SkyKey key);

  /**
   * A version of {@link #getBatch} that returns an {@link InterruptibleSupplier} to possibly
   * retrieve the results later.
   */
  @CanIgnoreReturnValue
  default InterruptibleSupplier<NodeBatch> getBatchAsync(
      @Nullable SkyKey requestor, Reason reason, Iterable<? extends SkyKey> keys) {
    return MemoizingInterruptibleSupplier.of(() -> getBatch(requestor, reason, keys));
  }

  /**
   * Fetches all the given nodes. Returns a map {@code m} such that, for all {@code k} in {@code
   * keys}, {@code m.get(k) == get(k)} and {@code !m.containsKey(k)} iff {@code get(k) == null}.
   *
   * <p>Prefer calling {@link #getBatch} over this method if it is not necessary to represent the
   * result as a {@link Map}, as it may be significantly more efficient.
   *
   * @param requestor if non-{@code null}, the node on behalf of which the given {@code keys} are
   *     being requested.
   * @param reason the reason the nodes are being requested.
   */
  Map<SkyKey, ? extends NodeEntry> getBatchMap(
      @Nullable SkyKey requestor, Reason reason, Iterable<? extends SkyKey> keys)
      throws InterruptedException;

  /**
   * A version of {@link #getBatchMap} that returns an {@link InterruptibleSupplier} to possibly
   * retrieve the results later.
   */
  @CanIgnoreReturnValue
  default InterruptibleSupplier<Map<SkyKey, ? extends NodeEntry>> getBatchMapAsync(
      @Nullable SkyKey requestor, Reason reason, Iterable<? extends SkyKey> keys) {
    return MemoizingInterruptibleSupplier.of(() -> getBatchMap(requestor, reason, keys));
  }

  /**
   * Optimistically prefetches dependencies.
   *
   * @param requestor the key whose deps to fetch
   * @param oldDeps deps from the previous build
   * @param previouslyRequestedDeps deps that have already been requested during this build and
   *     should not be prefetched because they will be subsequently fetched anyway
   * @return {@code previouslyRequestedDeps} as a set if the implementation called {@link
   *     GroupedDeps#toSet} (so that the caller may reuse it), otherwise {@code null}
   */
  @CanIgnoreReturnValue
  @Nullable
  default ImmutableSet<SkyKey> prefetchDeps(
      SkyKey requestor, Set<SkyKey> oldDeps, GroupedDeps previouslyRequestedDeps)
      throws InterruptedException {
    return null;
  }

  default ImmutableSet<SkyKey> getAllKeysForTesting() {
    throw new UnsupportedOperationException();
  }

  /**
   * The reason that a node is being looked up in the Skyframe graph.
   *
   * <p>Alternate graph implementations may wish to make use of this information.
   */
  enum Reason {
    /**
     * The node is being fetched in order to see if it needs to be evaluated or because it was just
     * evaluated, but *not* because it was just requested during evaluation of a SkyFunction
     * (see {@link #DEP_REQUESTED}).
     */
    PRE_OR_POST_EVALUATION,

    /**
     * The node is being looked up as part of the prefetch step before evaluation of a SkyFunction.
     */
    PREFETCH,

    /**
     * The node is being fetched because it is about to be evaluated, but *not* because it was just
     * requested during evaluation of a SkyFunction (see {@link #DEP_REQUESTED}).
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

    /** The node is being looked up for any graph clean-up effort that may be necessary. */
    CLEAN_UP,

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

    /** The node is being looked up merely for an existence check. */
    EXISTENCE_CHECKING,

    /** The node is being looked up merely to see if it is done or not. */
    DONE_CHECKING,

    /** The node is being looked up so that it can be {@linkplain DirtyType#REWIND rewound}. */
    REWINDING,

    /**
     * The node is being looked up to service {@link WalkableGraph#getValue},
     * {@link WalkableGraph#getException}, {@link WalkableGraph#getMissingAndExceptions}, or
     * {@link WalkableGraph#getSuccessfulValues}.
     */
    WALKABLE_GRAPH_VALUE,

    /** The node is being looked up to service {@link WalkableGraph#getDirectDeps}. */
    WALKABLE_GRAPH_DEPS,

    /** The node is being looked up to service {@link WalkableGraph#getReverseDeps}. */
    WALKABLE_GRAPH_RDEPS,

    /** The node is being looked up to service {@link WalkableGraph#getValueAndRdeps}. */
    WALKABLE_GRAPH_VALUE_AND_RDEPS,

    /** The node is being looked up to service another "graph lookup" function. */
    WALKABLE_GRAPH_OTHER,

    /** The node is being looked up to vendor external repos from its dependencies. */
    VENDOR_EXTERNAL_REPOS,

    /** Some other reason than one of the above that needs the node's value and deps. */
    OTHER_NEEDING_VALUE_AND_DEPS,

    /** Some other reason than one of the above that needs the node's reverse deps. */
    OTHER_NEEDING_REVERSE_DEPS,

    /** Some other reason than one of the above that needs the node's value and reverse deps. */
    OTHER_NEEDING_VALUE_AND_REVERSE_DEPS,

    /** Some other reason than one of the above. */
    OTHER;

    public boolean isWalkable() {
      return this == WALKABLE_GRAPH_VALUE
          || this == WALKABLE_GRAPH_DEPS
          || this == WALKABLE_GRAPH_RDEPS
          || this == WALKABLE_GRAPH_VALUE_AND_RDEPS
          || this == WALKABLE_GRAPH_OTHER;
    }
  }
}
