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
import com.google.common.base.Predicate;
import com.google.devtools.build.lib.collect.nestedset.NestedSetVisitor;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadHostile;
import com.google.devtools.build.lib.events.ErrorEventListener;

import java.io.PrintStream;
import java.util.Map;

import javax.annotation.Nullable;

/**
 * A graph, defined by a set of functions that can construct nodes from node keys.
 *
 * <p>The node constructor functions ({@link NodeBuilder}s) can declare dependencies on prerequisite
 * {@link Node}s. The {@link AutoUpdatingGraph} implementation makes sure that those are created
 * beforehand.
 *
 * <p>The graph caches previously computed node values. Arbitrary nodes can be invalidated between
 * calls to {@link #update}; they will be recreated the next time they are requested.
 */
public interface AutoUpdatingGraph {

  /**
   * Computes the transitive closure of a given set of nodes at the given {@link Version}. See
   * {@link EagerInvalidator#invalidate}.
   */
  <T extends Node> UpdateResult<T> update(Iterable<NodeKey> roots, Version version,
                                          boolean keepGoing, int numThreads,
                                          ErrorEventListener reporter)
      throws InterruptedException;

  /**
   * Ensures that after the next completed {@link #update} call the current values of any node
   * matching this predicate (and all nodes that transitively depend on them) will be removed from
   * the node cache. All nodes that were already marked dirty in the graph will also be deleted,
   * regardless of whether or not they match the predicate.
   *
   * <p>If a later call to {@link #update} requests some of the deleted nodes, those nodes will be
   * recomputed and the new values stored in the cache again.
   *
   * <p>To delete all dirty nodes, you can specify a predicate that's always false.
   */
  void delete(Predicate<NodeKey> pred);

  /**
   * Marks dirty nodes for deletion if they have been dirty for at least as many graph versions
   * as the specified limit.
   *
   * <p>This ensures that after the next completed {@link #update} call, all such nodes, along
   * with all nodes that transitively depend on them, will be removed from the node cache. Nodes
   * that were marked dirty after the threshold version will not be affected by this call.
   *
   * <p>If a later call to {@link #update} requests some of the deleted nodes, those nodes will be
   * recomputed and the new values stored in the cache again.
   *
   * <p>To delete all dirty nodes, you can specify 0 for the limit.
   */
  void deleteDirty(final long versionAgeLimit);

  /**
   * Returns the nodes in the graph.
   *
   * <p>The returned map may be a live view of the graph.
   */
  Map<NodeKey, Node> getNodes();


  /**
   * Returns the done (without error) nodes in the graph.
   *
   * <p>The returned map may be a live view of the graph.
   */
  Map<NodeKey, Node> getDoneNodes();

  /**
   * Returns a node if and only if an earlier call to {@link #update} created it; null otherwise.
   *
   * <p>This method should only be used by tests that need to verify the presence of a node in the
   * graph after an {@link #update} call.
   */
  @VisibleForTesting
  @Nullable
  Node getExistingNodeForTesting(NodeKey key);

  /**
   * Returns an error if and only if an earlier call to {@link #update} created it; null otherwise.
   *
   * <p>This method should only be used by tests that need to verify the presence of an error in the
   * graph after an {@link #update} call.
   */
  @VisibleForTesting
  @Nullable
  ErrorInfo getExistingErrorForTesting(NodeKey key);

  /**
   * Write the graph to the output stream. Not necessarily thread-safe. Use only for debugging
   * purposes.
   */
  @ThreadHostile
  void dump(PrintStream out);

  /**
   * A supplier for creating instances of a particular graph implementation.
   */
  public static interface GraphSupplier {
    AutoUpdatingGraph createGraph(Map<? extends NodeType, ? extends NodeBuilder> nodeBuilders,
        Differencer differencer, @Nullable NodeProgressReceiver invalidationReceiver,
        EmittedEventState emittedEventState);
  }

  /**
   * Keeps track of already-emitted events. Users of the graph should instantiate an
   * {@code EmittedEventState} first and pass it to the graph during creation. This allows them to
   * determine whether or not to replay events.
   */
  public static class EmittedEventState extends NestedSetVisitor.VisitedState<TaggedEvents> {}

}
