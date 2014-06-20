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
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.DeletingInvalidationState;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.DirtyingInvalidationState;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.InvalidationState;
import com.google.devtools.build.skyframe.NodeEntry.DependencyState;

import java.io.PrintStream;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

import javax.annotation.Nullable;

/**
 * An inmemory implementation that uses the eager invalidation strategy. This class is, by itself,
 * not thread-safe. Neither is it thread-safe to use this class in parallel with any of the
 * returned graphs. However, it is allowed to access the graph from multiple threads as long as
 * that does not happen in parallel with an {@link #update} call.
 */
public final class InMemoryAutoUpdatingGraph implements AutoUpdatingGraph {

  private final ImmutableMap<? extends NodeType, ? extends NodeBuilder> nodeBuilders;
  @Nullable private final NodeProgressReceiver progressReceiver;
  // Not final only for testing.
  private InMemoryGraph graph;
  private long graphVersion = 0L;

  // State related to invalidation and deletion.
  private Set<NodeKey> nodesToDelete = new LinkedHashSet<>();
  private Set<NodeKey> nodesToDirty = new LinkedHashSet<>();
  private Map<NodeKey, Node> nodesToInject = new HashMap<>();
  private final InvalidationState deleterState = new DeletingInvalidationState();
  // Nodes that the caller explicitly specified are assumed to be changed -- they will be
  // re-evaluated even if none of their children are changed.
  private final InvalidationState invalidatorState = new DirtyingInvalidationState();

  private final EmittedEventState emittedEventState;

  private final AtomicBoolean updating = new AtomicBoolean(false);

  public InMemoryAutoUpdatingGraph(Map<? extends NodeType, ? extends NodeBuilder> nodeBuilders) {
    this(nodeBuilders, null);
  }

  public InMemoryAutoUpdatingGraph(Map<? extends NodeType, ? extends NodeBuilder> nodeBuilders,
      @Nullable NodeProgressReceiver invalidationReceiver) {
    this(nodeBuilders, invalidationReceiver, new EmittedEventState());
  }

  public InMemoryAutoUpdatingGraph(Map<? extends NodeType, ? extends NodeBuilder> nodeBuilders,
      @Nullable NodeProgressReceiver invalidationReceiver, EmittedEventState emittedEventState) {
    this.nodeBuilders = ImmutableMap.copyOf(nodeBuilders);
    this.progressReceiver = invalidationReceiver;
    this.graph = new InMemoryGraph();
    this.emittedEventState = emittedEventState;
  }

  @Override
  public void invalidate(Iterable<NodeKey> diff) {
    Iterables.addAll(nodesToDirty, diff);
  }

  @Override
  public void invalidateErrors() {
    // All error nodes have a dependency on the single global ERROR_TRANSIENCE node,
    // so we only have to invalidate that one node to catch everything.
    nodesToDirty.add(ErrorTransienceNode.key());
  }

  @Override
  public void delete(final Predicate<NodeKey> deletePredicate) {
    nodesToDelete.addAll(
        Maps.filterEntries(graph.getAllNodes(), new Predicate<Entry<NodeKey, NodeEntry>>() {
          @Override
          public boolean apply(Entry<NodeKey, NodeEntry> input) {
            return input.getValue().isDirty() || deletePredicate.apply(input.getKey());
          }
        }).keySet());
  }

  @Override
  public <T extends Node> UpdateResult<T> update(Iterable<NodeKey> roots, boolean keepGoing,
          int numThreads, ErrorEventListener listener) throws InterruptedException {
    // NOTE: Performance critical code. See bug "Null build performance parity".
    setAndCheckUpdateState(true, roots);
    try {
      pruneInjectedNodes();
      invalidate(nodesToInject.keySet());

      performNodeInvalidation(progressReceiver);

      injectNodes();

      ParallelEvaluator evaluator = new ParallelEvaluator(graph, graphVersion, nodeBuilders,
          listener, emittedEventState, keepGoing, numThreads, progressReceiver);
      // Increment graph version for next build.
      graphVersion++;
      return evaluator.eval(roots);
    } finally {
      setAndCheckUpdateState(false, roots);
    }
  }

  /**
   * Removes entries in {@link #nodesToInject} whose values are equal to the present values in the
   * graph.
   */
  private void pruneInjectedNodes() {
    for (Iterator<Entry<NodeKey, Node>> it = nodesToInject.entrySet().iterator(); it.hasNext();) {
      Entry<NodeKey, Node> entry = it.next();
      NodeKey key = entry.getKey();
      Node newValue = entry.getValue();
      NodeEntry prevEntry = graph.get(key);
      if (prevEntry != null && prevEntry.isDone()) {
        Iterable<NodeKey> directDeps = prevEntry.getDirectDeps();
        Preconditions.checkState(Iterables.isEmpty(directDeps),
            "existing entry for %s has deps: %s", key, directDeps);
        if (newValue.equals(prevEntry.getNode())
            && !nodesToDirty.contains(key) && !nodesToDelete.contains(key)) {
          it.remove();
        }
      }
    }
  }

  /**
   * Injects nodes in {@link #nodesToInject} into the graph.
   */
  private void injectNodes() {
    if (nodesToInject.isEmpty()) {
      return;
    }
    for (Entry<NodeKey, Node> entry : nodesToInject.entrySet()) {
      NodeKey key = entry.getKey();
      Node nodeValue = entry.getValue();
      Preconditions.checkState(nodeValue != null, key);
      NodeEntry prevEntry = graph.createIfAbsent(key);
      if (prevEntry.isDirty()) {
        // There was an existing entry for this key in the graph.
        // Get the node in the state where it is able to accept a value.
        Preconditions.checkState(prevEntry.getTemporaryDirectDeps().isEmpty(), key);

        DependencyState newState = prevEntry.addReverseDepAndCheckIfDone(null);
        Preconditions.checkState(newState == DependencyState.NEEDS_SCHEDULING, key);

        // Check that the previous node has no dependencies. Overwriting a node with deps with an
        // injected node (which is by definition deps-free) needs a little additional bookkeeping
        // (removing reverse deps from the dependencies), but more importantly it's something that
        // we want to avoid, because it indicates confusion of input nodes and derived nodes.
        Iterable<NodeKey> directDeps = prevEntry.getLastBuildDirectDeps();
        Preconditions.checkState(Iterables.isEmpty(directDeps),
            "existing entry for %s has deps: %s", key, directDeps);
      }
      prevEntry.setValue(entry.getValue(), graphVersion);
    }
    nodesToInject = new HashMap<>();
  }

  private void performNodeInvalidation(NodeProgressReceiver invalidationReceiver)
      throws InterruptedException {
    EagerInvalidator.delete(graph, nodesToDelete, invalidationReceiver, deleterState);
    // Note that clearing the nodesToDelete would not do an internal resizing. Therefore, if any
    // build has a large set of dirty nodes, subsequent operations (even clearing) will be slower.
    // Instead, just start afresh with a new LinkedHashSet.
    nodesToDelete = new LinkedHashSet<>();

    EagerInvalidator.invalidate(graph, nodesToDirty, invalidationReceiver, invalidatorState);
    // Ditto.
    nodesToDirty = new LinkedHashSet<>();
  }

  @Override
  public void inject(Map<NodeKey, ? extends Node> nodes) {
    nodesToInject.putAll(nodes);
  }

  private void setAndCheckUpdateState(boolean newValue, Object requestInfo) {
    Preconditions.checkState(updating.getAndSet(newValue) != newValue,
        "Re-entrant auto-graph-update for request: %s", requestInfo);
  }

  @Override
  public Map<NodeKey, Node> getNodes() {
    return graph.getNodes();
  }

  @Override
  public Map<NodeKey, Node> getDoneNodes() {
    return graph.getDoneNodes();
  }

  @Override
  @Nullable public Node getExistingNodeForTesting(NodeKey key) {
    return graph.getNode(key);
  }

  @Override
  @Nullable public ErrorInfo getExistingErrorForTesting(NodeKey key) {
    NodeEntry entry = graph.get(key);
    return (entry == null || !entry.isDone()) ? null : entry.getErrorInfo();
  }

  @Override
  public void setGraphForTesting(InMemoryGraph graph) {
    this.graph = graph;
  }

  @Override
  public void dump(PrintStream out) {
    Function<NodeKey, String> keyFormatter =
        new Function<NodeKey, String>() {
          @Override
          public String apply(NodeKey key) {
            return String.format("%s:%s",
                key.getNodeType(), key.getNodeName().toString().replace('\n', '_'));
          }
        };

    for (Entry<NodeKey, NodeEntry> mapPair : graph.getAllNodes().entrySet()) {
      NodeKey key = mapPair.getKey();
      NodeEntry entry = mapPair.getValue();
      if (entry.isDone()) {
        System.out.print(keyFormatter.apply(key));
        System.out.print("|");
        System.out.println(Joiner.on('|').join(
            Iterables.transform(entry.getDirectDeps(), keyFormatter)));
      }
    }
  }
}
