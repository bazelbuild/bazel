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
import com.google.common.base.Function;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.Maps;

import java.util.Collections;
import java.util.Map;
import java.util.concurrent.ConcurrentMap;

import javax.annotation.Nullable;

/**
 * An in-memory graph implementation. All operations are thread-safe with ConcurrentMap semantics.
 * Also see {@link NodeEntry}.
 * 
 * <p>This class is non-final only for testing purposes.
 */
class InMemoryGraph implements ProcessableGraph {
  
  private final ConcurrentMap<NodeKey, NodeEntry> nodeMap = Maps.newConcurrentMap();

  public InMemoryGraph() {
  }

  @Override
  public void remove(NodeKey nodeKey) {
    nodeMap.remove(nodeKey);
  }

  @Override
  public NodeEntry get(NodeKey nodeKey) {
    return nodeMap.get(nodeKey);
  }

  @Override
  public NodeEntry createIfAbsent(NodeKey key) {
    NodeEntry newval = new NodeEntry();
    NodeEntry oldval = nodeMap.putIfAbsent(key, newval);
    return oldval == null ? newval : oldval;
  }

  /** Only done nodes exist to the outside world. */
  private static final Predicate<NodeEntry> NODE_DONE_PREDICATE =
      new Predicate<NodeEntry>() {
        @Override
        public boolean apply(NodeEntry entry) {
          return entry != null && entry.isDone();
        }
      };

  /**
   * Returns a node, if it exists. If not, returns null.
   */
  @Nullable public Node getNode(NodeKey key) {
    NodeEntry entry = get(key);
    return NODE_DONE_PREDICATE.apply(entry) ? entry.getNode() : null;
  }

  /**
   * Returns a read-only live view of the nodes in the graph. All nodes are included. Dirty nodes
   * include their Node value. Nodes in error have a null value.
   */
  Map<NodeKey, Node> getNodes() {
    return Collections.unmodifiableMap(Maps.transformValues(
        nodeMap,
        new Function<NodeEntry, Node>() {
          @Override
          public Node apply(NodeEntry entry) {
            return entry.toNodeValue();
          }
        }));
  }

  /**
   * Returns a read-only live view of the done nodes in the graph. Dirty, changed, and error nodes
   * are not present in the returned map
   */
  Map<NodeKey, Node> getDoneNodes() {
    return Collections.unmodifiableMap(Maps.filterValues(Maps.transformValues(
        nodeMap,
        new Function<NodeEntry, Node>() {
          @Override
          public Node apply(NodeEntry entry) {
            return entry.isDone() ? entry.getNode() : null;
          }
        }), Predicates.notNull()));
  }

  // Only for use by AutoUpdatingGraph#delete
  Map<NodeKey, NodeEntry> getAllNodes() {
    return Collections.unmodifiableMap(nodeMap);
  }

  @VisibleForTesting
  protected ConcurrentMap<NodeKey, NodeEntry> getNodeMap() {
    return nodeMap;
  }
}
