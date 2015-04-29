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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.MapMaker;
import com.google.common.collect.Maps;

import java.util.Collections;
import java.util.Map;
import java.util.concurrent.ConcurrentMap;

import javax.annotation.Nullable;

/**
 * An in-memory graph implementation. All operations are thread-safe with ConcurrentMap semantics.
 * Also see {@link NodeEntry}.
 *
 * <p>This class is public only for use in alternative graph implementations.
 */
public class InMemoryGraph implements ProcessableGraph {

  protected final ConcurrentMap<SkyKey, NodeEntry> nodeMap =
      new MapMaker().initialCapacity(1024).concurrencyLevel(200).makeMap();
  private final boolean keepEdges;

  InMemoryGraph() {
    this(/*keepEdges=*/true);
  }

  public InMemoryGraph(boolean keepEdges) {
    this.keepEdges = keepEdges;
  }

  @Override
  public void remove(SkyKey skyKey) {
    nodeMap.remove(skyKey);
  }

  @Override
  public NodeEntry get(SkyKey skyKey) {
    return nodeMap.get(skyKey);
  }

  @Override
  public Map<SkyKey, NodeEntry> getBatch(Iterable<SkyKey> keys) {
    ImmutableMap.Builder<SkyKey, NodeEntry> builder = ImmutableMap.builder();
    for (SkyKey key : keys) {
      NodeEntry entry = get(key);
      if (entry != null) {
        builder.put(key, entry);
      }
    }
    return builder.build();
  }

  @Override
  public NodeEntry createIfAbsent(SkyKey key) {
    NodeEntry newval = keepEdges ? new InMemoryNodeEntry() : new EdgelessInMemoryNodeEntry();
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
   * Returns a value, if it exists. If not, returns null.
   */
  @Nullable public SkyValue getValue(SkyKey key) {
    NodeEntry entry = get(key);
    return NODE_DONE_PREDICATE.apply(entry) ? entry.getValue() : null;
  }

  /**
   * Returns a read-only live view of the nodes in the graph. All node are included. Dirty values
   * include their Node value. Values in error have a null value.
   */
  Map<SkyKey, SkyValue> getValues() {
    return Collections.unmodifiableMap(Maps.transformValues(
        nodeMap,
        new Function<NodeEntry, SkyValue>() {
          @Override
          public SkyValue apply(NodeEntry entry) {
            return entry.toValue();
          }
        }));
  }

  /**
   * Returns a read-only live view of the done values in the graph. Dirty, changed, and error values
   * are not present in the returned map
   */
  Map<SkyKey, SkyValue> getDoneValues() {
    return Collections.unmodifiableMap(Maps.filterValues(Maps.transformValues(
        nodeMap,
        new Function<NodeEntry, SkyValue>() {
          @Override
          public SkyValue apply(NodeEntry entry) {
            return entry.isDone() ? entry.getValue() : null;
          }
        }), Predicates.notNull()));
  }

  // Only for use by MemoizingEvaluator#delete
  Map<SkyKey, NodeEntry> getAllValues() {
    return Collections.unmodifiableMap(nodeMap);
  }

  @VisibleForTesting
  protected ConcurrentMap<SkyKey, NodeEntry> getNodeMap() {
    return nodeMap;
  }

  boolean keepsEdges() {
    return keepEdges;
  }
}
