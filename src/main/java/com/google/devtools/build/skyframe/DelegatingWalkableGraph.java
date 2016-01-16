// Copyright 2015 The Bazel Authors. All rights reserved.
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
import com.google.common.base.Predicates;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import javax.annotation.Nullable;

/**
 * {@link WalkableGraph} that looks nodes up in a {@link QueryableGraph}.
 */
public class DelegatingWalkableGraph implements WalkableGraph {

  private final QueryableGraph fullGraph;
  private final QueryableGraph thinGraph;

  public DelegatingWalkableGraph(QueryableGraph graph) {
    this(graph, graph);
  }

  /**
   * Use this constructor when you want to differentiate reads that require the node value vs reads
   * that only traverse dependencies.
   */
  public DelegatingWalkableGraph(QueryableGraph fullGraph, QueryableGraph thinGraph) {
    this.fullGraph = fullGraph;
    this.thinGraph = thinGraph;
  }

  private NodeEntry getEntry(SkyKey key) {
    NodeEntry entry = Preconditions.checkNotNull(fullGraph.get(key), key);
    Preconditions.checkState(entry.isDone(), "%s %s", key, entry);
    return entry;
  }

  /**
   * Returns a map giving the {@link NodeEntry} corresponding to the given {@code keys}. If there is
   * no node in the graph corresponding to a {@link SkyKey} in {@code keys}, it is silently ignored
   * and will not be present in the returned map. This tolerance allows callers to avoid
   * pre-filtering their keys by checking for existence, which can be expensive.
   */
  private static Map<SkyKey, NodeEntry> getEntries(Iterable<SkyKey> keys, QueryableGraph graph) {
    Map<SkyKey, NodeEntry> result = graph.getBatch(keys);
    for (Map.Entry<SkyKey, NodeEntry> entry : result.entrySet()) {
      Preconditions.checkState(entry.getValue().isDone(), entry);
    }
    return result;
  }

  @Override
  public boolean exists(SkyKey key) {
    NodeEntry entry = thinGraph.get(key);
    return entry != null && entry.isDone();
  }

  @Nullable
  @Override
  public SkyValue getValue(SkyKey key) {
    return getEntry(key).getValue();
  }

  private static final Function<NodeEntry, SkyValue> GET_SKY_VALUE_FUNCTION =
      new Function<NodeEntry, SkyValue>() {
        @Nullable
        @Override
        public SkyValue apply(NodeEntry entry) {
          return entry.isDone() ? entry.getValue() : null;
        }
      };

  @Override
  public Map<SkyKey, SkyValue> getSuccessfulValues(Iterable<SkyKey> keys) {
    return Maps.filterValues(Maps.transformValues(fullGraph.getBatch(keys), GET_SKY_VALUE_FUNCTION),
        Predicates.notNull());
  }

  @Override
  public Map<SkyKey, Exception> getMissingAndExceptions(Iterable<SkyKey> keys) {
    Map<SkyKey, Exception> result = new HashMap<>();
    Map<SkyKey, NodeEntry> graphResult = fullGraph.getBatch(keys);
    for (SkyKey key : keys) {
      NodeEntry nodeEntry = graphResult.get(key);
      if (nodeEntry == null || !nodeEntry.isDone()) {
        result.put(key, null);
      } else {
        ErrorInfo errorInfo = nodeEntry.getErrorInfo();
        if (errorInfo != null) {
          result.put(key, errorInfo.getException());
        }
      }
    }
    return result;
  }

  @Nullable
  @Override
  public Exception getException(SkyKey key) {
    ErrorInfo errorInfo = getEntry(key).getErrorInfo();
    return errorInfo == null ? null : errorInfo.getException();
  }

  @Override
  public Map<SkyKey, Iterable<SkyKey>> getDirectDeps(Iterable<SkyKey> keys) {
    Map<SkyKey, NodeEntry> entries = getEntries(keys, thinGraph);
    Map<SkyKey, Iterable<SkyKey>> result = new HashMap<>(entries.size());
    for (Entry<SkyKey, NodeEntry> entry : entries.entrySet()) {
      result.put(entry.getKey(), entry.getValue().getDirectDeps());
    }
    return result;
  }

  @Override
  public Map<SkyKey, Iterable<SkyKey>> getReverseDeps(Iterable<SkyKey> keys) {
    Map<SkyKey, NodeEntry> entries = getEntries(keys, thinGraph);
    Map<SkyKey, Iterable<SkyKey>> result = new HashMap<>(entries.size());
    for (Entry<SkyKey, NodeEntry> entry : entries.entrySet()) {
      result.put(entry.getKey(), entry.getValue().getReverseDeps());
    }
    return result;
  }

}
