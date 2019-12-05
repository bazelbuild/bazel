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

import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * {@link WalkableGraph} that looks nodes up in a {@link QueryableGraph}.
 */
public class DelegatingWalkableGraph implements WalkableGraph {
  protected final QueryableGraph graph;

  public DelegatingWalkableGraph(QueryableGraph graph) {
    this.graph = graph;
  }

  @Nullable
  private NodeEntry getEntryForValue(SkyKey key) throws InterruptedException {
    NodeEntry entry = graph.get(null, Reason.WALKABLE_GRAPH_VALUE, key);
    return entry != null && entry.isDone() ? entry : null;
  }

  @Nullable
  @Override
  public SkyValue getValue(SkyKey key) throws InterruptedException {
    NodeEntry entry = getEntryForValue(key);
    return entry == null ? null : entry.getValue();
  }

  private static SkyValue getValueFromNodeEntry(NodeEntry entry) throws InterruptedException {
    return entry.isDone() ? entry.getValue() : null;
  }

  @Override
  public Map<SkyKey, SkyValue> getSuccessfulValues(Iterable<SkyKey> keys)
      throws InterruptedException {
    Map<SkyKey, ? extends NodeEntry> batchGet = getBatch(null, Reason.WALKABLE_GRAPH_VALUE, keys);
    Map<SkyKey, SkyValue> result = Maps.newHashMapWithExpectedSize(batchGet.size());
    for (Map.Entry<SkyKey, ? extends NodeEntry> entryPair : batchGet.entrySet()) {
      SkyValue value = getValueFromNodeEntry(entryPair.getValue());
      if (value != null) {
        result.put(entryPair.getKey(), value);
      }
    }
    return result;
  }

  @Override
  public Map<SkyKey, Exception> getMissingAndExceptions(Iterable<SkyKey> keys)
      throws InterruptedException {
    Map<SkyKey, Exception> result = new HashMap<>();
    Map<SkyKey, ? extends NodeEntry> graphResult =
        getBatch(null, Reason.WALKABLE_GRAPH_VALUE, keys);
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

  @Override
  public boolean isCycle(SkyKey key) throws InterruptedException {
    NodeEntry entry = getEntryForValue(key);
    if (entry == null) {
      return false;
    }
    ErrorInfo errorInfo = entry.getErrorInfo();
    return errorInfo != null && !errorInfo.getCycleInfo().isEmpty();
  }

  @Nullable
  @Override
  public Exception getException(SkyKey key) throws InterruptedException {
    NodeEntry entry = getEntryForValue(key);
    if (entry == null) {
      return null;
    }
    ErrorInfo errorInfo = entry.getErrorInfo();
    return errorInfo == null ? null : errorInfo.getException();
  }

  @Override
  public Map<SkyKey, Iterable<SkyKey>> getDirectDeps(Iterable<SkyKey> keys)
      throws InterruptedException {
    Map<SkyKey, ? extends NodeEntry> entries = getBatch(null, Reason.WALKABLE_GRAPH_DEPS, keys);
    Map<SkyKey, Iterable<SkyKey>> result = new HashMap<>(entries.size());
    for (Map.Entry<SkyKey, ? extends NodeEntry> entry : entries.entrySet()) {
      Preconditions.checkState(entry.getValue().isDone(), entry);
      result.put(entry.getKey(), entry.getValue().getDirectDeps());
    }
    return result;
  }

  @Override
  public Iterable<SkyKey> getDirectDeps(SkyKey key) throws InterruptedException {
    NodeEntry entry = getEntryForValue(key);
    Preconditions.checkNotNull(entry, key);
    Preconditions.checkState(entry.isDone(), "Node %s (with key %s) isn't done yet.", entry, key);
    return entry.getDirectDeps();
  }

  @Override
  public Map<SkyKey, Iterable<SkyKey>> getReverseDeps(Iterable<SkyKey> keys)
      throws InterruptedException {
    Map<SkyKey, ? extends NodeEntry> entries = getBatch(null, Reason.WALKABLE_GRAPH_RDEPS, keys);
    Map<SkyKey, Iterable<SkyKey>> result = new HashMap<>(entries.size());
    for (Map.Entry<SkyKey, ? extends NodeEntry> entry : entries.entrySet()) {
      Preconditions.checkState(entry.getValue().isDone(), entry);
      result.put(entry.getKey(), entry.getValue().getReverseDepsForDoneEntry());
    }
    return result;
  }

  protected Map<SkyKey, ? extends NodeEntry> getBatch(
      @Nullable SkyKey requestor, Reason reason, Iterable<? extends SkyKey> keys)
      throws InterruptedException {
    return graph.getBatch(requestor, reason, keys);
  }

  @Override
  public Map<SkyKey, Pair<SkyValue, Iterable<SkyKey>>> getValueAndRdeps(Iterable<SkyKey> keys)
      throws InterruptedException {
    Map<SkyKey, ? extends NodeEntry> entries =
        getBatch(null, Reason.WALKABLE_GRAPH_VALUE_AND_RDEPS, keys);
    Map<SkyKey, Pair<SkyValue, Iterable<SkyKey>>> result =
        Maps.newHashMapWithExpectedSize(entries.size());
    for (Map.Entry<SkyKey, ? extends NodeEntry> entry : entries.entrySet()) {
      Preconditions.checkState(entry.getValue().isDone(), entry);
      result.put(
          entry.getKey(),
          Pair.of(
              getValueFromNodeEntry(entry.getValue()),
              entry.getValue().getReverseDepsForDoneEntry()));
    }
    return result;
  }
}
