// Copyright 2015 Google Inc. All rights reserved.
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
import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;

import java.util.Map;

import javax.annotation.Nullable;

/**
 * {@link WalkableGraph} that looks nodes up in a {@link QueryableGraph}.
 */
public class DelegatingWalkableGraph implements WalkableGraph {
  private final QueryableGraph graph;

  public DelegatingWalkableGraph(QueryableGraph graph) {
    this.graph = graph;
  }

  private NodeEntry getEntry(SkyKey key) {
    NodeEntry entry = Preconditions.checkNotNull(graph.get(key), key);
    Preconditions.checkState(entry.isDone(), "%s %s", key, entry);
    return entry;
  }

  private Map<SkyKey, NodeEntry> getEntries(Iterable<SkyKey> keys) {
    Map<SkyKey, NodeEntry> result = graph.getBatch(keys);
    Preconditions.checkState(result.size() == Iterables.size(keys), "%s %s", keys, result);
    for (Map.Entry<SkyKey, NodeEntry> entry : result.entrySet()) {
      Preconditions.checkState(entry.getValue().isDone(), entry);
    }
    return result;
  }

  @Override
  public boolean exists(SkyKey key) {
    NodeEntry entry = graph.get(key);
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
  public Map<SkyKey, SkyValue> getValuesMaybe(Iterable<SkyKey> keys) {
    return Maps.filterValues(Maps.transformValues(graph.getBatch(keys), GET_SKY_VALUE_FUNCTION),
        Predicates.notNull());
  }

  @Nullable
  @Override
  public Exception getException(SkyKey key) {
    ErrorInfo errorInfo = getEntry(key).getErrorInfo();
    return errorInfo == null ? null : errorInfo.getException();
  }

  @Override
  public Iterable<SkyKey> getDirectDeps(SkyKey key) {
    return getEntry(key).getDirectDeps();
  }

  private static final Function<NodeEntry, Iterable<SkyKey>> GET_DIRECT_DEPS_FUNCTION =
      new Function<NodeEntry, Iterable<SkyKey>>() {
        @Override
        public Iterable<SkyKey> apply(NodeEntry entry) {
          return entry.getDirectDeps();
        }
      };

  @Override
  public Map<SkyKey, Iterable<SkyKey>> getDirectDeps(Iterable<SkyKey> keys) {
    return Maps.transformValues(getEntries(keys), GET_DIRECT_DEPS_FUNCTION);
  }

  @Override
  public Iterable<SkyKey> getReverseDeps(SkyKey key) {
    return getEntry(key).getReverseDeps();
  }

  private static final Function<NodeEntry, Iterable<SkyKey>> GET_REVERSE_DEPS_FUNCTION =
      new Function<NodeEntry, Iterable<SkyKey>>() {
        @Override
        public Iterable<SkyKey> apply(NodeEntry entry) {
          return entry.getReverseDeps();
        }
      };

  @Override
  public Map<SkyKey, Iterable<SkyKey>> getReverseDeps(Iterable<SkyKey> keys) {
    return Maps.transformValues(getEntries(keys), GET_REVERSE_DEPS_FUNCTION);
  }
}
