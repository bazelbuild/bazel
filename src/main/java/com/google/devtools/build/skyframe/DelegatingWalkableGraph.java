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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

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
    NodeEntry entry =
        Preconditions.checkNotNull(
            graph.getBatchWithFieldHints(ImmutableList.of(key), NodeEntryField.VALUE_ONLY).get(key),
            key);
    Preconditions.checkState(entry.isDone(), "%s %s", key, entry);
    return entry;
  }

  @Override
  public boolean exists(SkyKey key) {
    NodeEntry entry =
        graph.getBatchWithFieldHints(ImmutableList.of(key), NodeEntryField.NO_FIELDS).get(key);
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
    return Maps.filterValues(
        Maps.transformValues(
            graph.getBatchWithFieldHints(keys, NodeEntryField.VALUE_ONLY), GET_SKY_VALUE_FUNCTION),
        Predicates.notNull());
  }

  @Override
  public Map<SkyKey, Exception> getMissingAndExceptions(Iterable<SkyKey> keys) {
    Map<SkyKey, Exception> result = new HashMap<>();
    Map<SkyKey, NodeEntry> graphResult =
        graph.getBatchWithFieldHints(keys, NodeEntryField.VALUE_ONLY);
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
    Map<SkyKey, NodeEntry> entries =
        graph.getBatchWithFieldHints(keys, EnumSet.of(NodeEntryField.DIRECT_DEPS));
    Map<SkyKey, Iterable<SkyKey>> result = new HashMap<>(entries.size());
    for (Entry<SkyKey, NodeEntry> entry : entries.entrySet()) {
      Preconditions.checkState(entry.getValue().isDone(), entry);
      result.put(entry.getKey(), entry.getValue().getDirectDeps());
    }
    return result;
  }

  @Override
  public Map<SkyKey, Iterable<SkyKey>> getReverseDeps(Iterable<SkyKey> keys) {
    Map<SkyKey, NodeEntry> entries =
        graph.getBatchWithFieldHints(keys, EnumSet.of(NodeEntryField.REVERSE_DEPS));
    Map<SkyKey, Iterable<SkyKey>> result = new HashMap<>(entries.size());
    for (Entry<SkyKey, NodeEntry> entry : entries.entrySet()) {
      Preconditions.checkState(entry.getValue().isDone(), entry);
      result.put(entry.getKey(), entry.getValue().getReverseDeps());
    }
    return result;
  }

}
