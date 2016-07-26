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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.MapMaker;
import com.google.common.collect.Maps;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Map;
import java.util.concurrent.ConcurrentMap;
import javax.annotation.Nullable;

/**
 * An in-memory graph implementation. All operations are thread-safe with ConcurrentMap semantics.
 * Also see {@link NodeEntry}.
 *
 * <p>This class is public only for use in alternative graph implementations.
 */
public class InMemoryGraphImpl implements InMemoryGraph {

  protected final ConcurrentMap<SkyKey, NodeEntry> nodeMap =
      new MapMaker().initialCapacity(1024).concurrencyLevel(200).makeMap();
  private final boolean keepEdges;

  InMemoryGraphImpl() {
    this(/*keepEdges=*/ true);
  }

  public InMemoryGraphImpl(boolean keepEdges) {
    this.keepEdges = keepEdges;
  }

  @Override
  public void remove(SkyKey skyKey) {
    nodeMap.remove(skyKey);
  }

  @Override
  public NodeEntry get(@Nullable SkyKey requestor, Reason reason, SkyKey skyKey) {
    return nodeMap.get(skyKey);
  }

  @Override
  public Map<SkyKey, NodeEntry> getBatchForInvalidation(Iterable<SkyKey> keys) {
    ImmutableMap.Builder<SkyKey, NodeEntry> builder = ImmutableMap.builder();
    for (SkyKey key : keys) {
      NodeEntry entry = get(null, Reason.OTHER, key);
      if (entry != null) {
        builder.put(key, entry);
      }
    }
    return builder.build();
  }

  @Override
  public Map<SkyKey, NodeEntry> getBatchWithFieldHints(
      SkyKey requestor, Reason reason, Iterable<SkyKey> keys, EnumSet<NodeEntryField> fields) {
    return getBatchForInvalidation(keys);
  }

  protected NodeEntry createIfAbsent(SkyKey key) {
    NodeEntry newval = keepEdges ? new InMemoryNodeEntry() : new EdgelessInMemoryNodeEntry();
    NodeEntry oldval = nodeMap.putIfAbsent(key, newval);
    return oldval == null ? newval : oldval;
  }

  @Override
  public Map<SkyKey, NodeEntry> createIfAbsentBatch(
      @Nullable SkyKey requestor, Reason reason, Iterable<SkyKey> keys) {
    ImmutableMap.Builder<SkyKey, NodeEntry> builder = ImmutableMap.builder();
    for (SkyKey key : keys) {
      builder.put(key, createIfAbsent(key));
    }
    return builder.build();
  }

  @Override
  public Map<SkyKey, SkyValue> getValues() {
    return Collections.unmodifiableMap(
        Maps.transformValues(
            nodeMap,
            new Function<NodeEntry, SkyValue>() {
              @Override
              public SkyValue apply(NodeEntry entry) {
                return entry.toValue();
              }
            }));
  }

  @Override
  public Map<SkyKey, SkyValue> getDoneValues() {
    return Collections.unmodifiableMap(
        Maps.filterValues(
            Maps.transformValues(
                nodeMap,
                new Function<NodeEntry, SkyValue>() {
                  @Override
                  public SkyValue apply(NodeEntry entry) {
                    return entry.isDone() ? entry.getValue() : null;
                  }
                }),
            Predicates.notNull()));
  }

  @Override
  public Map<SkyKey, NodeEntry> getAllValues() {
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
