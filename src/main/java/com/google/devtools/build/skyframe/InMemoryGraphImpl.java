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
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.MapMaker;
import com.google.common.collect.Maps;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
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

  protected final ConcurrentMap<SkyKey, InMemoryNodeEntry> nodeMap =
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
  public InMemoryNodeEntry get(@Nullable SkyKey requestor, Reason reason, SkyKey skyKey) {
    return nodeMap.get(skyKey);
  }

  @Override
  public Map<SkyKey, NodeEntry> getBatch(
      SkyKey requestor, Reason reason, Iterable<? extends SkyKey> keys) {
    // Use a HashMap, not an ImmutableMap.Builder, because we have not yet deduplicated these keys
    // and ImmutableMap.Builder does not tolerate duplicates. The map will be thrown away shortly.
    HashMap<SkyKey, NodeEntry> result = new HashMap<>();
    for (SkyKey key : keys) {
      InMemoryNodeEntry entry = get(null, Reason.OTHER, key);
      if (entry != null) {
        result.put(key, entry);
      }
    }
    return result;
  }

  protected InMemoryNodeEntry createIfAbsent(SkyKey key) {
    InMemoryNodeEntry newval =
        keepEdges ? new InMemoryNodeEntry() : new EdgelessInMemoryNodeEntry();
    InMemoryNodeEntry oldval = nodeMap.putIfAbsent(key, newval);
    return oldval == null ? newval : oldval;
  }

  @Override
  public Map<SkyKey, InMemoryNodeEntry> createIfAbsentBatch(
      @Nullable SkyKey requestor, Reason reason, Iterable<SkyKey> keys) {
    ImmutableMap.Builder<SkyKey, InMemoryNodeEntry> builder = ImmutableMap.builder();
    for (SkyKey key : keys) {
      builder.put(key, createIfAbsent(key));
    }
    return builder.build();
  }

  @Override
  public DepsReport analyzeDepsDoneness(SkyKey parent, Collection<SkyKey> deps) {
    return DepsReport.NO_INFORMATION;
  }

  @Override
  public Map<SkyKey, SkyValue> getValues() {
    return Collections.unmodifiableMap(
        Maps.transformValues(
            nodeMap,
            new Function<InMemoryNodeEntry, SkyValue>() {
              @Override
              public SkyValue apply(InMemoryNodeEntry entry) {
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
                new Function<InMemoryNodeEntry, SkyValue>() {
                  @Override
                  public SkyValue apply(InMemoryNodeEntry entry) {
                    return entry.isDone() ? entry.getValue() : null;
                  }
                }),
            Predicates.notNull()));
  }

  @Override
  public Map<SkyKey, InMemoryNodeEntry> getAllValues() {
    return Collections.unmodifiableMap(nodeMap);
  }

  @Override
  public Map<SkyKey, ? extends NodeEntry> getAllValuesMutable() {
    return nodeMap;
  }

  @VisibleForTesting
  protected ConcurrentMap<SkyKey, ? extends NodeEntry> getNodeMap() {
    return nodeMap;
  }

  boolean keepsEdges() {
    return keepEdges;
  }

  @Override
  public Iterable<SkyKey> getCurrentlyAvailableNodes(Iterable<SkyKey> keys, Reason reason) {
    ImmutableSet.Builder<SkyKey> builder = ImmutableSet.builder();
    for (SkyKey key : keys) {
      if (get(null, reason, key) != null) {
        builder.add(key);
      }
    }
    return builder.build();
  }
}
