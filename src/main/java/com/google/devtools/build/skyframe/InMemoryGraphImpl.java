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

import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.collect.compacthashmap.CompactHashMap;
import com.google.errorprone.annotations.ForOverride;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import javax.annotation.Nullable;

/**
 * An in-memory graph implementation. All operations are thread-safe with ConcurrentMap semantics.
 * Also see {@link NodeEntry}.
 *
 * <p>This class is public only for use in alternative graph implementations.
 */
public class InMemoryGraphImpl implements InMemoryGraph {

  protected final ConcurrentHashMap<SkyKey, InMemoryNodeEntry> nodeMap;

  InMemoryGraphImpl() {
    this(/*initialCapacity=*/ 1 << 10);
  }

  protected InMemoryGraphImpl(int initialCapacity) {
    this.nodeMap = new ConcurrentHashMap<>(initialCapacity);
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
  public Map<SkyKey, NodeEntry> getBatch(
      SkyKey requestor, Reason reason, Iterable<? extends SkyKey> keys) {
    // Use a HashMap, not an ImmutableMap.Builder, because we have not yet deduplicated these keys
    // and ImmutableMap.Builder does not tolerate duplicates. The map will be thrown away shortly.
    HashMap<SkyKey, NodeEntry> result = new HashMap<>();
    for (SkyKey key : keys) {
      NodeEntry entry = get(null, Reason.OTHER, key);
      if (entry != null) {
        result.put(key, entry);
      }
    }
    return result;
  }

  @ForOverride
  protected InMemoryNodeEntry newNodeEntry(SkyKey key) {
    return new InMemoryNodeEntry();
  }

  /**
   * This is used to call newNodeEntry() from within computeIfAbsent. Instantiated here to avoid
   * lambda instantiation overhead.
   */
  @SuppressWarnings("UnnecessaryLambda")
  private final Function<SkyKey, InMemoryNodeEntry> newNodeEntryFunction = this::newNodeEntry;

  @Override
  public Map<SkyKey, NodeEntry> createIfAbsentBatch(
      @Nullable SkyKey requestor, Reason reason, Iterable<? extends SkyKey> keys) {
    Map<SkyKey, NodeEntry> result = CompactHashMap.createWithExpectedSize(Iterables.size(keys));
    for (SkyKey key : keys) {
      result.put(key, nodeMap.computeIfAbsent(key, newNodeEntryFunction));
    }
    return result;
  }

  @Override
  public DepsReport analyzeDepsDoneness(SkyKey parent, Collection<SkyKey> deps) {
    return DepsReport.NO_INFORMATION;
  }

  @Override
  public Map<SkyKey, SkyValue> getValues() {
    return Collections.unmodifiableMap(Maps.transformValues(nodeMap, InMemoryNodeEntry::toValue));
  }

  @Override
  public Map<SkyKey, InMemoryNodeEntry> getAllValues() {
    return Collections.unmodifiableMap(nodeMap);
  }

  @Override
  public ConcurrentHashMap<SkyKey, InMemoryNodeEntry> getAllValuesMutable() {
    return nodeMap;
  }

  static final class EdgelessInMemoryGraphImpl extends InMemoryGraphImpl {

    @Override
    protected InMemoryNodeEntry newNodeEntry(SkyKey key) {
      return new EdgelessInMemoryNodeEntry();
    }

    @Override
    public boolean storesReverseDeps() {
      return false;
    }
  }
}
