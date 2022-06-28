// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.Maps;
import com.google.devtools.build.skyframe.InMemoryGraphImpl.EdgelessInMemoryGraphImpl;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;

/** {@link ProcessableGraph} that exposes the contents of the entire graph. */
public interface InMemoryGraph extends ProcessableGraph {

  /** Creates a new in-memory graph suitable for incremental builds. */
  static InMemoryGraph create() {
    return new InMemoryGraphImpl();
  }

  /**
   * Creates a new in-memory graph that discards graph edges to save memory and cannot be used for
   * incremental builds.
   */
  static InMemoryGraph createEdgeless() {
    return new EdgelessInMemoryGraphImpl();
  }

  @Override
  Map<SkyKey, ? extends NodeEntry> createIfAbsentBatch(
      @Nullable SkyKey requestor, Reason reason, Iterable<? extends SkyKey> keys);

  @Nullable
  @Override
  NodeEntry get(@Nullable SkyKey requestor, Reason reason, SkyKey key);

  @Override
  Map<SkyKey, ? extends NodeEntry> getBatch(
      @Nullable SkyKey requestor, Reason reason, Iterable<? extends SkyKey> keys);

  /**
   * Returns a read-only live view of the nodes in the graph. All node are included. Dirty values
   * include their Node value. Values in error have a null value.
   */
  Map<SkyKey, SkyValue> getValues();

  default int valuesSize() {
    return getValues().size();
  }

  /**
   * Returns a read-only live view of the done values in the graph. Dirty, changed, and error values
   * are not present in the returned map
   */
  default Map<SkyKey, SkyValue> getDoneValues() {
    return transformDoneEntries(getAllValuesMutable());
  }

  // Only for use by MemoizingEvaluator#delete
  Map<SkyKey, InMemoryNodeEntry> getAllValues();

  ConcurrentHashMap<SkyKey, InMemoryNodeEntry> getAllValuesMutable();

  static Map<SkyKey, SkyValue> transformDoneEntries(Map<SkyKey, InMemoryNodeEntry> nodeMap) {
    return Collections.unmodifiableMap(
        Maps.filterValues(
            Maps.transformValues(nodeMap, entry -> entry.isDone() ? entry.getValue() : null),
            Objects::nonNull));
  }
}
