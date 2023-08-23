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

import com.google.devtools.build.skyframe.InMemoryGraphImpl.EdgelessInMemoryGraphImpl;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Collection;
import java.util.Map;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/** {@link ProcessableGraph} that exposes the contents of the entire graph. */
public interface InMemoryGraph extends ProcessableGraph {

  /** Creates a new in-memory graph suitable for incremental builds. */
  static InMemoryGraph create() {
    return new InMemoryGraphImpl(/* usePooledInterning= */ true);
  }

  static InMemoryGraph create(boolean usePooledInterning) {
    return new InMemoryGraphImpl(usePooledInterning);
  }

  /**
   * Creates a new in-memory graph that discards graph edges to save memory and cannot be used for
   * incremental builds.
   */
  static InMemoryGraph createEdgeless(boolean usePooledInterning) {
    return new EdgelessInMemoryGraphImpl(usePooledInterning);
  }

  @Override
  @CanIgnoreReturnValue
  NodeBatch createIfAbsentBatch(
      @Nullable SkyKey requestor, Reason reason, Iterable<? extends SkyKey> keys);

  @Nullable
  @Override
  NodeEntry get(@Nullable SkyKey requestor, Reason reason, SkyKey key);

  @Override
  default NodeBatch getBatch(
      @Nullable SkyKey requestor, Reason reason, Iterable<? extends SkyKey> keys) {
    return getBatchMap(requestor, reason, keys)::get;
  }

  @Override
  default LookupHint getLookupHint(SkyKey key) {
    return LookupHint.INDIVIDUAL;
  }

  @Override
  Map<SkyKey, ? extends NodeEntry> getBatchMap(
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
  Map<SkyKey, SkyValue> getDoneValues();

  /** Returns an unmodifiable, live view of all nodes in the graph. */
  Collection<InMemoryNodeEntry> getAllNodeEntries();

  /** Applies the given consumer to each node in the graph, potentially in parallel. */
  void parallelForEach(Consumer<InMemoryNodeEntry> consumer);

  /**
   * Removes the node entry associated with the given {@link SkyKey} from the graph if it is done.
   */
  void removeIfDone(SkyKey key);

  /**
   * Cleans up {@linkplain com.google.devtools.build.lib.concurrent.PooledInterner.Pool interning
   * pools} by moving objects to weak interners and uninstalling the current pools.
   *
   * <p>May destroy this graph. Only call when the graph is about to be thrown away.
   */
  void cleanupInterningPools();

  /**
   * Returns the {@link InMemoryNodeEntry} for a given {@link SkyKey} if present in the graph.
   * Otherwise, returns null.
   */
  @Nullable
  InMemoryNodeEntry getIfPresent(SkyKey key);
}
