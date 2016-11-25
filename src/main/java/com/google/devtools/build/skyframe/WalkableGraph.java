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

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import java.util.Collection;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Read-only graph that exposes the dependents, dependencies (reverse dependents), and value and
 * exception (if any) of a given node.
 *
 * <p>Certain graph implementations can throw {@link InterruptedException} when trying to retrieve
 * node entries. Such exceptions should not be caught locally -- they should be allowed to propagate
 * up.
 */
@ThreadSafe
public interface WalkableGraph {

  /**
   * Returns whether the given key exists as a done node in the graph. If there is a chance that the
   * given node does not exist, this method should be called before any others, since the others
   * throw a {@link RuntimeException} on failure to access a node.
   */
  boolean exists(SkyKey key) throws InterruptedException;

  /**
   * Returns the value of the given key, or {@code null} if it has no value due to an error during
   * its computation. A node with this key must exist in the graph.
   */
  @Nullable
  SkyValue getValue(SkyKey key) throws InterruptedException;

  /**
   * Returns a map giving the values of the given keys for done keys that were successfully
   * computed. Or in other words, it filters out non-existent nodes, pending nodes and nodes that
   * produced an exception.
   */
  Map<SkyKey, SkyValue> getSuccessfulValues(Iterable<SkyKey> keys) throws InterruptedException;

  /**
   * Returns a map giving exceptions associated to the given keys for done keys. Keys not present in
   * the graph or whose nodes are not done will be present in the returned map, with null value. In
   * other words, if {@code key} is in {@param keys}, then the returned map will contain an entry
   * for {@code key} if and only if the node for {@code key} did <i>not</i> evaluate successfully
   * without error.
   */
  Map<SkyKey, Exception> getMissingAndExceptions(Iterable<SkyKey> keys) throws InterruptedException;

  /**
   * Returns the exception thrown when computing the node with the given key, if any. If the node
   * was computed successfully, returns null. A node with this key must exist and be done in the
   * graph.
   */
  @Nullable
  Exception getException(SkyKey key) throws InterruptedException;

  /**
   * Returns a map giving the direct dependencies of the nodes with the given keys. A node for each
   * given key must exist and be done in the graph.
   */
  Map<SkyKey, Iterable<SkyKey>> getDirectDeps(Iterable<SkyKey> keys) throws InterruptedException;

  /**
   * Returns a map giving the reverse dependencies of the nodes with the given keys. A node for each
   * given key must exist and be done in the graph.
   */
  Map<SkyKey, Iterable<SkyKey>> getReverseDeps(Iterable<SkyKey> keys) throws InterruptedException;

  /**
   * Examines all the given keys. Returns an iterable of keys whose corresponding nodes are
   * currently available to be fetched.
   *
   * <p>Note: An unavailable node does not mean it is not in the graph. It only means it's not ready
   * to be fetched immediately.
   */
  Iterable<SkyKey> getCurrentlyAvailableNodes(Iterable<SkyKey> keys, Reason reason);

  /** Provides a WalkableGraph on demand after preparing it. */
  interface WalkableGraphFactory {
    EvaluationResult<SkyValue> prepareAndGet(
        SkyKey universeKey, int numThreads, EventHandler eventHandler) throws InterruptedException;

    /**
     * Returns true if this instance has already been used to {@link #prepareAndGet} {@code
     * universeKey}. If so, cached results from {@link #prepareAndGet} can be re-used safely,
     * potentially saving some processing time.
     */
    boolean isUpToDate(SkyKey universeKey);

    /** Returns the {@link SkyKey} that defines this universe. */
    SkyKey getUniverseKey(Collection<String> roots, String offset);
  }
}
