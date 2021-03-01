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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.util.Pair;
import java.util.Map;
import java.util.Set;
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
   * Returns the value of the given key, or {@code null} if it has no value due to an error during
   * its computation or it is not done in the graph.
   *
   * <p>A node that is done in the graph must have either a non-null getValue, a non-null {@link
   * #getException}, or a true {@link #isCycle}.
   *
   * <p>These three methods should all be reading the same {@link
   * NodeEntry#getValueMaybeWithMetadata} value internally, so once that value is indirectly
   * retrieved via one of these methods, the others can read it for free. This is relevant for graph
   * implementations that may throw an {@link InterruptedException} on retrieving entries and value.
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
   * was computed successfully, depends on a cycle without any other error, or is not done in the
   * graph, returns null.
   */
  @Nullable
  Exception getException(SkyKey key) throws InterruptedException;

  /**
   * Returns true if the node with the given {@code key} depends on a cycle. Returns false if the
   * node does not depend on a cycle, or is not done in the graph.
   */
  boolean isCycle(SkyKey key) throws InterruptedException;

  /**
   * Returns a map giving the direct dependencies of the nodes with the given keys. A node for each
   * given key must be done in the graph if it exists.
   */
  Map<SkyKey, Iterable<SkyKey>> getDirectDeps(Iterable<SkyKey> keys) throws InterruptedException;

  /**
   * Returns the direct dependencies of the node with the given key. A node for that key must exist
   * in the graph and be done.
   */
  Iterable<SkyKey> getDirectDeps(SkyKey key) throws InterruptedException;

  /**
   * Returns a map giving the reverse dependencies of the nodes with the given keys. A node for each
   * given key must be done in the graph if it exists.
   */
  Map<SkyKey, Iterable<SkyKey>> getReverseDeps(Iterable<? extends SkyKey> keys)
      throws InterruptedException;

  /**
   * Returns a map giving the reverse dependencies of the nodes with the given keys as well as the
   * value
   */
  Map<SkyKey, Pair<SkyValue, Iterable<SkyKey>>> getValueAndRdeps(Iterable<SkyKey> keys)
      throws InterruptedException;

  default ImmutableSet<SkyKey> getAllKeysForTesting() {
    throw new UnsupportedOperationException();
  }

  /** Provides a WalkableGraph on demand after preparing it. */
  interface WalkableGraphFactory {
    EvaluationResult<SkyValue> prepareAndGet(Set<SkyKey> roots, EvaluationContext evaluationContext)
        throws InterruptedException;
  }
}
