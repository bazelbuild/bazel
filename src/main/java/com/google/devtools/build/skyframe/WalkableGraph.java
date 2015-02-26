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

import com.google.devtools.build.lib.events.EventHandler;

import java.util.Collection;

import javax.annotation.Nullable;

/**
 * Read-only graph that exposes the dependents, dependencies (reverse dependents), and value and
 * exception (if any) of a given node.
 */
public interface WalkableGraph {

  /**
   * Returns whether the given key exists as a done node in the graph. If there is a chance that the
   * given node does not exist, this method should be called before any others, since the others
   * throw a {@link RuntimeException} on failure to access a node.
   */
  boolean exists(SkyKey key);

  /**
   * Returns the value of the given key, or {@code null} if it has no value due to an error during
   * its computation. A node with this key must exist in the graph.
   */
  @Nullable
  SkyValue getValue(SkyKey key);

  /**
   * Returns the exception thrown when computing the node with the given key, if any. If the node
   * was computed successfully, returns null. A node with this key must exist in the graph.
   */
  @Nullable Exception getException(SkyKey key);

  /**
   * Returns the direct dependencies of the node with the given key. A node with this key must exist
   * in the graph.
   */
  Iterable<SkyKey> getDirectDeps(SkyKey key);

  /**
   * Returns the reverse dependencies of the node with the given key. A node with this key must
   * exist in the graph.
   */
  Iterable<SkyKey> getReverseDeps(SkyKey key);

  /** Provides a WalkableGraph on demand after preparing it. */
  interface WalkableGraphFactory {
    WalkableGraph prepareAndGet(Collection<String> roots, int numThreads,
        EventHandler eventHandler) throws InterruptedException;
  }
}
