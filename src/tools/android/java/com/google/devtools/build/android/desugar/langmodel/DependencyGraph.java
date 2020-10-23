/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *    http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.langmodel;

import com.google.common.collect.ImmutableSet;
import java.util.ArrayDeque;
import java.util.HashSet;
import java.util.Queue;
import java.util.Set;

/** Encloses a graph-based utilities. */
public final class DependencyGraph {

  /** Searches and obtains all nodes that are reachable from the given {@code roots}. */
  public static <N extends Node<N>> ImmutableSet<N> findAllReachableNodes(ImmutableSet<N> roots) {
    // BFS implementation of reachable node searching.
    Set<N> discoveredNodes = new HashSet<>(roots);
    Queue<N> workingNodes = new ArrayDeque<>(roots);
    while (!workingNodes.isEmpty()) {
      N node = workingNodes.remove();
      for (N child : node.getAllChildren()) {
        if (discoveredNodes.add(child)) {
          workingNodes.add(child);
        }
      }
    }
    return ImmutableSet.copyOf(discoveredNodes);
  }

  /** Represents a node in a directed graph, possibly cyclic or acyclic. */
  public interface Node<T extends Node<T>> {
    Set<T> getAllChildren();
  }

  private DependencyGraph() {}
}
