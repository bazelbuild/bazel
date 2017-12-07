// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.query2;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ThreadSafeMutableSet;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/** Utility class for Skyframe-based query implementations. */
class SkyQueryUtils {
  interface GetFwdDeps<T> {
    ThreadSafeMutableSet<T> getFwdDeps(Iterable<T> t) throws InterruptedException;
  }

  static <T> ThreadSafeMutableSet<T> getTransitiveClosure(
      ThreadSafeMutableSet<T> targets, GetFwdDeps<T> getFwdDeps, ThreadSafeMutableSet<T> visited)
      throws InterruptedException {
    ThreadSafeMutableSet<T> current = targets;
    while (!current.isEmpty()) {
      Iterable<T> toVisit =
          current.stream().filter(obj -> !visited.contains(obj)).collect(Collectors.toList());
      current = getFwdDeps.getFwdDeps(toVisit);
      Iterables.addAll(visited, toVisit);
    }
    return visited;
  }

  /**
   * Gets a path from {@code from} to {@code to}, walking the graph revealed by {@code getFwdDeps}.
   *
   * <p>In case the type {@link T} does not implement equality, {@code label} will be used to map
   * elements of type {@link T} to elements of type {@link L} which does implement equality. {@code
   * label} should be an injective function. For instance, if {@link T} is of type {@link Target}
   * then {@link L} could be of type {@link Label} and {@code label} could be {@link
   * Target::getLabel}.
   *
   * <p>Implemented with a breadth-first search.
   */
  static <T, L> ImmutableList<T> getNodesOnPath(
      T from, T to, GetFwdDeps<T> getFwdDeps, Function<T, L> label) throws InterruptedException {
    // Tree of nodes visited so far.
    Map<L, L> nodeToParent = new HashMap<>();
    Map<L, T> labelToTarget = new HashMap<>();
    // Contains all nodes left to visit in a (LIFO) stack.
    Deque<T> toVisit = new ArrayDeque<>();
    toVisit.add(from);
    nodeToParent.put(label.apply(from), null);
    labelToTarget.put(label.apply(from), from);
    while (!toVisit.isEmpty()) {
      T current = toVisit.removeFirst();
      if (label.apply(to).equals(label.apply(current))) {
        List<L> labelPath = Digraph.getPathToTreeNode(nodeToParent, label.apply(to));
        ImmutableList.Builder<T> targetPathBuilder = ImmutableList.builder();
        for (L item : labelPath) {
          targetPathBuilder.add(Preconditions.checkNotNull(labelToTarget.get(item), item));
        }
        return targetPathBuilder.build();
      }
      for (T dep : getFwdDeps.getFwdDeps(ImmutableList.of(current))) {
        L depLabel = label.apply(dep);
        if (!nodeToParent.containsKey(depLabel)) {
          nodeToParent.put(depLabel, label.apply(current));
          labelToTarget.put(depLabel, dep);
          toVisit.addFirst(dep);
        }
      }
    }
    // Note that the only current caller of this method checks first to see if there is a path
    // before calling this method. It is not clear what the return value should be here.
    return null;
  }
}
