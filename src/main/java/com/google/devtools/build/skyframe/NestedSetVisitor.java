// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.Order;

import java.util.Set;

/**
 * NestedSetVisitor facilitates a transitive visitation over a NestedSet, which must be in STABLE
 * order. The callback may be called from multiple threads, and must be thread-safe.
 *
 * <p>The visitation is iterative: The caller may invoke a NestedSet within the top-level NestedSet
 * in any order.
 *
 * <p>Currently this class is only used in Skyframe to facilitate iterative replay of transitive
 * warnings/errors.
 *
 * @param <E> the data type
 */
// @ThreadSafety.ThreadSafe
final class NestedSetVisitor<E> {
  interface Receiver<E> {
    void accept(E arg);
  }

  private final Receiver<E> callback;

  private final VisitedState<E> visited;

  public NestedSetVisitor(Receiver<E> callback, VisitedState<E> visited) {
    this.callback = Preconditions.checkNotNull(callback);
    this.visited = Preconditions.checkNotNull(visited);
  }

  /**
   * Transitively visit a nested set.
   *
   * @param nestedSet the nested set to visit transitively.
   */
  public void visit(NestedSet<E> nestedSet) {
    Preconditions.checkArgument(nestedSet.getOrder() == Order.STABLE_ORDER);
    if (!visited.add(nestedSet)) {
      return;
    }

    for (NestedSet<E> subset : nestedSet.transitiveSets()) {
      visit(subset);
    }
    for (E member : nestedSet.directMembers()) {
      if (visited.add(member)) {
        callback.accept(member);
      }
    }
  }

  static class VisitedState<E> {
    private final Set<NestedSet<E>> seenSets = Sets.newConcurrentHashSet();
    private final Set<E> seenNodes = Sets.newConcurrentHashSet();

    public void clear() {
      seenSets.clear();
      seenNodes.clear();
    }

    private boolean add(E node) {
      return seenNodes.add(node);
    }

    private boolean add(NestedSet<E> set) {
      return seenSets.add(set);
    }
  }
}
