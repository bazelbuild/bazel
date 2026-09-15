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
package com.google.devtools.build.lib.collect.nestedset;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.collect.ConcurrentIdentitySet;
import com.google.devtools.build.lib.collect.nestedset.NestedSet.VisitedArraySet;
import java.util.Collection;
import java.util.function.Predicate;

/**
 * NestedSetVisitor facilitates a transitive visitation over a NestedSet. The callback may be called
 * from multiple threads, and must be thread-safe.
 *
 * <p>The visitation is iterative: The caller may invoke a NestedSet within the top-level NestedSet
 * in any order.
 *
 * @param <E> the data type
 */
public final class NestedSetVisitor<E> {

  /**
   * For each element of the NestedSet the {@code Receiver} will receive one element during the
   * visitation.
   */
  public interface Receiver<E> {
    void accept(E arg);
  }

  private final Receiver<E> callback;

  private final VisitedState<E> visited;

  public NestedSetVisitor(Receiver<E> callback, VisitedState<E> visited) {
    this.callback = checkNotNull(callback);
    this.visited = checkNotNull(visited);
  }

  /**
   * Transitively visit a nested set.
   *
   * @param nestedSet the nested set to visit transitively.
   */
  public void visit(NestedSet<E> nestedSet) throws InterruptedException {
    // We can short-circuit empty nested set visitation here, avoiding load on the shared map
    // VisitedState#seenNodes.
    if (!nestedSet.isEmpty()) {
      visitRaw(nestedSet.getChildrenInterruptibly());
    }
  }

  /** Visit every entry in a collection. */
  public void visit(Collection<E> collection) {
    for (E e : collection) {
      if (visited.needToVisitLeaf.test(e)) {
        callback.accept(e);
      }
    }
  }

  private void visitRaw(Object node) {
    if (node instanceof Object[] array) {
      if (visited.needToVisitNonLeaf.test(array)) {
        for (Object child : array) {
          visitRaw(child);
        }
      }
    } else {
      @SuppressWarnings("unchecked") // It's not an Object[] so must be a leaf.
      E leaf = (E) node;
      if (visited.needToVisitLeaf.test(leaf)) {
        callback.accept(leaf);
      }
    }
  }

  /** Allows {@link NestedSetVisitor} to keep track of the seen nodes and transitive sets. */
  public static final class VisitedState<E> {

    /** Creates a new visited state with the given predicate of whether to visit leaves. */
    public static <E> VisitedState<E> create(Predicate<E> needToVisitLeaf) {
      return new VisitedState<>(new VisitedArraySet()::add, needToVisitLeaf);
    }

    /**
     * Creates a new thread-safe visited state with the given predicate of whether to visit leaves.
     */
    public static <E> VisitedState<E> createConcurrent(Predicate<E> needToVisitLeaf) {
      return new VisitedState<>(
          new ConcurrentIdentitySet(/* sizeHint= */ 1024)::add, needToVisitLeaf);
    }

    private final Predicate<Object[]> needToVisitNonLeaf;
    private final Predicate<E> needToVisitLeaf;

    private VisitedState(Predicate<Object[]> needToVisitNonLeaf, Predicate<E> needToVisitLeaf) {
      this.needToVisitNonLeaf = checkNotNull(needToVisitNonLeaf);
      this.needToVisitLeaf = checkNotNull(needToVisitLeaf);
    }
  }
}
