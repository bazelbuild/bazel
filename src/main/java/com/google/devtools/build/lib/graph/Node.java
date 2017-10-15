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
package com.google.devtools.build.lib.graph;

import com.google.common.collect.Sets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

/**
 * <p>A generic directed-graph Node class.  Type parameter T is the type
 * of the node's label.
 *
 * <p>Each node is identified by a label, which is unique within the graph
 * owning the node.
 *
 * <p>Nodes are immutable, that is, their labels cannot be changed.  However,
 * their predecessor/successor lists are mutable.
 *
 * <p>Nodes cannot be created directly by clients.
 *
 * <p>Clients should not confuse nodes belonging to two different graphs!  (Use
 * Digraph.checkNode() to catch such errors.)  There is no way to find the
 * graph to which a node belongs; it is intentionally not represented, to save
 * space.
 */
public final class Node<T> {

  private static final int ARRAYLIST_THRESHOLD = 6;
  private static final int INITIAL_HASHSET_CAPACITY = 12;

  // The succs and preds set representation changes depending on its size.
  // It is implemented using the following collections:
  // - null for size = 0.
  // - Collections$SingletonList for size = 1.
  // - ArrayList(6) for size = [2..6].
  // - HashSet(12) for size > 6.
  // These numbers were chosen based on profiling.

  private final T label;

  /**
   * A duplicate-free collection of edges from this node.  May be null,
   * indicating the empty set.
   */
  private Collection<Node<T>> succs = null;

  /**
   * A duplicate-free collection of edges to this node.  May be null,
   * indicating the empty set.
   */
  private Collection<Node<T>> preds = null;

  private final int hashCode;

  /**
   * Only Digraph.createNode() can call this!
   */
  Node(T label, int hashCode) {
    if (label == null) { throw new NullPointerException("label"); }
    this.label = label;
    this.hashCode = hashCode;
  }

  /**
   * Returns the label for this node.
   */
  public T getLabel() {
    return label;
  }

  /**
   * Returns a duplicate-free collection of the nodes that this node links to.
   */
  public Collection<Node<T>> getSuccessors() {
    return succs == null ? Collections.emptyList() : Collections.unmodifiableCollection(succs);
  }

  /**
   * Equivalent to {@code !getSuccessors().isEmpty()} but possibly more
   * efficient.
   */
  public boolean hasSuccessors() {
    return succs != null;
  }

  /**
   * Equivalent to {@code getSuccessors().size()} but possibly more efficient.
   */
  public int numSuccessors() {
    return succs == null ? 0 : succs.size();
  }

  /**
   * Removes all edges to/from this node.
   * Private: breaks graph invariant!
   */
  void removeAllEdges() {
    this.succs = null;
    this.preds = null;
  }

  /**
   * Returns an (unordered, possibly immutable) set of the nodes that link to
   * this node.
   */
  public Collection<Node<T>> getPredecessors() {
    return preds == null ? Collections.emptyList() : Collections.unmodifiableCollection(preds);
  }

  /**
   * Equivalent to {@code getPredecessors().size()} but possibly more
   * efficient.
   */
  public int numPredecessors() {
    return preds == null ? 0 : preds.size();
  }

  /**
   * Equivalent to {@code !getPredecessors().isEmpty()} but possibly more
   * efficient.
   */
  public boolean hasPredecessors() {
    return preds != null;
  }

  /**
   * Adds 'value' to either the predecessor or successor set, updating the
   * appropriate field as necessary.
   * @return {@code true} if the set was modified; {@code false} if the set
   * was not modified
   */
  private boolean add(boolean predecessorSet, Node<T> value) {
    final Collection<Node<T>> set = predecessorSet ? preds : succs;
    if (set == null) {
      // null -> SingletonList
      return updateField(predecessorSet, Collections.singletonList(value));
    }
    if (set.contains(value)) {
      // already exists in this set
      return false;
    }
    int previousSize = set.size();
    if (previousSize == 1) {
      // SingletonList -> ArrayList
      Collection<Node<T>> newSet = new ArrayList<>(ARRAYLIST_THRESHOLD);
      newSet.addAll(set);
      newSet.add(value);
      return updateField(predecessorSet, newSet);
    } else if (previousSize < ARRAYLIST_THRESHOLD) {
      // ArrayList
      set.add(value);
      return true;
  } else if (previousSize == ARRAYLIST_THRESHOLD) {
      // ArrayList -> HashSet
      Collection<Node<T>> newSet = Sets.newHashSetWithExpectedSize(INITIAL_HASHSET_CAPACITY);
      newSet.addAll(set);
      newSet.add(value);
      return updateField(predecessorSet, newSet);
    } else {
      // HashSet
      set.add(value);
      return true;
    }
  }

  /**
   * Removes 'value' from either 'preds' or 'succs', updating the appropriate
   * field as necessary.
   * @return {@code true} if the set was modified; {@code false} if the set
   * was not modified
   */
  private boolean remove(boolean predecessorSet, Node<T> value) {
    final Collection<Node<T>> set = predecessorSet ? preds : succs;
    if (set == null) {
      // null
      return false;
    }

    int previousSize = set.size();
    if (previousSize == 1) {
      if (set.contains(value)) {
        // -> null
        return updateField(predecessorSet, null);
      } else {
        return false;
      }
    }
    // now remove the value
    if (set.remove(value)) {
      // may need to change representation
      if (previousSize == 2) {
        // -> SingletonList
        List<Node<T>> list =
          Collections.singletonList(set.iterator().next());
        return updateField(predecessorSet, list);

      } else if (previousSize == 1 + ARRAYLIST_THRESHOLD) {
        // -> ArrayList
        Collection<Node<T>> newSet = new ArrayList<>(ARRAYLIST_THRESHOLD);
        newSet.addAll(set);
        return updateField(predecessorSet, newSet);
      }
      return true;
    }
    return false;
  }

  /**
   * Update either the {@link #preds} or {@link #succs} field to point to the
   * new set.
   * @return {@code true}, because the set must have been updated
   */
  private boolean updateField(boolean predecessorSet,
      Collection<Node<T>> newSet) {
    if (predecessorSet) {
      preds = newSet;
    } else {
      succs = newSet;
    }
    return true;
  }


  /**
   * Add 'to' as a successor of 'this' node.  Returns true iff
   * the graph changed.  Private: breaks graph invariant!
   */
  boolean addSuccessor(Node<T> to) {
    return add(false, to);
  }

  /**
   * Add 'from' as a predecessor of 'this' node.  Returns true iff
   * the graph changed.  Private: breaks graph invariant!
   */
  boolean addPredecessor(Node<T> from) {
    return add(true, from);
  }

  /**
   * Remove edge: fromNode.succs = {n | n in fromNode.succs && n != toNode}
   * Private: breaks graph invariant!
   */
  boolean removeSuccessor(Node<T> to) {
    return remove(false, to);
  }

  /**
   * Remove edge: toNode.preds = {n | n in toNode.preds && n != fromNode}
   * Private: breaks graph invariant!
   */
  boolean removePredecessor(Node<T> from) {
    return remove(true, from);
  }

  @Override
  public String toString() {
    return "node:" + label;
  }

  @Override
  public int hashCode() {
    return hashCode; // Fast, deterministic.
  }

  @Override
  public boolean equals(Object that) {
    return this == that; // Nodes are unique for a given label
  }
}
