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

import com.google.common.base.Preconditions;
import java.util.Collection;

/**
 * A generic directed-graph Node class. Type parameter T is the type of the node's label.
 *
 * <p>Each node is identified by a label, which is unique within the graph owning the node.
 *
 * <p>Nodes are immutable, that is, their labels cannot be changed. However, their
 * predecessor/successor lists are mutable.
 *
 * <p>Nodes cannot be created directly by clients.
 *
 * <p>Clients should not confuse nodes belonging to two different graphs! (Use Digraph.checkNode()
 * to catch such errors.) There is no way to find the graph to which a node belongs; it is
 * intentionally not represented, to save space.
 *
 * <p>During adding or removing edge locks always hold in specific order: first=nodeFrom.succs then
 * second=nodeTo.preds. That's why reordering deadlock never happens.
 */
public final class Node<T> {

  private final T label;

  /** A duplicate-free collection of edges from this node. May be null, indicating the empty set. */
  private final ConcurrentCollectionWrapper<Node<T>> succs = new ConcurrentCollectionWrapper<>();

  /** A duplicate-free collection of edges to this node. May be null, indicating the empty set. */
  private final ConcurrentCollectionWrapper<Node<T>> preds = new ConcurrentCollectionWrapper<>();

  /**
   * Only Digraph.createNode() can call this!
   */
  Node(T label) {
    this.label = Preconditions.checkNotNull(label, "label");
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
    return this.succs.get();
  }

  /**
   * Remove all successors edges and return collection of its. Self edge removed but did not
   * returned in result collection.
   *
   * @return all existed before successor nodes but this.
   */
  Collection<Node<T>> removeAllSuccessors() {
    this.removeEdge(this); // remove self edge
    Collection<Node<T>> successors = this.succs.clear();
    for (Node<T> s : successors) {
      if (!s.removePredecessor(this)) {
        throw new IllegalStateException("inconsistent graph state");
      }
    }
    return successors;
  }

  /**
   * Equivalent to {@code !getSuccessors().isEmpty()} but possibly more
   * efficient.
   */
  public boolean hasSuccessors() {
    return !this.succs.get().isEmpty();
  }

  /**
   * Equivalent to {@code getSuccessors().size()} but possibly more efficient.
   */
  public int numSuccessors() {
    return this.succs.size();
  }

  /**
   * Returns an (unordered, possibly immutable) set of the nodes that link to
   * this node.
   */
  public Collection<Node<T>> getPredecessors() {
    return this.preds.get();
  }

  /**
   * Remove all predecessors edges and return collection of its. Self edge removed but did not
   * returned in result collection.
   *
   * @return all existed before predecessor nodes but this.
   */
  Collection<Node<T>> removeAllPredecessors() {
    this.removeEdge(this); // remove self edge
    Collection<Node<T>> predecessors = this.preds.clear();
    for (Node<T> p : predecessors) {
      if (!p.removeSuccessor(this)) {
        throw new IllegalStateException("inconsistent graph state");
      }
    }
    return predecessors;
  }

  /**
   * Equivalent to {@code !getPredecessors().isEmpty()} but possibly more
   * efficient.
   */
  public boolean hasPredecessors() {
    return !preds.get().isEmpty();
  }

  /** Equivalent to {@code getPredecessors().size()} but possibly more efficient. */
  public int numPredecessors() {
    return this.preds.size();
  }

  /**
   * Adds edge from this node to target
   *
   * <p>In this method one lock held inside another lock. But it can not be reason of reordering
   * deadlock. Lock always holds in direction fromNode.succs -> toNode.preds.
   * @see #removeEdge(Node)
   *
   * @return true if edge had been added. false - otherwise.
   */
  boolean addEdge(Node<T> target) {
    synchronized (succs) {
      boolean isNewSuccessor = this.succs.add(target);
      boolean isNewPredecessor = target.addPredecessor(this);
      if (isNewPredecessor != isNewSuccessor) {
        throw new IllegalStateException("inconsistent graph state");
      }
      return isNewSuccessor;
    }
  }

  /**
   * Adds edge from this node to target
   *
   * <p>In this method one lock held inside another lock. But it can not be reason of reordering
   * deadlock. Lock always holds in direction fromNode.succs -> toNode.preds.
   * @see #addEdge(Node)
   *
   * @return true if edge had been removed. false - otherwise.
   */
  boolean removeEdge(Node<T> target) {
    synchronized (succs) {
      boolean isSuccessorRemoved = this.succs.remove(target);
      if (isSuccessorRemoved) {
        boolean isPredecessorRemoved = target.removePredecessor(this);
        if (!isPredecessorRemoved) {
          throw new IllegalStateException("inconsistent graph state");
        }
        return true;
      }
      return false;
    }
  }

  /**
   * Add 'from' as a predecessor of 'this' node. Returns true iff the graph changed. Private: breaks
   * graph invariant!
   */
  private boolean addPredecessor(Node<T> from) {
    return preds.add(from);
  }

  /**
   * Remove edge: toNode.preds = {n | n in toNode.preds && n != fromNode} Private: breaks graph
   * invariant!
   */
  private boolean removePredecessor(Node<T> from) {
    return preds.remove(from);
  }

  private boolean removeSuccessor(Node<T> to) {
    return succs.remove(to);
  }

  @Override
  public String toString() {
    return "node:" + label;
  }

  @Override
  public int hashCode() {
    return super.hashCode();
  }

  @Override
  public boolean equals(Object that) {
    return this == that; // Nodes are unique for a given label
  }
}
