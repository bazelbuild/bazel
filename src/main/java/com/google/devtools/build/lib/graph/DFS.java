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

import static java.util.Comparator.comparing;
import static java.util.stream.Collectors.toCollection;

import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Set;

/**
 *  <p> The DFS class encapsulates a depth-first search visitation, including
 *  the order in which nodes are to be visited relative to their successors
 *  (PREORDER/POSTORDER), whether the forward or transposed graph is to be
 *  used, and which nodes have been seen already. </p>
 *
 *  <p> A variety of common uses of DFS are offered through methods of
 *  Digraph; however clients can use this class directly for maximum
 *  flexibility.  See the implementation of
 *  Digraph.getStronglyConnectedComponents() for an example. </p>
 *
 *  <p> Clients should not modify the enclosing Digraph instance of a DFS
 *  while a traversal is in progress. </p>
 */
public class DFS<T> {

  // (Preferred over a boolean to avoid parameter confusion.)
  public enum Order {
    PREORDER,
    POSTORDER
  }

  private final Order order; // = (PREORDER|POSTORDER)

  private final Comparator<Node<T>> edgeOrder;

  private final boolean transpose;

  private final Set<Node<T>> marked = CompactHashSet.create();

  /**
   *  Constructs a DFS instance for searching over the enclosing Digraph
   *  instance, using the specified visitation parameters.
   *
   *  @param order PREORDER or POSTORDER, determines node visitation order
   *  @param edgeOrder an ordering in which the edges originating from the same
   *      node should be visited (if null, the order is unspecified)
   *  @param transpose iff true, the graph is implicitly transposed during
   *  visitation.
   */
  public DFS(Order order, final Comparator<? super T> edgeOrder, boolean transpose) {
    this.order = order;
    this.transpose = transpose;

    this.edgeOrder = (edgeOrder == null) ? null : comparing(Node::getLabel, edgeOrder::compare);
  }

  public DFS(Order order, boolean transpose) {
    this(order, null, transpose);
  }

  /**
   *  Returns the (immutable) set of nodes visited so far.
   */
  public Set<Node<T>> getMarked() {
    return Collections.unmodifiableSet(marked);
  }

  public void visit(Node<T> node, GraphVisitor<T> visitor) {
    if (!marked.add(node)) {
      return;
    }

    if (order == Order.PREORDER) {
      visitor.visitNode(node);
    }

    Collection<Node<T>> edgeTargets = transpose
        ? node.getPredecessors() : node.getSuccessors();
    if (edgeOrder != null) {
      List<Node<T>> mutableNodeList =
          edgeTargets.stream().sorted(edgeOrder).collect(toCollection(ArrayList::new));
      edgeTargets = mutableNodeList;
    }

    for (Node<T> v: edgeTargets) {
      visit(v, visitor);
    }

    if (order == Order.POSTORDER) {
      visitor.visitNode(node);
    }
  }
}
