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
// All Rights Reserved.

package com.google.devtools.build.lib.graph;

/**
 *  <p> An graph visitor interface; particularly useful for allowing subclasses
 *  to specify how to output a graph.  The order in which node and edge
 *  callbacks are made (DFS, BFS, etc) is defined by the choice of Digraph
 *  visitation method used.  </p>
 */
public interface GraphVisitor<T> {

  /**
   *  Called before visitation commences.
   */
  void beginVisit();

  /**
   *  Called after visitation is complete.
   */
  void endVisit();

  /**
   *  <p> Called for each edge. </p>
   *
   *  TODO(bazel-team): This method is not essential, and in all known cases so
   *  far, the visitEdge code can always be placed within visitNode.  Perhaps
   *  we should remove it, and the begin/end methods, and make this just a
   *  NodeVisitor?  Are there any algorithms for which edge-visitation order is
   *  important?
   */
  void visitEdge(Node<T> lhs, Node<T> rhs);
  /**
   *  Called for each node.
   */
  void visitNode(Node<T> node);
}
