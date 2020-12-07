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
package com.google.devtools.build.lib.actions;

import java.util.HashMap;
import java.util.Map;

/**
 * A visitor helper class for bipartite graphs. The alternate kinds of nodes are arbitrarily
 * designated "black" or "white".
 *
 * <p>Subclasses implement the black() and white() hook functions which are called as nodes are
 * visited. The class holds a mapping from each node to a small integer; this is available to
 * subclasses if they wish.
 */
abstract class BipartiteVisitor<BLACK, WHITE> {

  protected BipartiteVisitor() {}

  private int nextNodeId = 0;

  // Maps each visited black node to a small integer.
  protected final Map<BLACK, Integer> visitedBlackNodes = new HashMap<>();

  // Maps each visited white node to a small integer.
  protected final Map<WHITE, Integer> visitedWhiteNodes = new HashMap<>();

  /**
   * Visit the specified black node. If this node has not already been visited, the black() hook is
   * called and true is returned; otherwise, false is returned.
   */
  protected final boolean visitBlackNode(BLACK blackNode) throws InterruptedException {
    if (blackNode == null) { throw new NullPointerException(); }
    if (!visitedBlackNodes.containsKey(blackNode)) {
      visitedBlackNodes.put(blackNode, nextNodeId++);
      black(blackNode);
      return true;
    }
    return false;
  }

  /** Visit all specified black nodes. */
  protected final void visitBlackNodes(Iterable<BLACK> blackNodes) throws InterruptedException {
    for (BLACK blackNode : blackNodes) {
      visitBlackNode(blackNode);
    }
  }

  /**
   * Visit the specified white node. If this node has not already been visited, the white() hook is
   * called and true is returned; otherwise, false is returned.
   */
  protected final boolean visitWhiteNode(WHITE whiteNode) throws InterruptedException {
    if (whiteNode == null) {
      throw new NullPointerException();
    }
    if (!visitedWhiteNodes.containsKey(whiteNode)) {
      visitedWhiteNodes.put(whiteNode, nextNodeId++);
      white(whiteNode);
      return true;
    }
    return false;
  }

  /** Visit all specified white nodes. */
  public final void visitWhiteNodes(Iterable<WHITE> whiteNodes) throws InterruptedException {
    for (WHITE whiteNode : whiteNodes) {
      visitWhiteNode(whiteNode);
    }
  }

  /** Called whenever a white node is visited. Hook for subclasses. */
  protected abstract void white(WHITE whiteNode) throws InterruptedException;

  /** Called whenever a black node is visited. Hook for subclasses. */
  protected abstract void black(BLACK blackNode) throws InterruptedException;
}
