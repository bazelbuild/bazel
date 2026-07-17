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

import java.io.PrintWriter;

/**
 *  <p> An implementation of GraphVisitor for displaying graphs in dot
 *  format. </p>
 */
public class DotOutputVisitor<T> implements GraphVisitor<T> {

  /**
   * Constructs a dot output visitor.
   *
   * <p>The visitor writes to writer 'out', and rendering node labels as strings using the specified
   * displayer, 'disp'.
   */
  public DotOutputVisitor(PrintWriter out, String lineTerminator, LabelSerializer<T> disp) {
    this.out = out;
    this.lineTerminator = lineTerminator;
    this.disp = disp;
  }

  private final LabelSerializer<T> disp;
  protected final PrintWriter out;
  protected final String lineTerminator;
  private boolean closeAtEnd = false;

  @Override
  public void beginVisit() {
    out.printf("digraph mygraph {%s", lineTerminator);
  }

  @Override
  public void endVisit() {
    out.printf("}%s", lineTerminator);
    out.flush();
    if (closeAtEnd) {
      out.close();
    }
  }

  @Override
  public void visitEdge(Node<T> lhs, Node<T> rhs) {
    out.printf("  \"%s\" -> \"%s\"%s", disp.serialize(lhs), disp.serialize(rhs), lineTerminator);
  }

  @Override
  public void visitNode(Node<T> node) {
    out.printf("  \"%s\"%s", disp.serialize(node), lineTerminator);
  }
}
