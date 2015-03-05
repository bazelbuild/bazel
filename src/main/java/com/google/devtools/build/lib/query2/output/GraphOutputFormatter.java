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
package com.google.devtools.build.lib.query2.output;

import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.collect.EquivalenceRelation;
import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.graph.DotOutputVisitor;
import com.google.devtools.build.lib.graph.LabelSerializer;
import com.google.devtools.build.lib.graph.Node;
import com.google.devtools.build.lib.packages.Target;

import java.io.PrintStream;
import java.io.PrintWriter;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

/**
 * An output formatter that prints the result as factored graph in AT&amp;T
 * GraphViz format.
 */
class GraphOutputFormatter extends OutputFormatter {

  private int graphNodeStringLimit;
  private boolean graphFactored;

  @Override
  public String getName() {
    return "graph";
  }

  @Override
  public void output(QueryOptions options, Digraph<Target> result, PrintStream out) {
    this.graphNodeStringLimit = options.graphNodeStringLimit;
    this.graphFactored = options.graphFactored;

    if (graphFactored) {
      outputFactored(result, new PrintWriter(out));
    } else {
      outputUnfactored(result, new PrintWriter(out));
    }
  }

  private void outputUnfactored(Digraph<Target> result, PrintWriter out) {
    result.visitNodesBeforeEdges(
        new DotOutputVisitor<Target>(out, LABEL_STRINGIFIER) {
          @Override
          public void beginVisit() {
            super.beginVisit();
            // TODO(bazel-team): (2009) make this the default in Digraph.
            out.println("  node [shape=box];");
          }
        });
  }

  private void outputFactored(Digraph<Target> result, PrintWriter out) {
    EquivalenceRelation<Node<Target>> equivalenceRelation = createEquivalenceRelation();

    // Notes on ordering:
    // - Digraph.getNodes() returns nodes in no particular order
    // - CollectionUtils.partition inserts elements into unordered sets
    // This means partitions may contain nodes in a different order than perhaps expected.
    // Example (package //foo):
    //   some_rule(
    //       name = 'foo',
    //       srcs = ['a', 'b', 'c'],
    //   )
    // Querying for deps('foo') will return (among others) the 'foo' node with successors 'a', 'b'
    // and 'c' (in this order), however when asking the Digraph for all of its nodes, the returned
    // collection may be ordered differently.
    Collection<Set<Node<Target>>> partition =
        CollectionUtils.partition(result.getNodes(), equivalenceRelation);

    Digraph<Set<Node<Target>>> factoredGraph = result.createImageUnderPartition(partition);

    // Concatenate the labels of all topologically-equivalent nodes.
    LabelSerializer<Set<Node<Target>>> labelSerializer = new LabelSerializer<Set<Node<Target>>>() {
      @Override
      public String serialize(Node<Set<Node<Target>>> node) {
        int actualLimit = graphNodeStringLimit - RESERVED_LABEL_CHARS;
        boolean firstItem = true;
        StringBuilder buf = new StringBuilder();
        int count = 0;
        for (Node<Target> eqNode : node.getLabel()) {
          String labelString = eqNode.getLabel().getLabel().toString();
          if (!firstItem) {
            buf.append("\\n");

            // Use -1 to denote no limit, as it is easier than trying to pass MAX_INT on the cmdline
            if (graphNodeStringLimit != -1 && (buf.length() + labelString.length() > actualLimit)) {
              buf.append("...and ");
              buf.append(node.getLabel().size() - count);
              buf.append(" more items");
              break;
            }
          }

          buf.append(labelString);
          count++;
          firstItem = false;
        }
        return buf.toString();
      }
    };

    factoredGraph.visitNodesBeforeEdges(
        new DotOutputVisitor<Set<Node<Target>>>(out, labelSerializer) {
          @Override
          public void beginVisit() {
            super.beginVisit();
            // TODO(bazel-team): (2009) make this the default in Digraph.
            out.println("  node [shape=box];");
          }
        });
  }

  /**
   * Returns an equivalence relation for nodes in the specified graph.
   *
   * <p>Two nodes are considered equal iff they have equal topology (predecessors and successors).
   *
   * TODO(bazel-team): Make this a method of Digraph.
   */
  private static <LABEL> EquivalenceRelation<Node<LABEL>> createEquivalenceRelation() {
    return new EquivalenceRelation<Node<LABEL>>() {
      @Override
      public int compare(Node<LABEL> x, Node<LABEL> y) {
        if (x == y) {
          return 0;
        }

        if (x.numPredecessors() != y.numPredecessors()
            || x.numSuccessors() != y.numSuccessors()) {
          return -1;
        }

        Set<Node<LABEL>> xpred = new HashSet<>(x.getPredecessors());
        Set<Node<LABEL>> ypred = new HashSet<>(y.getPredecessors());
        if (!xpred.equals(ypred)) {
          return -1;
        }

        Set<Node<LABEL>> xsucc = new HashSet<>(x.getSuccessors());
        Set<Node<LABEL>> ysucc = new HashSet<>(y.getSuccessors());
        if (!xsucc.equals(ysucc)) {
          return -1;
        }

        return 0;
      }
    };
  }

  private static final int RESERVED_LABEL_CHARS = "\\n...and 9999999 more items".length();

  private static final LabelSerializer<Target> LABEL_STRINGIFIER = new LabelSerializer<Target>() {
    @Override
    public String serialize(Node<Target> node) {
      return node.getLabel().getLabel().toString();
    }
  };
}
