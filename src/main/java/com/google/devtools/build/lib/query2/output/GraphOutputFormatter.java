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
package com.google.devtools.build.lib.query2.output;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.collect.EquivalenceRelation;
import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.graph.DotOutputVisitor;
import com.google.devtools.build.lib.graph.LabelSerializer;
import com.google.devtools.build.lib.graph.Node;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.output.QueryOptions.OrderOutput;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * An output formatter that prints the result as factored graph in AT&amp;T
 * GraphViz format.
 */
class GraphOutputFormatter extends OutputFormatter {

  private int graphNodeStringLimit;

  @Override
  public String getName() {
    return "graph";
  }

  @Override
  public void output(QueryOptions options, Digraph<Target> result, OutputStream out,
      AspectResolver aspectProvider) {
    this.graphNodeStringLimit = options.graphNodeStringLimit;

    boolean sortLabels = options.orderOutput == OrderOutput.FULL;
    if (options.graphFactored) {
      outputFactored(result, new PrintWriter(out), sortLabels);
    } else {
      outputUnfactored(result, new PrintWriter(out), sortLabels, options);
    }
  }

  private void outputUnfactored(
      Digraph<Target> result, PrintWriter out, boolean sortLabels, final QueryOptions options) {
    result.visitNodesBeforeEdges(
        new DotOutputVisitor<Target>(out, LABEL_STRINGIFIER) {
          @Override
          public void beginVisit() {
            super.beginVisit();
            // TODO(bazel-team): (2009) make this the default in Digraph.
            out.printf("  node [shape=box];%s", options.getLineTerminator());
          }
        },
        sortLabels ? new TargetOrdering() : null);
  }

  private static final Ordering<Node<Target>> NODE_COMPARATOR =
      Ordering.from(new TargetOrdering()).onResultOf(EXTRACT_NODE_LABEL);

  private static final Comparator<Iterable<Node<Target>>> ITERABLE_COMPARATOR =
      NODE_COMPARATOR.lexicographical();

  /**
   * Given {@code collectionOfUnorderedSets}, a collection of sets of nodes, returns a collection
   * of sets with the same elements as {@code collectionOfUnorderedSets} but with a stable
   * iteration order within each set given by the target ordering, and the collection ordered by the
   * same induced order.
   */
  private static Collection<Set<Node<Target>>> orderPartition(
      Collection<Set<Node<Target>>> collectionOfUnorderedSets) {
    List<Set<Node<Target>>> result = new ArrayList<>();
    for (Set<Node<Target>> part : collectionOfUnorderedSets) {
      List<Node<Target>> toSort = new ArrayList<>(part);
      Collections.sort(toSort, NODE_COMPARATOR);
      result.add(ImmutableSet.copyOf(toSort));
    }
    Collections.sort(result, ITERABLE_COMPARATOR);
    return result;
  }

  private void outputFactored(Digraph<Target> result, PrintWriter out, final boolean sortLabels) {
    EquivalenceRelation<Node<Target>> equivalenceRelation = createEquivalenceRelation();

    Collection<Set<Node<Target>>> partition =
        CollectionUtils.partition(result.getNodes(), equivalenceRelation);
    if (sortLabels) {
      partition = orderPartition(partition);
    }

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
        },
        sortLabels ? ITERABLE_COMPARATOR : null);
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
