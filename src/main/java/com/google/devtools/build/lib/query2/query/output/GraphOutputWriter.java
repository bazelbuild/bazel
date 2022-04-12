// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.query.output;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.collect.EquivalenceRelation;
import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.graph.DotOutputVisitor;
import com.google.devtools.build.lib.graph.LabelSerializer;
import com.google.devtools.build.lib.graph.Node;
import com.google.devtools.build.lib.packages.Target;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Generic logic for writing query expression results to <a
 * href="http://graphviz.org/doc/info/lang.html">GraphViz</a> format.
 *
 * <p>This can be used by any query implementation that can provide results as a {@link Digraph}.
 */
public final class GraphOutputWriter<T> {
  /** Interface for reading the contents of a {@link Digraph} {@link Node}. */
  public interface NodeReader<T> {
    /**
     * Returns the label to associate with a GraphViz node.
     *
     * <p>This is not the same as a build {@link Label}. This is just the text associated with a
     * node in a GraphViz graph.
     */
    String getLabel(Node<T> node);

    /** Returns a comparator for the build graph nodes that form the payloads of GraphViz nodes. */
    Comparator<T> comparator();
  }

  private final NodeReader<T> nodeReader;
  private final String lineTerminator;
  private final boolean sortLabels;
  private final int maxLabelSize;
  private final int maxConditionalEdges;
  private final boolean mergeEquivalentNodes;
  private final Ordering<Node<T>> nodeComparator;

  private static final int RESERVED_LABEL_CHARS = "\\n...and 9999999 more items".length();

  /**
   * Constructors a new writer.
   *
   * @param nodeReader {@link NodeReader} for reading node content
   * @param lineTerminator line string terminator
   * @param sortLabels if true, output nodes in sorted order with {@link NodeReader#comparator})
   * @param maxLabelSize maximum characters in label output. Longer labels are truncated. -1 means
   *     no limit.
   * @param maxConditionalEdges maximum number of {@code select() conditional labels} to show on
   *     each edge. -1 means no limit. 0 means no labels.
   * @param mergeEquivalentNodes if true, topologically equivalent nodes are merged together as
   *     multiple labels in the same node. This condenses the graph. For example, given graph {@code
   *     (nodes=[A, B, C], edges=[A->B, A->C]) }, the output has two nodes: "A" and "B,C".
   */
  public GraphOutputWriter(
      NodeReader<T> nodeReader,
      String lineTerminator,
      boolean sortLabels,
      int maxLabelSize,
      int maxConditionalEdges,
      boolean mergeEquivalentNodes) {
    this.nodeReader = nodeReader;
    this.lineTerminator = lineTerminator;
    this.sortLabels = sortLabels;
    this.maxLabelSize = maxLabelSize;
    this.maxConditionalEdges = maxConditionalEdges;
    this.mergeEquivalentNodes = mergeEquivalentNodes;
    nodeComparator = Ordering.from(nodeReader.comparator()).onResultOf(Node::getLabel);
  }

  /**
   * Writes the given graph.
   *
   * @param graph build graph to write
   * @param conditionalEdges edges corresponding to select()s (see {@link ConditionalEdges})
   * @param out output stream to write to
   */
  public void write(
      Digraph<T> graph, @Nullable ConditionalEdges conditionalEdges, OutputStream out) {
    PrintWriter printWriter = new PrintWriter(new OutputStreamWriter(out, UTF_8));
    if (mergeEquivalentNodes) {
      outputFactored(graph, conditionalEdges, printWriter);
    } else {
      outputUnfactored(graph, conditionalEdges, printWriter);
    }
  }

  private void outputUnfactored(
      Digraph<T> graph, @Nullable ConditionalEdges conditionalEdges, PrintWriter out) {
    graph.visitNodesBeforeEdges(
        new DotOutputVisitor<T>(out, nodeReader::getLabel) {
          @Override
          public void beginVisit() {
            super.beginVisit();
            // TODO(bazel-team): (2009) make this the default in Digraph.
            out.printf("  node [shape=box];%s", lineTerminator);
          }

          @Override
          public void visitEdge(Node<T> lhs, Node<T> rhs) {
            super.visitEdge(lhs, rhs);
            String outputLabel =
                getConditionsGraphLabel(
                    ImmutableSet.of(lhs), ImmutableSet.of(rhs), conditionalEdges);
            if (!outputLabel.isEmpty()) {
              out.printf("  [label=\"%s\"];\n", outputLabel);
            }
          }
        },
        sortLabels ? nodeReader.comparator() : null);
  }

  /**
   * Given {@code collectionOfUnorderedSets}, a collection of sets of nodes, returns a collection of
   * sets with the same elements as {@code collectionOfUnorderedSets} but with a stable iteration
   * order within each set given by the target ordering, and the collection ordered by the same
   * induced order.
   */
  private Collection<Set<Node<T>>> orderPartition(
      Collection<Set<Node<T>>> collectionOfUnorderedSets) {
    List<Set<Node<T>>> result = new ArrayList<>();
    for (Set<Node<T>> part : collectionOfUnorderedSets) {
      List<Node<T>> toSort = new ArrayList<>(part);
      Collections.sort(toSort, nodeComparator);
      result.add(ImmutableSet.copyOf(toSort));
    }
    Collections.sort(result, nodeComparator.lexicographical());
    return result;
  }

  private void outputFactored(
      Digraph<T> graph, ConditionalEdges conditionalEdges, PrintWriter out) {
    EquivalenceRelation<Node<T>> equivalenceRelation = createEquivalenceRelation();

    Collection<Set<Node<T>>> partition =
        CollectionUtils.partition(graph.getNodes(), equivalenceRelation);
    if (sortLabels) {
      partition = orderPartition(partition);
    }

    Digraph<Set<Node<T>>> factoredGraph = graph.createImageUnderPartition(partition);

    // Concatenate the labels of all topologically-equivalent nodes.
    LabelSerializer<Set<Node<T>>> labelSerializer =
        node -> {
          int actualLimit = maxLabelSize - RESERVED_LABEL_CHARS;
          boolean firstItem = true;
          StringBuilder buf = new StringBuilder();
          int count = 0;
          for (Node<T> eqNode : node.getLabel()) {
            String labelString = nodeReader.getLabel(eqNode);
            if (!firstItem) {
              buf.append("\\n");

              // Use -1 to denote no limit, as it is easier than trying to pass MAX_INT on the
              // cmdline
              if (maxLabelSize != -1 && (buf.length() + labelString.length() > actualLimit)) {
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
        };

    factoredGraph.visitNodesBeforeEdges(
        new DotOutputVisitor<Set<Node<T>>>(out, labelSerializer) {
          @Override
          public void beginVisit() {
            super.beginVisit();
            // TODO(bazel-team): (2009) make this the default in Digraph.
            out.println("  node [shape=box];");
          }

          @Override
          public void visitEdge(Node<Set<Node<T>>> lhs, Node<Set<Node<T>>> rhs) {
            super.visitEdge(lhs, rhs);

            String outputLabel =
                getConditionsGraphLabel(lhs.getLabel(), rhs.getLabel(), conditionalEdges);
            if (!outputLabel.isEmpty()) {
              out.printf("  [label=\"%s\"];\n", outputLabel);
            }
          }
        },
        sortLabels ? nodeComparator.lexicographical() : null);
  }

  /**
   * Returns an equivalence relation for nodes in the specified graph.
   *
   * <p>Two nodes are considered equal iff they have equal topology (predecessors and successors).
   *
   * <p>TODO(bazel-team): Make this a method of Digraph.
   */
  @SuppressWarnings("ReferenceEquality")
  private static <LABEL> EquivalenceRelation<Node<LABEL>> createEquivalenceRelation() {
    return new EquivalenceRelation<Node<LABEL>>() {
      @Override
      public int compare(Node<LABEL> x, Node<LABEL> y) {
        if (x == y) {
          return 0;
        }

        if (x.numPredecessors() != y.numPredecessors() || x.numSuccessors() != y.numSuccessors()) {
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

  private String getConditionsGraphLabel(
      Iterable<Node<T>> lhs, Iterable<Node<T>> rhs, ConditionalEdges conditionalEdges) {
    StringBuilder buf = new StringBuilder();
    if (conditionalEdges == null || maxConditionalEdges == 0) {
      return buf.toString();
    }

    Set<Label> annotatedLabels = new HashSet<>();
    for (Node<T> src : lhs) {
      Label srcLabel = ((Target) src.getLabel()).getLabel();
      for (Node<T> dest : rhs) {
        Label destLabel = ((Target) dest.getLabel()).getLabel();
        Optional<Set<Label>> conditions = conditionalEdges.get(srcLabel, destLabel);
        if (conditions.isPresent()) {
          boolean firstItem = true;

          int limit =
              (maxConditionalEdges == -1) ? conditions.get().size() : (maxConditionalEdges - 1);

          for (Label conditionLabel : Iterables.limit(conditions.get(), limit)) {
            if (!annotatedLabels.add(conditionLabel)) {
              // duplicate label; skip.
              continue;
            }

            if (!firstItem) {
              buf.append("\\n");
            }

            buf.append(conditionLabel.getCanonicalForm());
            firstItem = false;
          }
          if (conditions.get().size() > limit) {
            buf.append("...");
          }
        }
      }
    }
    return buf.toString();
  }
}
