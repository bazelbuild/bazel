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

package com.google.devtools.build.lib.query2.query.output;

import static java.util.Comparator.comparingInt;

import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.graph.Node;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.query.aspectresolvers.AspectResolver;
import com.google.devtools.build.lib.query2.query.output.QueryOptions.OrderOutput;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * An output formatter that prints the labels in maximum rank order, preceded
 * by their rank number.  "Roots" have rank 0, all other nodes have a rank
 * which is one greater than the maximum rank of each of their predecessors.
 * All nodes in a cycle are considered of equal rank.  MAXRANK shows the
 * highest rank for a given node, i.e. the length of the longest non-cyclic
 * path from a zero-rank node to it.
 *
 * <p>If the result came from a <code>deps(x)</code> query, then the MAXRANKs
 * correspond to the longest path from x to each of its prerequisites.
 */
class MaxrankOutputFormatter extends OutputFormatter {

  @Override
  public String getName() {
    return "maxrank";
  }

  @Override
  public void output(
      QueryOptions options, Digraph<Target> result, OutputStream out, AspectResolver aspectResolver)
      throws IOException {
    // In order to handle cycles correctly, we need work on the strong
    // component graph, as cycles should be treated a "clump" of nodes all on
    // the same rank. Graphs may contain cycles because there are errors in BUILD files.

    // Dynamic programming algorithm:
    // rank(x) = max(rank(p)) + 1 foreach p in preds(x)
    // TODO(bazel-team): Move to Digraph.
    class DP {
      final Map<Node<Set<Node<Target>>>, Integer> ranks = new HashMap<>();

      int rank(Node<Set<Node<Target>>> node) {
        Integer rank = ranks.get(node);
        if (rank == null) {
          int maxPredRank = -1;
          for (Node<Set<Node<Target>>> p : node.getPredecessors()) {
            maxPredRank = Math.max(maxPredRank, rank(p));
          }
          rank = maxPredRank + 1;
          ranks.put(node, rank);
        }
        return rank;
      }
    }
    DP dp = new DP();

    // Now sort by rank...
    List<RankAndLabel> output = new ArrayList<>();
    for (Node<Set<Node<Target>>> x : result.getStrongComponentGraph().getNodes()) {
      int rank = dp.rank(x);
      for (Node<Target> y : x.getLabel()) {
        output.add(new RankAndLabel(rank, y.getLabel().getLabel()));
      }
    }
    if (options.orderOutput == OrderOutput.FULL) {
      // Use the natural order for RankAndLabels, which breaks ties alphabetically.
      Collections.sort(output);
    } else {
      Collections.sort(output, comparingInt(RankAndLabel::getRank));
    }
    final String lineTerm = options.getLineTerminator();
    PrintStream printStream = new PrintStream(out);
    for (RankAndLabel item : output) {
      printStream.print(item + lineTerm);
    }
    flushAndCheckError(printStream);
  }

  private static void flushAndCheckError(PrintStream printStream) throws IOException {
    if (printStream.checkError()) {
      throw new IOException("PrintStream encountered an error");
    }
  }
}