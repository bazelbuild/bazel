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

import com.google.common.hash.HashFunction;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
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
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * An output formatter that prints the labels in minimum rank order, preceded by
 * their rank number.  "Roots" have rank 0, their direct prerequisites have
 * rank 1, etc.  All nodes in a cycle are considered of equal rank.  MINRANK
 * shows the lowest rank for a given node, i.e. the length of the shortest
 * path from a zero-rank node to it.
 *
 * <p>If the result came from a <code>deps(x)</code> query, then the MINRANKs
 * correspond to the shortest path from x to each of its prerequisites.
 */
class MinrankOutputFormatter extends OutputFormatter {

  @Override
  public String getName() {
    return "minrank";
  }

  private static void outputToStreamOrSave(
      int rank,
      Label label,
      PrintStream out,
      @Nullable List<RankAndLabel> toSave,
      final String lineTerminator) {
    if (toSave != null) {
      toSave.add(new RankAndLabel(rank, label));
    } else {
      out.print(rank + " " + label.getCanonicalForm() + lineTerminator);
    }
  }

  @Override
  public void output(
      QueryOptions options,
      Digraph<Target> result,
      OutputStream out,
      AspectResolver aspectResolver,
      EventHandler eventHandler,
      HashFunction hashFunction)
      throws IOException {
    PrintStream printStream = new PrintStream(out);
    // getRoots() isn't defined for cyclic graphs, so in order to handle
    // cycles correctly, we need work on the strong component graph, as
    // cycles should be treated a "clump" of nodes all on the same rank.
    // Graphs may contain cycles because there are errors in BUILD files.

    List<RankAndLabel> outputToOrder =
        options.orderOutput == OrderOutput.FULL ? new ArrayList<>() : null;
    Digraph<Set<Node<Target>>> scGraph = result.getStrongComponentGraph();
    Set<Node<Set<Node<Target>>>> rankNodes = scGraph.getRoots();
    Set<Node<Set<Node<Target>>>> seen = new HashSet<>();
    seen.addAll(rankNodes);
    final String lineTerm = options.getLineTerminator();
    for (int rank = 0; !rankNodes.isEmpty(); rank++) {
      // Print out this rank:
      for (Node<Set<Node<Target>>> xScc : rankNodes) {
        for (Node<Target> x : xScc.getLabel()) {
          outputToStreamOrSave(
              rank, x.getLabel().getLabel(), printStream, outputToOrder, lineTerm);
        }
      }

      // Find the next rank:
      Set<Node<Set<Node<Target>>>> nextRankNodes = new LinkedHashSet<>();
      for (Node<Set<Node<Target>>> x : rankNodes) {
        for (Node<Set<Node<Target>>> y : x.getSuccessors()) {
          if (seen.add(y)) {
            nextRankNodes.add(y);
          }
        }
      }
      rankNodes = nextRankNodes;
    }
    if (outputToOrder != null) {
      Collections.sort(outputToOrder);
      for (RankAndLabel item : outputToOrder) {
        printStream.print(item + lineTerm);
      }
    }

    flushAndCheckError(printStream);
  }

  private static void flushAndCheckError(PrintStream printStream) throws IOException {
    if (printStream.checkError()) {
      throw new IOException("PrintStream encountered an error");
    }
  }
}