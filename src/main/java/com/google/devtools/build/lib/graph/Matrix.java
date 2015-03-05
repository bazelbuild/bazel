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

package com.google.devtools.build.lib.graph;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * <p>A simple and inefficient directed graph with the adjacency
 * relation represented as a 2-D bit-matrix. </p>
 *
 * <p> Used as an adjunct to Digraph for performing certain algorithms
 * which are more naturally implemented on this representation,
 * e.g. transitive closure and reduction. </p>
 *
 * <p> Not many operations are supported. </p>
 */
final class Matrix<T> {

  /**
   * Constructs a square bit-matrix, initially empty, with the ith row/column
   * corresponding to the ith element of 'labels', in iteration order.
   *
   * Does not retain a references to 'labels'.
   */
  public Matrix(Set<T> labels) {
    this.N = labels.size();
    this.values = new ArrayList<T>(N);
    this.indices = new HashMap<>();
    this.m = new boolean[N][N];

    for (T label: labels) {
      int idx = values.size();
      values.add(label);
      indices.put(label, idx);
    }
  }

  /**
   * Constructs a matrix from the set of logical values specified.  There is
   * one row/column for each node in the graph, and the entry matrix[i,j] is
   * set iff there is an edge in 'graph' from the node labelled values[i] to
   * the node labelled values[j].
   */
  public Matrix(Digraph<T> graph) {
    this(graph.getLabels());

    for (Node<T> nfrom: graph.getNodes()) {
      Integer ifrom = indices.get(nfrom.getLabel());
      for (Node<T> nto: nfrom.getSuccessors()) {
        Integer ito = indices.get(nto.getLabel());
        m[ifrom][ito] = true;
      }
    }
  }

  /**
   * The size of one side of the matrix.
   */
  private final int N;

  /**
   * The logical values associated with each row/column.
   */
  private final List<T> values;

  /**
   * The mapping from logical values to row/column index.
   */
  private final Map<T, Integer>  indices;

  /**
   * The bit-matrix itself.
   * m[from][to] indicates an edge from-->to.
   */
  private final boolean[][] m;

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    for (int ii = 0; ii < N; ++ii) {
      for (int jj = 0; jj < N; ++jj) {
        sb.append(m[ii][jj] ? '1' : '0');
      }
      sb.append(' ').append(values.get(ii)).append('\n');
    }
    return sb.toString();
  }

}
