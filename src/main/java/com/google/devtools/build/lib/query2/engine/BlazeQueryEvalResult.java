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

package com.google.devtools.build.lib.query2.engine;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.graph.Digraph;

import java.util.Set;

/** {@link QueryEvalResult} along with a digraph giving the structure of the results. */
public class BlazeQueryEvalResult<T> extends QueryEvalResult<T> {

  private final Digraph<T> graph;

  public BlazeQueryEvalResult(boolean success, Set<T> resultSet, Digraph<T> graph) {
    super(success, resultSet);
    this.graph = Preconditions.checkNotNull(graph);
  }

  /** Returns the result as a directed graph over elements. */
  public Digraph<T> getResultGraph() {
    return graph.extractSubgraph(resultSet);
  }
}
