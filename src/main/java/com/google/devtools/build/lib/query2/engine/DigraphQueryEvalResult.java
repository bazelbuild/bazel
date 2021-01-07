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
import com.google.devtools.build.lib.util.DetailedExitCode;

/** {@link QueryEvalResult} along with a digraph giving the structure of the results. */
public class DigraphQueryEvalResult<T> extends QueryEvalResult {

  private final Digraph<T> graph;

  public DigraphQueryEvalResult(
      boolean success, boolean isEmpty, DetailedExitCode detailedExitCode, Digraph<T> graph) {
    super(success, isEmpty, detailedExitCode);
    this.graph = Preconditions.checkNotNull(graph);
  }

  /** Returns the recorded graph */
  public Digraph<T> getGraph() {
    return graph;
  }
}
