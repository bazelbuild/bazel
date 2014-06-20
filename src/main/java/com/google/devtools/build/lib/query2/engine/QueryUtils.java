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
package com.google.devtools.build.lib.query2.engine;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.graph.Node;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Utility functions over sets of target nodes, for evaluation.
 */
public final class QueryUtils {

  private QueryUtils() {} // uninstantiable

  /**
   * Given a set of target nodes and a predicate over targets, returns the set
   * of nodes whose targets match the predicate.
   */
  public static <T> Set<Node<T>> filterTargets(Set<Node<T>> input,
                                               Predicate<T> predicate) {
    Set<Node<T>> result = new LinkedHashSet<>();
    for (Node<T> node : input) {
      if (predicate.apply(node.getLabel())) {
        result.add(node);
      }
    }
    return result;
  }

  /**
   * Given a set of target nodes, returns the targets.
   */
  public static <T> Set<T> getTargetsFromNodes(Set<Node<T>> input) {
    Set<T> result = new LinkedHashSet<>();
    for (Node<T> node : input) {
      result.add(node.getLabel());
    }
    return result;
  }

  public static ImmutableList<QueryEnvironment.QueryFunction> getDefaultFunctions() {
    return ImmutableList.<QueryEnvironment.QueryFunction>of(
        new AllPathsFunction(),
        new BuildFilesFunction(),
        new AttrFunction(),
        new FilterFunction(),
        new LabelsFunction(),
        new KindFunction(),
        new SomeFunction(),
        new SomePathFunction(),
        new TestsFunction(),
        new DepsFunction(),
        new RdepsFunction()
        );
  }
}
