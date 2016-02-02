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

import java.util.Collection;

/**
 * Base class for expressions in the Blaze query language, revision 2.
 *
 * <p>All queries return a subgraph of the dependency graph, represented
 * as a set of target nodes.
 *
 * <p>All queries must ensure that sufficient graph edges are created in the
 * QueryEnvironment so that all nodes in the result are correctly ordered
 * according to the type of query.  For example, "deps" queries require that
 * all the nodes in the transitive closure of its argument set are correctly
 * ordered w.r.t. each other; "somepath" queries require that the order of the
 * nodes on the resulting path are correctly ordered; algebraic set operations
 * such as intersect and union are inherently unordered.
 *
 * <h2>Package overview</h2>
 *
 * <p>This package consists of two basic class hierarchies.  The first, {@code
 * QueryExpression}, is the set of different query expressions in the language,
 * and the {@link #eval} method of each defines the semantics.  The result of
 * evaluating a query is set of Blaze {@code Target}s (a file or rule).  The
 * set may be interpreted as either a set or as nodes of a DAG, depending on
 * the context.
 *
 * <p>The second hierarchy is {@code OutputFormatter}.  Its subclasses define
 * different ways of printing out the result of a query.  Each accepts a {@code
 * Digraph} of {@code Target}s, and an output stream.
 */
public abstract class QueryExpression {

  /**
   * Scan and parse the specified query expression.
   */
  public static QueryExpression parse(String query, QueryEnvironment<?> env)
      throws QueryException {
    return QueryParser.parse(query, env);
  }

  protected QueryExpression() {}

  /**
   * Evaluates this query in the specified environment, and notifies the callback with a result.
   * Note that it is allowed to notify the callback with partial results instead of just one final
   * result.
   *
   * <p>Failures resulting from evaluation of an ill-formed query cause
   * QueryException to be thrown.
   *
   * <p>The reporting of failures arising from errors in BUILD files depends on
   * the --keep_going flag.  If enabled (the default), then QueryException is
   * thrown.  If disabled, evaluation will stumble on to produce a (possibly
   * inaccurate) result, but a result nonetheless.
   */
  public abstract <T> void eval(QueryEnvironment<T> env, Callback<T> callback)
      throws QueryException, InterruptedException;

  /**
   * Collects all target patterns that are referenced anywhere within this query expression and adds
   * them to the given collection, which must be mutable.
   */
  public abstract void collectTargetPatterns(Collection<String> literals);

  /**
   * Returns this query expression pretty-printed.
   */
  @Override
  public abstract String toString();
}
