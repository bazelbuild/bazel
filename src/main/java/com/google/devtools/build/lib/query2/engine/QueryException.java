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

/**
 */
public class QueryException extends Exception {

  /**
   * Returns a better error message for the query.
   */
  static String describeFailedQuery(QueryException e, QueryExpression toplevel) {
    QueryExpression badQuery = e.getFailedExpression();
    if (badQuery == null) {
      return "Evaluation failed: " + e.getMessage();
    }
    return badQuery == toplevel
        ? "Evaluation of query \"" + toplevel + "\" failed: " + e.getMessage()
        : "Evaluation of subquery \"" + badQuery
            + "\" failed (did you want to use --keep_going?): " + e.getMessage();
  }

  private final QueryExpression expression;

  public QueryException(QueryException e, QueryExpression toplevel) {
    super(describeFailedQuery(e, toplevel), e);
    this.expression = null;
  }

  public QueryException(QueryExpression expression, String message) {
    super(message);
    this.expression = expression;
  }

  public QueryException(String message) {
    this(null, message);
  }

  /**
   * Returns the subexpression for which evaluation failed, or null if
   * the failure occurred during lexing/parsing.
   */
  public QueryExpression getFailedExpression() {
    return expression;
  }

}
