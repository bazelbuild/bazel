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
import com.google.devtools.build.lib.server.FailureDetails.ActionQuery;
import com.google.devtools.build.lib.server.FailureDetails.ConfigurableQuery;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Query;

/** Exception indicating a failure in Blaze query, aquery, or cquery. */
public class QueryException extends Exception {

  /** Returns a better error message for the query. */
  static String describeFailedQuery(QueryException e, QueryExpression toplevel) {
    QueryExpression badQuery = e.getFailedExpression();
    if (badQuery == null) {
      return "Evaluation failed: " + e.getMessage();
    }
    return badQuery == toplevel
        ? "Evaluation of query \"" + toplevel.toTrunctatedString() + "\" failed: " + e.getMessage()
        : "Evaluation of subquery \""
            + badQuery.toTrunctatedString()
            + "\" failed (did you want to use --keep_going?): "
            + e.getMessage();
  }

  private final QueryExpression expression;
  private final FailureDetail failureDetail;

  public QueryException(QueryException e, QueryExpression toplevel) {
    super(describeFailedQuery(e, toplevel), e);
    this.expression = null;
    this.failureDetail = e.getFailureDetail();
  }

  public QueryException(
      QueryExpression expression, String message, Throwable cause, FailureDetail failureDetail) {
    super(message, cause);
    this.expression = expression;
    this.failureDetail = Preconditions.checkNotNull(failureDetail);
  }

  public QueryException(QueryExpression expression, String message, FailureDetail failureDetail) {
    super(message);
    this.expression = expression;
    this.failureDetail = Preconditions.checkNotNull(failureDetail);
  }

  public QueryException(QueryExpression expression, String message, Query.Code queryCode) {
    this(
        expression,
        message,
        FailureDetail.newBuilder()
            .setMessage(message)
            .setQuery(Query.newBuilder().setCode(queryCode).build())
            .build());
  }

  public QueryException(
      QueryExpression expression, String message, ActionQuery.Code actionQueryCode) {
    this(
        expression,
        message,
        FailureDetail.newBuilder()
            .setMessage(message)
            .setActionQuery(ActionQuery.newBuilder().setCode(actionQueryCode).build())
            .build());
  }

  public QueryException(
      QueryExpression expression, String message, ConfigurableQuery.Code configurableQueryCode) {
    this(
        expression,
        message,
        FailureDetail.newBuilder()
            .setMessage(message)
            .setConfigurableQuery(
                ConfigurableQuery.newBuilder().setCode(configurableQueryCode).build())
            .build());
  }

  public QueryException(String message, Throwable cause, FailureDetail failureDetail) {
    super(message, cause);
    this.expression = null;
    this.failureDetail = Preconditions.checkNotNull(failureDetail);
  }

  public QueryException(String message, FailureDetail failureDetail) {
    super(message);
    this.expression = null;
    this.failureDetail = Preconditions.checkNotNull(failureDetail);
  }

  public QueryException(String message, Query.Code queryCode) {
    this(null, message, queryCode);
  }

  public QueryException(String message, ActionQuery.Code actionQueryCode) {
    this(null, message, actionQueryCode);
  }

  public QueryException(String message, ConfigurableQuery.Code configurableQueryCode) {
    this(null, message, configurableQueryCode);
  }

  /**
   * Returns the subexpression for which evaluation failed, or null if
   * the failure occurred during lexing/parsing.
   */
  public QueryExpression getFailedExpression() {
    return expression;
  }

  /** Returns a {@link FailureDetail} with a corresponding code of the query error. */
  public FailureDetail getFailureDetail() {
    return failureDetail;
  }
}
