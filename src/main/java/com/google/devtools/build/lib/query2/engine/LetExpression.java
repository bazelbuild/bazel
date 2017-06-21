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

import com.google.common.base.Function;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ThreadSafeMutableSet;
import java.util.Collection;
import java.util.regex.Pattern;

/**
 * A let expression.
 *
 * <pre>expr ::= LET WORD = expr IN expr</pre>
 */
class LetExpression extends QueryExpression {

  private static final String VAR_NAME_PATTERN = "[a-zA-Z_][a-zA-Z0-9_]*$";

  // Variables names may be any legal identifier in the C programming language
  private static final Pattern NAME_PATTERN = Pattern.compile("^" + VAR_NAME_PATTERN);

  // Variable references are prepended with the "$" character.
  // A variable named "x" is referenced as "$x".
  private static final Pattern REF_PATTERN = Pattern.compile("^\\$" + VAR_NAME_PATTERN);

  static boolean isValidVarReference(String varName) {
    return REF_PATTERN.matcher(varName).matches();
  }

  static String getNameFromReference(String reference) {
    return reference.substring(1);
  }

  private final String varName;
  private final QueryExpression varExpr;
  private final QueryExpression bodyExpr;

  LetExpression(String varName, QueryExpression varExpr, QueryExpression bodyExpr) {
    this.varName = varName;
    this.varExpr = varExpr;
    this.bodyExpr = bodyExpr;
  }

  String getVarName() {
    return varName;
  }

  QueryExpression getVarExpr() {
    return varExpr;
  }

  QueryExpression getBodyExpr() {
    return bodyExpr;
  }

  @Override
  public <T> QueryTaskFuture<Void> eval(
      final QueryEnvironment<T> env,
      final VariableContext<T> context,
      final Callback<T> callback) {
    if (!NAME_PATTERN.matcher(varName).matches()) {
      return env.immediateFailedFuture(
          new QueryException(this, "invalid variable name '" + varName + "' in let expression"));
    }
    QueryTaskFuture<ThreadSafeMutableSet<T>> varValueFuture =
        QueryUtil.evalAll(env, context, varExpr);
    Function<ThreadSafeMutableSet<T>, QueryTaskFuture<Void>> evalBodyAsyncFunction =
        new Function<ThreadSafeMutableSet<T>, QueryTaskFuture<Void>>() {
          @Override
          public QueryTaskFuture<Void> apply(ThreadSafeMutableSet<T> varValue) {
            VariableContext<T> bodyContext = VariableContext.with(context, varName, varValue);
            return env.eval(bodyExpr, bodyContext, callback);
          }
    };
    return env.transformAsync(varValueFuture, evalBodyAsyncFunction);
  }

  @Override
  public void collectTargetPatterns(Collection<String> literals) {
    varExpr.collectTargetPatterns(literals);
    bodyExpr.collectTargetPatterns(literals);
  }

  @Override
  public QueryExpression getMapped(QueryExpressionMapper mapper) {
    return mapper.map(this);
  }

  @Override
  public String toString() {
    return "let " + varName + " = " + varExpr + " in " + bodyExpr;
  }
}
