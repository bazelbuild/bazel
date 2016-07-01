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
import java.util.Set;
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
  public <T> void eval(QueryEnvironment<T> env, Callback<T> callback)
      throws QueryException, InterruptedException {
    if (!NAME_PATTERN.matcher(varName).matches()) {
      throw new QueryException(this, "invalid variable name '" + varName + "' in let expression");
    }
    // We eval all because we would need a stack of variable contexts for implementing setVariable
    // correctly.
    Set<T> varValue = QueryUtil.evalAll(env, varExpr);
    Set<T> prevValue = env.setVariable(varName, varValue);
    try {
      // Same as varExpr. We cannot pass partial results to the parent without having
      // a stack of variable contexts.
      callback.process(QueryUtil.evalAll(env, bodyExpr));
    } finally {
      env.setVariable(varName, prevValue); // restore
    }
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
