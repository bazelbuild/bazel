// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.primitives.Booleans;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import java.util.HashMap;
import java.util.Map;

/**
 * An implementation of {@link QueryExpressionVisitor} which recursively visits all nested {@link
 * QueryExpression}s.
 */
public abstract class AggregatingQueryExpressionVisitor<T, C>
    implements QueryExpressionVisitor<T, C> {

  @Override
  public T visit(BinaryOperatorExpression binaryOperatorExpression, C context) {
    Map<QueryExpression, T> queryExpressionMapping = new HashMap<>();
    for (QueryExpression expr : binaryOperatorExpression.getOperands()) {
      queryExpressionMapping.put(expr, expr.accept(this, context));
    }
    return aggregate(ImmutableMap.copyOf(queryExpressionMapping));
  }

  @Override
  public T visit(FunctionExpression functionExpression, C context) {
    Map<QueryExpression, T> queryExpressionMapping = new HashMap<>();
    for (Argument argument : functionExpression.getArgs()) {
      if (argument.getType() == ArgumentType.EXPRESSION) {
        queryExpressionMapping.put(
            argument.getExpression(), argument.getExpression().accept(this, context));
      }
    }

    return aggregate(ImmutableMap.copyOf(queryExpressionMapping));
  }

  @Override
  public T visit(LetExpression letExpression, C context) {
    return aggregate(
        ImmutableMap.of(
            letExpression.getVarExpr(), letExpression.getVarExpr().accept(this, context),
            letExpression.getBodyExpr(), letExpression.getBodyExpr().accept(this, context)));
  }

  @Override
  public T visit(SetExpression setExpression, C context) {
    Map<QueryExpression, T> queryExpressionMapping = new HashMap<>();
    for (TargetLiteral targetLiteral : setExpression.getWords()) {
      queryExpressionMapping.put(targetLiteral, targetLiteral.accept(this, context));
    }

    return aggregate(ImmutableMap.copyOf(queryExpressionMapping));
  }

  /**
   * Aggregates results from all sub-expression visitations. The map is guaranteed to include
   * results for all direct sub-expressions.
   */
  protected abstract T aggregate(ImmutableMap<QueryExpression, T> resultMap);

  /**
   * Returns {@code true} when the query expression contains at least one {@link QueryFunction}
   * whose name is in the set of {@code functionName}.
   */
  public static class ContainsFunctionQueryExpressionVisitor
      extends AggregatingQueryExpressionVisitor<Boolean, Void>
      implements QueryExpressionVisitor<Boolean, Void> {

    private final ImmutableSet<String> functionNames;

    public ContainsFunctionQueryExpressionVisitor(Iterable<String> functionNames) {
      this.functionNames = ImmutableSet.copyOf(functionNames);
    }

    @Override
    public Boolean visit(TargetLiteral targetLiteral, Void context) {
      return false;
    }

    @Override
    public Boolean visit(SetExpression setExpression, Void context) {
      return false;
    }

    @Override
    public Boolean visit(FunctionExpression functionExpression, Void context) {
      QueryFunction function = functionExpression.getFunction();
      if (functionNames.contains(function.getName())) {
        return true;
      } else {
        return super.visit(functionExpression, context);
      }
    }

    @Override
    protected Boolean aggregate(ImmutableMap<QueryExpression, Boolean> resultMap) {
      return Booleans.contains(Booleans.toArray(resultMap.values()), true);
    }
  }
}
