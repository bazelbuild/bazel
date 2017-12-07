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

/**
 * An implementation of {@link QueryExpressionVisitor} which recursively visits all nested {@link
 * QueryExpression}s
 */
public abstract class AggregatingQueryExpressionVisitor<T> implements QueryExpressionVisitor<T> {

  @Override
  public T visit(BinaryOperatorExpression binaryOperatorExpression) {
    ImmutableMap.Builder<QueryExpression, T> builder = ImmutableMap.builder();
    for (QueryExpression expr : binaryOperatorExpression.getOperands()) {
      builder.put(expr, expr.accept(this));
    }

    return aggregate(builder.build());
  }

  @Override
  public T visit(FunctionExpression functionExpression) {
    ImmutableMap.Builder<QueryExpression, T> builder = ImmutableMap.builder();
    for (Argument argument : functionExpression.getArgs()) {
      if (argument.getType() == ArgumentType.EXPRESSION) {
        builder.put(argument.getExpression(), argument.getExpression().accept(this));
      }
    }

    return aggregate(builder.build());
  }

  @Override
  public T visit(LetExpression letExpression) {
    return aggregate(
        ImmutableMap.of(
            letExpression.getVarExpr(), letExpression.getVarExpr().accept(this),
            letExpression.getBodyExpr(), letExpression.getBodyExpr().accept(this)));
  }

  @Override
  public T visit(SetExpression setExpression) {
    ImmutableMap.Builder<QueryExpression, T> builder = ImmutableMap.builder();
    for (TargetLiteral targetLiteral : setExpression.getWords()) {
      builder.put(targetLiteral, targetLiteral.accept(this));
    }

    return aggregate(builder.build());
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
      extends AggregatingQueryExpressionVisitor<Boolean>
      implements QueryExpressionVisitor<Boolean> {

    private final ImmutableSet<String> functionNames;

    public ContainsFunctionQueryExpressionVisitor(Iterable<String> functionNames) {
      this.functionNames = ImmutableSet.copyOf(functionNames);
    }

    @Override
    public Boolean visit(TargetLiteral targetLiteral) {
      return false;
    }

    @Override
    public Boolean visit(SetExpression setExpression) {
      return false;
    }

    @Override
    public Boolean visit(FunctionExpression functionExpression) {
      QueryFunction function = functionExpression.getFunction();
      if (functionNames.contains(function.getName())) {
        return true;
      } else {
        return super.visit(functionExpression);
      }
    }

    @Override
    protected Boolean aggregate(ImmutableMap<QueryExpression, Boolean> resultMap) {
      return Booleans.contains(Booleans.toArray(resultMap.values()), true);
    }
  }
}
