// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;

/**
 * Performs an arbitrary contextual transformation of a {@link QueryExpression}.
 *
 * <p>For each subclass of {@link QueryExpression}, there's a corresponding {@link #visit} overload
 * that transforms a node of that type. By default, this method recursively applies this {@link
 * QueryExpressionMapper}'s transformation in a structure-preserving manner (trying to maintain
 * reference-equality, as an optimization). Subclasses of {@link QueryExpressionMapper} can override
 * these methods in order to implement an arbitrary transformation.
 */
public abstract class QueryExpressionMapper<C>
    implements QueryExpressionVisitor<QueryExpression, C> {

  @Override
  public QueryExpression visit(TargetLiteral targetLiteral, C context) {
    return targetLiteral;
  }

  @Override
  public QueryExpression visit(BinaryOperatorExpression binaryOperatorExpression, C context) {
    boolean changed = false;
    ImmutableList.Builder<QueryExpression> mappedOperandsBuilder = ImmutableList.builder();
    for (QueryExpression operand : binaryOperatorExpression.getOperands()) {
      QueryExpression mappedOperand = operand.accept(this, context);
      if (mappedOperand != operand) {
        changed = true;
      }
      mappedOperandsBuilder.add(mappedOperand);
    }
    return changed
        ? new BinaryOperatorExpression(
            binaryOperatorExpression.getOperator(), mappedOperandsBuilder.build())
        : binaryOperatorExpression;
  }

  @Override
  public QueryExpression visit(FunctionExpression functionExpression, C context) {
    boolean changed = false;
    ImmutableList.Builder<Argument> mappedArgumentBuilder = ImmutableList.builder();
    for (Argument argument : functionExpression.getArgs()) {
      switch (argument.getType()) {
        case EXPRESSION -> {
          QueryExpression expr = argument.getExpression();
          QueryExpression mappedExpression = expr.accept(this, context);
          mappedArgumentBuilder.add(Argument.of(mappedExpression));
          if (expr != mappedExpression) {
            changed = true;
          }
        }
        default -> mappedArgumentBuilder.add(argument);
      }
    }
    return changed
        ? new FunctionExpression(functionExpression.getFunction(), mappedArgumentBuilder.build())
        : functionExpression;
  }

  @Override
  public QueryExpression visit(LetExpression letExpression, C context) {
    boolean changed = false;
    QueryExpression mappedVarExpr = letExpression.getVarExpr().accept(this, context);
    if (mappedVarExpr != letExpression.getVarExpr()) {
      changed = true;
    }
    QueryExpression mappedBodyExpr = letExpression.getBodyExpr().accept(this, context);
    if (mappedBodyExpr != letExpression.getBodyExpr()) {
      changed = true;
    }
    return changed
        ? new LetExpression(letExpression.getVarName(), mappedVarExpr, mappedBodyExpr)
        : letExpression;
  }

  @Override
  public QueryExpression visit(SetExpression setExpression, C context) {
    return setExpression;
  }

  public static QueryExpressionMapper<Void> identity() {
    return IdentityMapper.INSTANCE;
  }

  /**
   * Returns a {@link QueryExpressionMapper} which applies all the mappings provided by {@code
   * mappers}, in the reverse order of mapper array.
   */
  public static <C> QueryExpressionMapper<C> compose(
      ImmutableList<QueryExpressionMapper<C>> mappers) {
    return new ComposedQueryExpressionMapper<>(mappers);
  }

  private static class ComposedQueryExpressionMapper<C> extends QueryExpressionMapper<C> {
    private final ImmutableList<QueryExpressionMapper<C>> mappers;

    private ComposedQueryExpressionMapper(ImmutableList<QueryExpressionMapper<C>> mappers) {
      this.mappers = mappers;
    }

    @Override
    public QueryExpression visit(TargetLiteral targetLiteral, C context) {
      return mapAll(targetLiteral, mappers, context);
    }

    @Override
    public QueryExpression visit(BinaryOperatorExpression binaryOperatorExpression, C context) {
      return mapAll(binaryOperatorExpression, mappers, context);
    }

    @Override
    public QueryExpression visit(FunctionExpression functionExpression, C context) {
      return mapAll(functionExpression, mappers, context);
    }

    @Override
    public QueryExpression visit(LetExpression letExpression, C context) {
      return mapAll(letExpression, mappers, context);
    }

    @Override
    public QueryExpression visit(SetExpression setExpression, C context) {
      return mapAll(setExpression, mappers, context);
    }

    private static <C> QueryExpression mapAll(
        QueryExpression expression, ImmutableList<QueryExpressionMapper<C>> mappers, C context) {
      QueryExpression expr = expression;
      for (int i = mappers.size() - 1; i >= 0; i--) {
        expr = expr.accept(mappers.get(i), context);
      }

      return expr;
    }
  }

  private static class IdentityMapper extends QueryExpressionMapper<Void> {
    private static final IdentityMapper INSTANCE = new IdentityMapper();

    private IdentityMapper() {
    }

    @Override
    public QueryExpression visit(TargetLiteral targetLiteral, Void context) {
      return targetLiteral;
    }

    @Override
    public QueryExpression visit(BinaryOperatorExpression binaryOperatorExpression, Void context) {
      return binaryOperatorExpression;
    }

    @Override
    public QueryExpression visit(FunctionExpression functionExpression, Void context) {
      return functionExpression;
    }

    @Override
    public QueryExpression visit(LetExpression letExpression, Void context) {
      return letExpression;
    }

    @Override
    public QueryExpression visit(SetExpression setExpression, Void context) {
      return setExpression;
    }
  }
}

