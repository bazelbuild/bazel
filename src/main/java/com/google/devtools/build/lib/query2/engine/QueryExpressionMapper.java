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
 * Performs an arbitrary transformation of a {@link QueryExpression}.
 *
 * <p>For each subclass of {@link QueryExpression}, there's a corresponding {@link #map} overload
 * that transforms a node of that type. By default, this method recursively applies this
 * {@link QueryExpressionMapper}'s transformation in a structure-preserving manner (trying to
 * maintain reference-equality, as an optimization). Subclasses of {@link QueryExpressionMapper} can
 * override these methods in order to implement an arbitrary transformation.
 */
public abstract class QueryExpressionMapper {
  public QueryExpression map(TargetLiteral targetLiteral) {
    return targetLiteral;
  }

  public QueryExpression map(BinaryOperatorExpression binaryOperatorExpression) {
    boolean changed = false;
    ImmutableList.Builder<QueryExpression> mappedOperandsBuilder = ImmutableList.builder();
    for (QueryExpression operand : binaryOperatorExpression.getOperands()) {
      QueryExpression mappedOperand = operand.getMapped(this);
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

  public QueryExpression map(FunctionExpression functionExpression) {
    boolean changed = false;
    ImmutableList.Builder<Argument> mappedArgumentBuilder = ImmutableList.builder();
    for (Argument argument : functionExpression.getArgs()) {
      switch (argument.getType()) {
        case EXPRESSION: {
          QueryExpression expr = argument.getExpression();
          QueryExpression mappedExpression = expr.getMapped(this);
          mappedArgumentBuilder.add(Argument.of(mappedExpression));
          if (expr != mappedExpression) {
            changed = true;
          }
          break;
        }
        default:
          mappedArgumentBuilder.add(argument);
          break;
      }
    }
    return changed
        ? new FunctionExpression(functionExpression.getFunction(), mappedArgumentBuilder.build())
        : functionExpression;
  }

  public QueryExpression map(LetExpression letExpression) {
    boolean changed = false;
    QueryExpression mappedVarExpr = letExpression.getVarExpr().getMapped(this);
    if (mappedVarExpr != letExpression.getVarExpr()) {
      changed = true;
    }
    QueryExpression mappedBodyExpr = letExpression.getBodyExpr().getMapped(this);
    if (mappedBodyExpr != letExpression.getBodyExpr()) {
      changed = true;
    }
    return changed
        ? new LetExpression(letExpression.getVarName(), mappedVarExpr, mappedBodyExpr)
        : letExpression;
  }

  public QueryExpression map(SetExpression setExpression) {
    return setExpression;
  }

  public static QueryExpressionMapper identity() {
    return IdentityMapper.INSTANCE;
  }

  public static QueryExpressionMapper compose(
      QueryExpressionMapper outerMapper, QueryExpressionMapper innerMapper) {
    return new ComposedQueryExpressionMapper(outerMapper, innerMapper);
  }

  private static class ComposedQueryExpressionMapper extends QueryExpressionMapper {
    private QueryExpressionMapper outerMapper;
    private QueryExpressionMapper innerMapper;

    private ComposedQueryExpressionMapper(
        QueryExpressionMapper outerMapper,
        QueryExpressionMapper innerMapper) {
      this.outerMapper = outerMapper;
      this.innerMapper = innerMapper;
    }

    @Override
    public QueryExpression map(TargetLiteral targetLiteral) {
      return innerMapper.map(targetLiteral).getMapped(outerMapper);
    }

    @Override
    public QueryExpression map(BinaryOperatorExpression binaryOperatorExpression) {
      return innerMapper.map(binaryOperatorExpression).getMapped(outerMapper);
    }

    @Override
    public QueryExpression map(FunctionExpression functionExpression) {
      return innerMapper.map(functionExpression).getMapped(outerMapper);
    }

    @Override
    public QueryExpression map(LetExpression letExpression) {
      return innerMapper.map(letExpression).getMapped(outerMapper);
    }

    @Override
    public QueryExpression map(SetExpression setExpression) {
      return innerMapper.map(setExpression).getMapped(outerMapper);
    }
  }

  private static class IdentityMapper extends QueryExpressionMapper {
    private static final IdentityMapper INSTANCE = new IdentityMapper();

    private IdentityMapper() {
    }

    @Override
    public QueryExpression map(TargetLiteral targetLiteral) {
      return targetLiteral;
    }

    @Override
    public QueryExpression map(BinaryOperatorExpression binaryOperatorExpression) {
      return binaryOperatorExpression;
    }

    @Override
    public QueryExpression map(FunctionExpression functionExpression) {
      return functionExpression;
    }

    @Override
    public QueryExpression map(LetExpression letExpression) {
      return letExpression;
    }

    @Override
    public QueryExpression map(SetExpression setExpression) {
      return setExpression;
    }
  }
}

