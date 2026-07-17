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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskCallable;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ThreadSafeMutableSet;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;

/**
 * A binary algebraic set operation.
 *
 * <pre>
 * expr ::= expr (INTERSECT expr)+
 *        | expr ('^' expr)+
 *        | expr (UNION expr)+
 *        | expr ('+' expr)+
 *        | expr (EXCEPT expr)+
 *        | expr ('-' expr)+
 * </pre>
 */
public class BinaryOperatorExpression extends QueryExpression {

  private final Lexer.TokenKind operator; // ::= INTERSECT/CARET | UNION/PLUS | EXCEPT/MINUS
  private final ImmutableList<QueryExpression> operands;

  public BinaryOperatorExpression(Lexer.TokenKind operator, List<QueryExpression> operands) {
    Preconditions.checkState(operands.size() > 1);
    this.operator = operator;
    this.operands = ImmutableList.copyOf(operands);
  }

  public Lexer.TokenKind getOperator() {
    return operator;
  }

  public ImmutableList<QueryExpression> getOperands() {
    return operands;
  }

  @Override
  public <T> QueryTaskFuture<Void> eval(
      QueryEnvironment<T> env, QueryExpressionContext<T> context, Callback<T> callback) {
    return switch (operator) {
      case PLUS, UNION -> evalPlus(operands, env, context, callback);
      case MINUS, EXCEPT -> evalMinus(operands, env, context, callback);
      case INTERSECT, CARET -> evalIntersect(env, context, callback);
      default -> throw new IllegalStateException(operator.toString());
    };
  }

  /**
   * Evaluates an expression of the form "e1 + e2 + ... + eK" by evaluating all the subexpressions
   * separately.
   *
   * <p>N.B. {@code operands.size()} may be {@code 1}.
   */
  private static <T> QueryTaskFuture<Void> evalPlus(
      ImmutableList<QueryExpression> operands,
      QueryEnvironment<T> env,
      QueryExpressionContext<T> context,
      Callback<T> callback) {
    ArrayList<QueryTaskFuture<Void>> queryTasks = new ArrayList<>(operands.size());
    for (QueryExpression operand : operands) {
      queryTasks.add(env.eval(operand, context, callback));
    }
    return env.whenAllSucceed(queryTasks);
  }

  /**
   * Evaluates an expression of the form "e1 - e2 - ... - eK" by noting its equivalence to "e1 - (e2
   * + ... + eK)" and evaluating the subexpressions on the right-hand-side separately.
   */
  private static <T> QueryTaskFuture<Void> evalMinus(
      final ImmutableList<QueryExpression> operands,
      final QueryEnvironment<T> env,
      final QueryExpressionContext<T> context,
      final Callback<T> callback) {
    QueryTaskFuture<ThreadSafeMutableSet<T>> lhsValueFuture =
        QueryUtil.evalAll(env, context, operands.get(0));
    Function<ThreadSafeMutableSet<T>, QueryTaskFuture<Void>> subtractAsyncFunction =
        lhsValue -> {
          final Set<T> threadSafeLhsValue = lhsValue;
          Callback<T> subtractionCallback =
              new Callback<T>() {
                @Override
                public void process(Iterable<T> partialResult) {
                  for (T target : partialResult) {
                    threadSafeLhsValue.remove(target);
                  }
                }
              };
          QueryTaskFuture<Void> rhsEvaluatedFuture =
              evalPlus(operands.subList(1, operands.size()), env, context, subtractionCallback);
          return env.whenSucceedsCall(
              rhsEvaluatedFuture,
              new QueryTaskCallable<Void>() {
                @Override
                public Void call() throws QueryException, InterruptedException {
                  callback.process(threadSafeLhsValue);
                  return null;
                }
              });
        };
    return env.transformAsync(lhsValueFuture, subtractAsyncFunction);
  }

  private <T> QueryTaskFuture<Void> evalIntersect(
      final QueryEnvironment<T> env,
      final QueryExpressionContext<T> context,
      final Callback<T> callback) {
    // For each right-hand side operand, intersection cannot be performed in a streaming manner; the
    // entire result of that operand is needed. So, in order to avoid pinning too much in memory at
    // once, we process each right-hand side operand one at a time and throw away that operand's
    // result.
    // TODO(bazel-team): Consider keeping just the name / label of the right-hand side results
    // instead of the potentially heavy-weight instances of type T. This would let us process all
    // right-hand side operands in parallel without worrying about memory usage.
    QueryTaskFuture<ThreadSafeMutableSet<T>> rollingResultFuture =
        QueryUtil.evalAll(env, context, operands.get(0));
    for (int i = 1; i < operands.size(); i++) {
      final int index = i;
      Function<ThreadSafeMutableSet<T>, QueryTaskFuture<ThreadSafeMutableSet<T>>>
          evalOperandAndIntersectAsyncFunction =
              rollingResult -> {
                final QueryTaskFuture<ThreadSafeMutableSet<T>> rhsOperandValueFuture =
                    QueryUtil.evalAll(env, context, operands.get(index));
                return env.whenSucceedsCall(
                    rhsOperandValueFuture,
                    new QueryTaskCallable<ThreadSafeMutableSet<T>>() {
                      @Override
                      public ThreadSafeMutableSet<T> call()
                          throws QueryException, InterruptedException {
                        rollingResult.retainAll(rhsOperandValueFuture.getIfSuccessful());
                        return rollingResult;
                      }
                    });
              };
      rollingResultFuture =
          env.transformAsync(rollingResultFuture, evalOperandAndIntersectAsyncFunction);
    }
    final QueryTaskFuture<ThreadSafeMutableSet<T>> resultFuture = rollingResultFuture;
    return env.whenSucceedsCall(
        resultFuture,
        new QueryTaskCallable<Void>() {
          @Override
          public Void call() throws QueryException, InterruptedException {
            callback.process(resultFuture.getIfSuccessful());
            return null;
          }
        });
  }

  @Override
  public void collectTargetPatterns(Collection<String> literals) {
    for (QueryExpression subExpression : operands) {
      subExpression.collectTargetPatterns(literals);
    }
  }

  @Override
  public <T, C> T accept(QueryExpressionVisitor<T, C> visitor, C context) {
    return visitor.visit(this, context);
  }

  @Override
  public String toString() {
    StringBuilder result = new StringBuilder();
    result.append("(");
    result.append(operands.get(0));
    for (QueryExpression expr : operands.subList(1, operands.size())) {
      result.append(" ").append(operator.getPrettyName()).append(" ").append(expr);
    }
    result.append(")");
    return result.toString();
  }
}
