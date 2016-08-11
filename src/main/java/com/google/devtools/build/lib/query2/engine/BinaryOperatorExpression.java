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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.query2.engine.Lexer.TokenKind;
import com.google.devtools.build.lib.util.Preconditions;
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
  public <T> void eval(QueryEnvironment<T> env, VariableContext<T> context, Callback<T> callback)
      throws QueryException, InterruptedException {

    if (operator == TokenKind.PLUS || operator == TokenKind.UNION) {
      for (QueryExpression operand : operands) {
        env.eval(operand, context, callback);
      }
      return;
    }

    // Once we have fully evaluated the left-hand side, we can stream-process the right-hand side
    // for minus operations. Note that this is suboptimal if the left-hand side results are very
    // large compared to the right-hand side. Which is the case is hard to know before evaluating.
    // We could consider determining this dynamically, however, by evaluating both the left and
    // right hand side partially until one side finishes sooner.
    final Set<T> lhsValue = QueryUtil.evalAll(env, context, operands.get(0));
    if (operator == TokenKind.EXCEPT || operator == TokenKind.MINUS) {
      for (int i = 1; i < operands.size(); i++) {
        env.eval(operands.get(i), context,
            new Callback<T>() {
              @Override
              public void process(Iterable<T> partialResult)
                  throws QueryException, InterruptedException {
                for (T target : partialResult) {
                  lhsValue.remove(target);
                }
              }
            });
      }
      callback.process(lhsValue);
      return;
    }

    // Intersection is not associative, so we are forced to pin both the left-hand and right-hand
    // side of the operation at the same time.
    // TODO(bazel-team): Consider keeping just the name / label of the right-hand side results
    // instead of the potentially heavy-weight instances of type T.
    Preconditions.checkState(operator == TokenKind.INTERSECT || operator == TokenKind.CARET,
        operator);
    for (int i = 1; i < operands.size(); i++) {
      lhsValue.retainAll(QueryUtil.evalAll(env, context, operands.get(i)));
    }
    callback.process(lhsValue);
  }

  @Override
  public void collectTargetPatterns(Collection<String> literals) {
    for (QueryExpression subExpression : operands) {
      subExpression.collectTargetPatterns(literals);
    }
  }

  @Override
  public QueryExpression getMapped(QueryExpressionMapper mapper) {
    return mapper.map(this);
  }

  @Override
  public String toString() {
    StringBuilder result = new StringBuilder();
    for (int i = 1; i < operands.size(); i++) {
      result.append("(");
    }
    result.append(operands.get(0));
    for (int i = 1; i < operands.size(); i++) {
      result.append(" " + operator.getPrettyName() + " " + operands.get(i) + ")");
    }
    return result.toString();
  }
}
