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
class BinaryOperatorExpression extends QueryExpression {

  private final Lexer.TokenKind operator; // ::= INTERSECT/CARET | UNION/PLUS | EXCEPT/MINUS
  private final ImmutableList<QueryExpression> operands;

  BinaryOperatorExpression(Lexer.TokenKind operator,
                           List<QueryExpression> operands) {
    Preconditions.checkState(operands.size() > 1);
    this.operator = operator;
    this.operands = ImmutableList.copyOf(operands);
  }

  Lexer.TokenKind getOperator() {
    return operator;
  }

  ImmutableList<QueryExpression> getOperands() {
    return operands;
  }

  @Override
  public <T> void eval(QueryEnvironment<T> env, Callback<T> callback)
      throws QueryException, InterruptedException {

    if (operator == TokenKind.PLUS || operator == TokenKind.UNION) {
      for (QueryExpression operand : operands) {
        env.eval(operand, callback);
      }
      return;
    }
    // We cannot do differences with partial results. So we fully evaluate the operands
    Set<T> lhsValue = QueryUtil.evalAll(env, operands.get(0));
    for (int i = 1; i < operands.size(); i++) {
      Set<T> rhsValue = QueryUtil.evalAll(env, operands.get(i));
      switch (operator) {
        case INTERSECT:
        case CARET:
          lhsValue.retainAll(rhsValue);
          break;
        case EXCEPT:
        case MINUS:
          lhsValue.removeAll(rhsValue);
          break;
        case UNION:
        case PLUS:
        default:
          throw new IllegalStateException("operator=" + operator);
      }
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
