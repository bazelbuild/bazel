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
package com.google.devtools.build.lib.syntax;

import com.google.devtools.build.lib.events.Location;

/** Syntax node for a unary operator expression. */
public final class UnaryOperatorExpression extends Expression {

  private final UnaryOperator operator;

  private final Expression operand;

  public UnaryOperatorExpression(UnaryOperator operator, Expression operand) {
    this.operator = operator;
    this.operand = operand;
  }

  public UnaryOperator getOperator() {
    return operator;
  }

  public Expression getOperand() {
    return operand;
  }

  @Override
  public String toString() {
    // All current and planned unary operators happen to be prefix operators.
    // Non-symbolic operators have trailing whitespace built into their name.
    return operator.toString() + operand;
  }

  private static Object evaluate(
      UnaryOperator operator,
      Object value,
      Environment env,
      Location loc)
      throws EvalException, InterruptedException {
    switch (operator) {
      case NOT:
        return !EvalUtils.toBoolean(value);

      case MINUS:
        if (!(value instanceof Integer)) {
          throw new EvalException(
              loc,
              String.format("unsupported operand type for -: '%s'",
                  EvalUtils.getDataTypeName(value)));
        }
        return -((Integer) value);

      default:
        throw new AssertionError("Unsupported unary operator: " + operator);
    }
  }

  @Override
  Object doEval(Environment env) throws EvalException, InterruptedException {
    return evaluate(operator, operand.eval(env), env, getLocation());
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  void validate(ValidationEnvironment env) throws EvalException {
    operand.validate(env);
  }
}
