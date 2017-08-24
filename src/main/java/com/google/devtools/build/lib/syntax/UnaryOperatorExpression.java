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
import java.io.IOException;

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
  public void prettyPrint(Appendable buffer) throws IOException {
    // TODO(bazel-team): Possibly omit parentheses when they are not needed according to operator
    // precedence rules. This requires passing down more contextual information.
    buffer.append(operator.toString());
    buffer.append('(');
    operand.prettyPrint(buffer);
    buffer.append(')');
  }

  @Override
  public String toString() {
    // All current and planned unary operators happen to be prefix operators.
    // Non-symbolic operators have trailing whitespace built into their name.
    //
    // Note that this omits the parentheses for brevity, but is not correct in general due to
    // operator precedence rules. For example, "(not False) in mylist" prints as
    // "not False in mylist", which evaluates to opposite results in the case that mylist is empty.
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
              String.format(
                  "unsupported operand type for -: '%s'", EvalUtils.getDataTypeName(value)));
        }
        if (env.getSemantics().incompatibleCheckedArithmetic) {
          try {
            return Math.negateExact((Integer) value);
          } catch (ArithmeticException e) {
            // Fails for -MIN_INT.
            throw new EvalException(loc, e.getMessage());
          }
        } else {
          return -((Integer) value);
        }

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
  public Kind kind() {
    return Kind.UNARY_OPERATOR;
  }
}
