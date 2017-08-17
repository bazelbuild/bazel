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

package com.google.devtools.build.lib.syntax;

import java.io.IOException;

/** Syntax node for an augmented assignment statement. */
public final class AugmentedAssignmentStatement extends Statement {

  private final Operator operator;

  private final LValue lvalue;

  private final Expression expression;

  /** Constructs an augmented assignment: "lvalue ::= value". */
  public AugmentedAssignmentStatement(Operator operator, LValue lvalue, Expression expression) {
    this.operator = operator;
    this.lvalue = lvalue;
    this.expression = expression;
  }

  /** Returns the operator of the assignment. */
  public Operator getOperator() {
    return operator;
  }

  /** Returns the LValue of the assignment. */
  public LValue getLValue() {
    return lvalue;
  }

  /** Returns the RHS of the assignment. */
  public Expression getExpression() {
    return expression;
  }

  @Override
  public void prettyPrint(Appendable buffer, int indentLevel) throws IOException {
    printIndent(buffer, indentLevel);
    lvalue.prettyPrint(buffer);
    buffer.append(' ');
    buffer.append(operator.toString());
    buffer.append("= ");
    expression.prettyPrint(buffer);
    buffer.append('\n');
  }

  @Override
  void doExec(Environment env) throws EvalException, InterruptedException {
    lvalue.assignAugmented(operator, expression, env, getLocation());
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }
}
