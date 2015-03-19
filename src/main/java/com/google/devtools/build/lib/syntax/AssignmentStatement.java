// Copyright 2014 Google Inc. All rights reserved.
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

/**
 * Syntax node for an assignment statement.
 */
public final class AssignmentStatement extends Statement {

  private final LValue lvalue;

  private final Expression expression;

  /**
   *  Constructs an assignment: "lvalue := value".
   */
  AssignmentStatement(Expression lvalue, Expression expression) {
    this.lvalue = new LValue(lvalue);
    this.expression = expression;
  }

  /**
   *  Returns the LHS of the assignment.
   */
  public LValue getLValue() {
    return lvalue;
  }

  /**
   *  Returns the RHS of the assignment.
   */
  public Expression getExpression() {
    return expression;
  }

  @Override
  public String toString() {
    return lvalue + " = " + expression + '\n';
  }

  @Override
  void exec(Environment env) throws EvalException, InterruptedException {
    lvalue.assign(env, getLocation(), expression);
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  void validate(ValidationEnvironment env) throws EvalException {
    lvalue.validate(env, getLocation(), expression.validate(env));
  }
}
