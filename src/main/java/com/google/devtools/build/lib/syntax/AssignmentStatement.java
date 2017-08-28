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

/**
 * Syntax node for an assignment statement.
 */
public final class AssignmentStatement extends Statement {

  private final LValue lvalue;

  private final Expression expression;

  /**
   *  Constructs an assignment: "lvalue := value".
   */
  public AssignmentStatement(LValue lvalue, Expression expression) {
    this.lvalue = lvalue;
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
  public void prettyPrint(Appendable buffer, int indentLevel) throws IOException {
    printIndent(buffer, indentLevel);
    lvalue.prettyPrint(buffer, indentLevel);
    buffer.append(" = ");
    expression.prettyPrint(buffer, indentLevel);
    buffer.append('\n');
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.ASSIGNMENT;
  }
}
