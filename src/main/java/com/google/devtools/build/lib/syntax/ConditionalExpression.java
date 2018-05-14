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

/** Syntax node for an if/else expression. */
public final class ConditionalExpression extends Expression {

  // Python conditional expressions: $thenCase if $condition else $elseCase
  // https://docs.python.org/3.5/reference/expressions.html#conditional-expressions
  private final Expression thenCase;
  private final Expression condition;
  private final Expression elseCase;

  public Expression getThenCase() { return thenCase; }
  public Expression getCondition() { return condition; }
  public Expression getElseCase() { return elseCase; }

  /**
   * Constructor for a conditional expression
   */
  public ConditionalExpression(
      Expression thenCase, Expression condition, Expression elseCase) {
    this.thenCase = thenCase;
    this.condition = condition;
    this.elseCase = elseCase;
  }

  /**
   * Constructs a string representation of the if expression
   */
  @Override
  public void prettyPrint(Appendable buffer) throws IOException {
    thenCase.prettyPrint(buffer);
    buffer.append(" if ");
    condition.prettyPrint(buffer);
    buffer.append(" else ");
    elseCase.prettyPrint(buffer);
  }

  @Override
  Object doEval(Environment env) throws EvalException, InterruptedException {
    if (EvalUtils.toBoolean(condition.eval(env))) {
      return thenCase.eval(env);
    } else {
      return elseCase.eval(env);
    }
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.CONDITIONAL;
  }
}
