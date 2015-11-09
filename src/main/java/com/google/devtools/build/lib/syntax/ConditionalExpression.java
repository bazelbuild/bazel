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

import static com.google.devtools.build.lib.syntax.compiler.ByteCodeUtils.append;

import com.google.devtools.build.lib.syntax.compiler.ByteCodeUtils;
import com.google.devtools.build.lib.syntax.compiler.DebugInfo;
import com.google.devtools.build.lib.syntax.compiler.Jump;
import com.google.devtools.build.lib.syntax.compiler.Jump.PrimitiveComparison;
import com.google.devtools.build.lib.syntax.compiler.LabelAdder;
import com.google.devtools.build.lib.syntax.compiler.VariableScope;

import net.bytebuddy.implementation.bytecode.ByteCodeAppender;

import java.util.ArrayList;
import java.util.List;

/**
 * Syntax node for an if/else expression.
 */
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
  public String toString() {
    return thenCase + " if " + condition + " else " + elseCase;
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
  void validate(ValidationEnvironment env) throws EvalException {
    condition.validate(env);
    thenCase.validate(env);
    elseCase.validate(env);
  }

  @Override
  ByteCodeAppender compile(VariableScope scope, DebugInfo debugInfo) throws EvalException {
    List<ByteCodeAppender> code = new ArrayList<>();
    LabelAdder afterLabel = new LabelAdder();
    LabelAdder elseLabel = new LabelAdder();
    // compile condition and convert to boolean
    code.add(condition.compile(scope, debugInfo));
    append(
        code,
        EvalUtils.toBoolean,
        // jump to else block if false
        Jump.ifIntOperandToZero(PrimitiveComparison.EQUAL).to(elseLabel));
    // otherwise evaluate the expression for "then" and jump to end
    code.add(thenCase.compile(scope, debugInfo));
    append(
        code,
        Jump.to(afterLabel),
        // add label for "else" and evaluate the expression
        elseLabel);
    code.add(elseCase.compile(scope, debugInfo));
    append(code, afterLabel);

    return ByteCodeUtils.compoundAppender(code);
  }
}
