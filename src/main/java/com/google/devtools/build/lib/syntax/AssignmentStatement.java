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

import com.google.common.base.Optional;
import com.google.devtools.build.lib.syntax.compiler.DebugInfo;
import com.google.devtools.build.lib.syntax.compiler.LoopLabels;
import com.google.devtools.build.lib.syntax.compiler.VariableScope;

import net.bytebuddy.implementation.bytecode.ByteCodeAppender;

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
  void doExec(Environment env) throws EvalException, InterruptedException {
    Object rvalue = expression.eval(env);
    lvalue.assign(env, getLocation(), rvalue);
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  void validate(ValidationEnvironment env) throws EvalException {
    expression.validate(env);
    lvalue.validate(env, getLocation());
  }

  @Override
  ByteCodeAppender compile(
      VariableScope scope, Optional<LoopLabels> loopLabels, DebugInfo debugInfo)
          throws EvalException {
    return new ByteCodeAppender.Compound(
        expression.compile(scope, debugInfo),
        lvalue.compileAssignment(this, debugInfo.add(this), scope));
  }
}
