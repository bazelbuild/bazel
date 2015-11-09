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
 * Syntax node for a function call statement. Used for build rules.
 */
public final class ExpressionStatement extends Statement {

  private final Expression expr;

  public ExpressionStatement(Expression expr) {
    this.expr = expr;
  }

  public Expression getExpression() {
    return expr;
  }

  @Override
  public String toString() {
    return expr.toString() + '\n';
  }

  @Override
  void doExec(Environment env) throws EvalException, InterruptedException {
    expr.eval(env);
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  void validate(ValidationEnvironment env) throws EvalException {
    expr.validate(env);
  }

  @Override
  ByteCodeAppender compile(
      VariableScope scope, Optional<LoopLabels> loopLabels, DebugInfo debugInfo)
      throws EvalException {
    return expr.compile(scope, debugInfo);
  }
}
