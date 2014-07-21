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
 * As syntax node for the not boolean operation.
 */
public class NotExpression extends Expression {

  private final Expression expression;

  public NotExpression(Expression expression) {
    this.expression = expression;
  }

  Expression getExpression() {
    return expression;
  }

  @Override
  Object eval(Environment env) throws EvalException, InterruptedException {
    return !EvalUtils.toBoolean(expression.eval(env));
  }

  @Override
  public String toString() {
    return "not " + expression;
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  SkylarkType validate(ValidationEnvironment env) throws EvalException {
    // Don't need type check here since EvalUtils.toBoolean() converts everything.
    expression.validate(env);
    return SkylarkType.BOOL;
  }
}
