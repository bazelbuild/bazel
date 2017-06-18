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

import com.google.devtools.build.lib.events.Location;

/**
 * A wrapper Statement class for return expressions.
 */
public final class ReturnStatement extends Statement {

  /**
   * Exception sent by the return statement, to be caught by the function body.
   */
  public static class ReturnException extends EvalException {
    Object value;

    public ReturnException(Location location, Object value) {
      super(
          location,
          "return statements must be inside a function",
          /*dueToIncompleteAST=*/ false, /*fillInJavaStackTrace=*/
          false);
      this.value = value;
    }

    public Object getValue() {
      return value;
    }

    @Override
    public boolean canBeAddedToStackTrace() {
      return false;
    }
  }

  private final Expression returnExpression;

  public ReturnStatement(Expression returnExpression) {
    this.returnExpression = returnExpression;
  }

  @Override
  void doExec(Environment env) throws EvalException, InterruptedException {
    throw new ReturnException(returnExpression.getLocation(), returnExpression.eval(env));
  }

  public Expression getReturnExpression() {
    return returnExpression;
  }

  @Override
  public String toString() {
    return "return " + returnExpression;
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  void validate(ValidationEnvironment env) throws EvalException {
    if (env.isTopLevel()) {
      throw new EvalException(getLocation(), "Return statements must be inside a function");
    }
    returnExpression.validate(env);
  }
}
