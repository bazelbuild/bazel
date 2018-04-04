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
import java.io.IOException;
import javax.annotation.Nullable;

/** A wrapper Statement class for return expressions. */
public final class ReturnStatement extends Statement {

  /**
   * Exception sent by the return statement, to be caught by the function body.
   */
  public static class ReturnException extends EvalException {
    private final Object value;

    public ReturnException(Location location, Object value) {
      super(
          location,
          "return statements must be inside a function",
          /*dueToIncompleteAST=*/false,
          /*fillInJavaStackTrace=*/false);
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

  @Nullable private final Expression returnExpression;

  public ReturnStatement(@Nullable Expression returnExpression) {
    this.returnExpression = returnExpression;
  }

  @Nullable
  public Expression getReturnExpression() {
    return returnExpression;
  }

  @Override
  public void prettyPrint(Appendable buffer, int indentLevel) throws IOException {
    printIndent(buffer, indentLevel);
    buffer.append("return");
    if (returnExpression != null) {
      buffer.append(' ');
      returnExpression.prettyPrint(buffer, indentLevel);
    }
    buffer.append('\n');
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.RETURN;
  }
}
