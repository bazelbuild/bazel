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

import com.google.common.base.Preconditions;
import javax.annotation.Nullable;

/** A syntax node for return statements. */
public final class ReturnStatement extends Statement {

  private final int returnOffset;
  @Nullable private final Expression result;

  ReturnStatement(int returnOffset, @Nullable Expression result) {
    this.returnOffset = returnOffset;
    this.result = result;
  }

  /**
   * Returns a new return statement that returns expr. It has a dummy file offset and line number
   * table. It is provided only for use by the evaluator, and will be removed when it switches to a
   * compiled representation.
   */
  public static ReturnStatement make(Expression expr) {
    ReturnStatement stmt = new ReturnStatement(0, expr);
    stmt.lnt = Preconditions.checkNotNull(expr.lnt);
    return stmt;
  }

  // TODO(adonovan): rename to getResult.
  @Nullable
  public Expression getReturnExpression() {
    return result;
  }

  @Override
  public int getStartOffset() {
    return returnOffset;
  }

  @Override
  public int getEndOffset() {
    return result != null ? result.getEndOffset() : returnOffset + "return".length();
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.RETURN;
  }
}
