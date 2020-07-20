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

import com.google.common.collect.ImmutableList;
import java.util.List;
import javax.annotation.Nullable;

/** Syntax node for an if or elif statement. */
public final class IfStatement extends Statement {

  private final TokenKind token; // IF or ELIF
  private final int ifOffset;
  private final Expression condition;
  // These blocks may be non-null but empty after a misparse:
  private final ImmutableList<Statement> thenBlock; // non-empty
  @Nullable ImmutableList<Statement> elseBlock; // non-empty if non-null; set after construction

  IfStatement(
      FileLocations locs,
      TokenKind token,
      int ifOffset,
      Expression condition,
      List<Statement> thenBlock) {
    super(locs);
    this.token = token;
    this.ifOffset = ifOffset;
    this.condition = condition;
    this.thenBlock = ImmutableList.copyOf(thenBlock);
  }

  /**
   * Reports whether this is an 'elif' statement.
   *
   * <p>An elif statement may appear only as the sole statement in the "else" block of another
   * IfStatement.
   */
  public boolean isElif() {
    return token == TokenKind.ELIF;
  }

  public Expression getCondition() {
    return condition;
  }

  public ImmutableList<Statement> getThenBlock() {
    return thenBlock;
  }

  @Nullable
  public ImmutableList<Statement> getElseBlock() {
    return elseBlock;
  }

  void setElseBlock(List<Statement> elseBlock) {
    this.elseBlock = ImmutableList.copyOf(elseBlock);
  }

  @Override
  public int getStartOffset() {
    return ifOffset;
  }

  @Override
  public int getEndOffset() {
    List<Statement> body = elseBlock != null ? elseBlock : thenBlock;
    return body.isEmpty()
        ? condition.getEndOffset() // wrong, but tree is ill formed
        : body.get(body.size() - 1).getEndOffset();
  }

  @Override
  public String toString() {
    return String.format("if %s: ...\n", condition);
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.IF;
  }
}
