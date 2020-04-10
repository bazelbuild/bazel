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

import java.util.EnumSet;

/** A BinaryExpression represents a binary operator expression 'x op y'. */
public final class BinaryOperatorExpression extends Expression {

  private final Expression x;
  private final TokenKind op; // one of 'operators'
  private final int opOffset;
  private final Expression y;

  /** operators is the set of valid binary operators. */
  public static final EnumSet<TokenKind> operators =
      EnumSet.of(
          TokenKind.AND,
          TokenKind.EQUALS_EQUALS,
          TokenKind.GREATER,
          TokenKind.GREATER_EQUALS,
          TokenKind.IN,
          TokenKind.LESS,
          TokenKind.LESS_EQUALS,
          TokenKind.MINUS,
          TokenKind.NOT_EQUALS,
          TokenKind.NOT_IN,
          TokenKind.OR,
          TokenKind.PERCENT,
          TokenKind.SLASH,
          TokenKind.SLASH_SLASH,
          TokenKind.PLUS,
          TokenKind.PIPE,
          TokenKind.STAR);

  BinaryOperatorExpression(
      FileLocations locs, Expression x, TokenKind op, int opOffset, Expression y) {
    super(locs);
    this.x = x;
    this.op = op;
    this.opOffset = opOffset;
    this.y = y;
  }

  /** Returns the left operand. */
  public Expression getX() {
    return x;
  }

  /** Returns the operator. */
  public TokenKind getOperator() {
    return op;
  }

  public Location getOperatorLocation() {
    return locs.getLocation(opOffset);
  }

  /** Returns the right operand. */
  public Expression getY() {
    return y;
  }

  @Override
  public int getStartOffset() {
    return x.getStartOffset();
  }

  @Override
  public int getEndOffset() {
    return y.getEndOffset();
  }

  @Override
  public String toString() {
    // This omits the parentheses for brevity, but is not correct in general due to operator
    // precedence rules.
    return x + " " + op + " " + y;
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.BINARY_OPERATOR;
  }
}
