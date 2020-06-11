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

import javax.annotation.Nullable;

/**
 * Syntax node for an assignment statement ({@code lhs = rhs}) or augmented assignment statement
 * ({@code lhs op= rhs}).
 */
public final class AssignmentStatement extends Statement {

  private final Expression lhs; // = IDENTIFIER | DOT | INDEX | LIST_EXPR
  @Nullable private final TokenKind op; // TODO(adonovan): make this mandatory even when '='.
  private final int opOffset;
  private final Expression rhs;

  /**
   * Constructs an assignment statement. For an ordinary assignment ({@code op == null}), the LHS
   * expression must be of the form {@code id}, {@code x.y}, {@code x[i]}, {@code [e, ...]}, or
   * {@code (e, ...)}, where x, i, and e are arbitrary expressions. For an augmented assignment, the
   * list and tuple forms are disallowed.
   */
  AssignmentStatement(
      FileLocations locs, Expression lhs, @Nullable TokenKind op, int opOffset, Expression rhs) {
    super(locs);
    this.lhs = lhs;
    this.op = op;
    this.opOffset = opOffset;
    this.rhs = rhs;
  }

  /** Returns the LHS of the assignment. */
  public Expression getLHS() {
    return lhs;
  }

  /** Returns the operator of an augmented assignment, or null for an ordinary assignment. */
  @Nullable
  public TokenKind getOperator() {
    return op;
  }

  /** Returns the location of the assignment operator. */
  public Location getOperatorLocation() {
    return locs.getLocation(opOffset);
  }

  @Override
  public int getStartOffset() {
    return lhs.getStartOffset();
  }

  @Override
  public int getEndOffset() {
    return rhs.getEndOffset();
  }

  /** Reports whether this is an augmented assignment ({@code getOperator() != null}). */
  public boolean isAugmented() {
    return op != null;
  }

  /** Returns the RHS of the assignment. */
  public Expression getRHS() {
    return rhs;
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.ASSIGNMENT;
  }
}
