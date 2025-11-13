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

package net.starlark.java.syntax;

import com.google.common.base.Preconditions;
import javax.annotation.Nullable;

/**
 * Syntax node for an assignment statement ({@code lhs = rhs}) or augmented assignment statement
 * ({@code lhs op= rhs}).
 */
public final class AssignmentStatement extends Statement {

  private final Expression lhs; // = IDENTIFIER | DOT | INDEX | LIST_EXPR

  // non-null only when lhs is an identifier and we're not augmented
  @Nullable private final Expression type;

  @Nullable private final TokenKind op; // TODO(adonovan): make this mandatory even when '='.
  private final int opOffset;

  private final Expression rhs;

  @Nullable private final DocComments docComments;

  /**
   * Constructs an assignment statement. For an ordinary assignment ({@code op == null}), the LHS
   * expression must be of the form {@code id}, {@code x.y}, {@code x[i]}, {@code [e, ...]}, or
   * {@code (e, ...)}, where x, i, and e are arbitrary expressions. For an augmented assignment, the
   * list and tuple forms are disallowed.
   *
   * <p>If a type annotation is present ({@code x : T = ...}), the LHS expression must be an
   * identifier, and the assignment must not be augmented.
   */
  AssignmentStatement(
      FileLocations locs,
      Expression lhs,
      @Nullable Expression type,
      @Nullable TokenKind op,
      int opOffset,
      Expression rhs,
      @Nullable DocComments docComments) {
    super(locs, Kind.ASSIGNMENT);
    this.lhs = lhs;
    this.type = type;
    this.op = op;
    this.opOffset = opOffset;
    this.rhs = rhs;
    this.docComments = docComments;
    if (type != null) {
      Preconditions.checkState(
          lhs.kind() == Expression.Kind.IDENTIFIER, "Can't have type annotation on complex LHS");
      Preconditions.checkState(op == null, "Can't have augmented assignment with type annotation");
    }
  }

  /** Returns the LHS of the assignment. */
  public Expression getLHS() {
    return lhs;
  }

  /** Returns the type expression (if present) of the variable on the LHS. */
  @Nullable
  public Expression getType() {
    return type;
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

  /** Returns the Sphinx autodoc-style doc comments attached to this statement, if any. */
  @Nullable
  public DocComments getDocComments() {
    return docComments;
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }
}
