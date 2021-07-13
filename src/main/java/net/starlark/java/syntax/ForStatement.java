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
import com.google.common.collect.ImmutableList;

/** Syntax node for a for loop statement, {@code for vars in iterable: ...}. */
public final class ForStatement extends Statement {

  private final int forOffset;
  private final Expression vars;
  private final Expression iterable;
  private final ImmutableList<Statement> body; // non-empty if well formed

  /** Constructs a for loop statement. */
  ForStatement(
      FileLocations locs,
      int forOffset,
      Expression vars,
      Expression iterable,
      ImmutableList<Statement> body) {
    super(locs, Kind.FOR);
    this.forOffset = forOffset;
    this.vars = Preconditions.checkNotNull(vars);
    this.iterable = Preconditions.checkNotNull(iterable);
    this.body = body;
  }

  /**
   * Returns variables assigned by each iteration. May be a compound target such as {@code (a[b],
   * c.d)}.
   */
  public Expression getVars() {
    return vars;
  }

  /** Returns the iterable value. */
  // TODO(adonovan): rename to getIterable.
  public Expression getCollection() {
    return iterable;
  }

  /** Returns the statements of the loop body. Non-empty if parsing succeeded. */
  public ImmutableList<Statement> getBody() {
    return body;
  }

  @Override
  public int getStartOffset() {
    return forOffset;
  }

  @Override
  public int getEndOffset() {
    return body.isEmpty()
        ? iterable.getEndOffset() // wrong, but tree is ill formed
        : body.get(body.size() - 1).getEndOffset();
  }

  @Override
  public String toString() {
    return "for " + vars + " in " + iterable + ": ...\n";
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }
}
