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

/**
 * Syntax node for list and dict comprehensions.
 *
 * <p>A comprehension contains one or more clauses, e.g. [a+d for a in b if c for d in e] contains
 * three clauses: "for a in b", "if c", "for d in e". For and If clauses can happen in any order,
 * except that the first one has to be a For.
 *
 * <p>The code above can be expanded as:
 *
 * <pre>
 *   for a in b:
 *     if c:
 *       for d in e:
 *         result.append(a+d)
 * </pre>
 *
 * result is initialized to [] (list) or {} (dict) and is the return value of the whole expression.
 */
public final class Comprehension extends Expression {

  /** For or If */
  public abstract static class Clause extends Node {
    Clause(FileLocations locs) {
      super(locs);
    }
  }

  /** A for clause in a comprehension, e.g. "for a in b" in the example above. */
  public static final class For extends Clause {
    private final int forOffset;
    private final Expression vars;
    private final Expression iterable;

    For(FileLocations locs, int forOffset, Expression vars, Expression iterable) {
      super(locs);
      this.forOffset = forOffset;
      this.vars = vars;
      this.iterable = iterable;
    }

    public Expression getVars() {
      return vars;
    }

    public Expression getIterable() {
      return iterable;
    }

    @Override
    public int getStartOffset() {
      return forOffset;
    }

    @Override
    public int getEndOffset() {
      return iterable.getEndOffset();
    }

    @Override
    public void accept(NodeVisitor visitor) {
      visitor.visit(this);
    }
  }

  /** A if clause in a comprehension, e.g. "if c" in the example above. */
  public static final class If extends Clause {
    private final int ifOffset;
    private final Expression condition;

    If(FileLocations locs, int ifOffset, Expression condition) {
      super(locs);
      this.ifOffset = ifOffset;
      this.condition = condition;
    }

    public Expression getCondition() {
      return condition;
    }

    @Override
    public int getStartOffset() {
      return ifOffset;
    }

    @Override
    public int getEndOffset() {
      return condition.getEndOffset();
    }

    @Override
    public void accept(NodeVisitor visitor) {
      visitor.visit(this);
    }
  }

  private final boolean isDict; // {k: v for vars in iterable}
  private final int lbracketOffset;
  private final Node body; // Expression or DictExpression.Entry
  private final ImmutableList<Clause> clauses;
  private final int rbracketOffset;

  Comprehension(
      FileLocations locs,
      boolean isDict,
      int lbracketOffset,
      Node body,
      ImmutableList<Clause> clauses,
      int rbracketOffset) {
    super(locs);
    this.isDict = isDict;
    this.lbracketOffset = lbracketOffset;
    this.body = body;
    this.clauses = clauses;
    this.rbracketOffset = rbracketOffset;
  }

  public boolean isDict() {
    return isDict;
  }

  /**
   * Returns the loop body: an expression for a list comprehension, or a DictExpression.Entry for a
   * dict comprehension.
   */
  public Node getBody() {
    return body;
  }

  public ImmutableList<Clause> getClauses() {
    return clauses;
  }

  @Override
  public int getStartOffset() {
    return lbracketOffset;
  }

  @Override
  public int getEndOffset() {
    return rbracketOffset + 1;
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.COMPREHENSION;
  }

}
