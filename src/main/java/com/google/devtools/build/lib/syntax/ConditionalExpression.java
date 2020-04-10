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

/** Syntax node for an expression of the form {@code t if cond else f}. */
public final class ConditionalExpression extends Expression {

  private final Expression t;
  private final Expression cond;
  private final Expression f;

  public Expression getThenCase() {
    return t;
  }

  public Expression getCondition() {
    return cond;
  }

  public Expression getElseCase() {
    return f;
  }

  /** Constructor for a conditional expression */
  ConditionalExpression(FileLocations locs, Expression t, Expression cond, Expression f) {
    super(locs);
    this.t = t;
    this.cond = cond;
    this.f = f;
  }

  @Override
  public int getStartOffset() {
    return t.getStartOffset();
  }

  @Override
  public int getEndOffset() {
    return f.getEndOffset();
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.CONDITIONAL;
  }
}
