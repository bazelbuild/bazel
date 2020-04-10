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
import com.google.common.collect.ImmutableList;
import java.util.List;

/** Syntax node for a for loop statement. */
public final class ForStatement extends Statement {

  private final int forOffset;
  private final Expression lhs;
  private final Expression collection;
  private final ImmutableList<Statement> block; // non-empty if well formed

  /** Constructs a for loop statement. */
  ForStatement(int forOffset, Expression lhs, Expression collection, List<Statement> block) {
    this.forOffset = forOffset;
    this.lhs = Preconditions.checkNotNull(lhs);
    this.collection = Preconditions.checkNotNull(collection);
    this.block = ImmutableList.copyOf(block);
  }

  // TODO(adonovan): rename to getVars.
  public Expression getLHS() {
    return lhs;
  }

  /**
   * @return The collection we iterate on, e.g. `col` in `for x in col:`
   */
  public Expression getCollection() {
    return collection;
  }

  public ImmutableList<Statement> getBlock() {
    return block;
  }

  @Override
  public int getStartOffset() {
    return forOffset;
  }

  @Override
  public int getEndOffset() {
    return block.isEmpty()
        ? collection.getEndOffset() // wrong, but tree is ill formed
        : block.get(block.size() - 1).getEndOffset();
  }

  @Override
  public String toString() {
    return "for " + lhs + " in " + collection + ": ...\n";
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.FOR;
  }
}
