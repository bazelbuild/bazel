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
import java.io.IOException;
import java.util.List;

/** Syntax node for a for loop statement. */
public final class ForStatement extends Statement {

  private final Expression lhs;
  private final Expression collection;
  private final ImmutableList<Statement> block;

  /** Constructs a for loop statement. */
  public ForStatement(Expression lhs, Expression collection, List<Statement> block) {
    this.lhs = Preconditions.checkNotNull(lhs);
    this.collection = Preconditions.checkNotNull(collection);
    this.block = ImmutableList.copyOf(block);
  }

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
  public void prettyPrint(Appendable buffer, int indentLevel) throws IOException {
    printIndent(buffer, indentLevel);
    buffer.append("for ");
    lhs.prettyPrint(buffer);
    buffer.append(" in ");
    collection.prettyPrint(buffer);
    buffer.append(":\n");
    printSuite(buffer, block, indentLevel);
  }

  @Override
  public String toString() {
    return "for " + lhs + " in " + collection + ": ...\n";
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.FOR;
  }
}
