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
  private final Expression vars;
  private final Expression collection;
  private final ImmutableList<Statement> body; // non-empty if well formed

  /** Constructs a for loop statement. */
  ForStatement(
      FileLocations locs,
      int forOffset,
      Expression vars,
      Expression collection,
      List<Statement> body) {
    super(locs);
    this.forOffset = forOffset;
    this.vars = Preconditions.checkNotNull(vars);
    this.collection = Preconditions.checkNotNull(collection);
    this.body = ImmutableList.copyOf(body);
  }

  public Expression getVars() {
    return vars;
  }

  /**
   * @return The collection we iterate on, e.g. `col` in `for x in col:`
   */
  public Expression getCollection() {
    return collection;
  }

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
        ? collection.getEndOffset() // wrong, but tree is ill formed
        : body.get(body.size() - 1).getEndOffset();
  }

  @Override
  public String toString() {
    return "for " + vars + " in " + collection + ": ...\n";
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
