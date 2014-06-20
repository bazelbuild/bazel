// Copyright 2014 Google Inc. All rights reserved.
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
import java.util.Map;

/**
 * Syntax node for a for loop statement.
 */
public final class ForStatement extends Statement {

  private final Ident variable;
  private final Expression collection;
  private final ImmutableList<Statement> block;

  /**
   * Constructs a for loop statement.
   */
  ForStatement(Ident variable, Expression collection, List<Statement> block) {
    this.variable = Preconditions.checkNotNull(variable);
    this.collection = Preconditions.checkNotNull(collection);
    this.block = ImmutableList.copyOf(block);
  }

  public Ident getVariable() {
    return variable;
  }

  public Expression getCollection() {
    return collection;
  }

  public ImmutableList<Statement> block() {
    return block;
  }

  @Override
  public String toString() {
    // TODO(bazel-team): if we want to print the complete statement, the function
    // needs an extra argument to specify indentation level.
    return "for " + variable + " in " + collection + ": ...\n";
  }

  @Override
  void exec(Environment env) throws EvalException, InterruptedException {
    Object o = collection.eval(env);
    Iterable<?> col;
    if (o instanceof Iterable) {
      col = (Iterable<?>) o;
    } else if (o instanceof Map<?, ?>) {
      // For dictionaries we iterate through the keys only
      col = ((Map<?, ?>) o).keySet();
    } else {
      throw new EvalException(getLocation(),
          "type '" + EvalUtils.getDatatypeName(o) + "' is not iterable");
    }

    int i = 0;
    // Create a copy of col to prevent infinite loops.
    for (Object it : ImmutableList.copyOf(col)) {
      env.update(variable.getName(), it);
      for (Statement stmt : block) {
        stmt.exec(env);
      }
      i++;
    }
    // TODO(bazel-team): This is not very efficient (we iterate through the Iterable 3 times).
    // It also doesn't catch changes where the size of the Iterable doesn't change, e.g.
    // Map updates with an existing key. However it's good enough to detect infinite loops.
    if (i != EvalUtils.size(col)) {
      throw new EvalException(getLocation(),
          String.format("Cannot modify '%s' during during iteration.", collection.toString()));
    }
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  void validate(ValidationEnvironment env) throws EvalException {
    // TODO(bazel-team): validate variable. Maybe make it temporarily readonly.
    Class<?> type = collection.validate(env);
    env.checkIterable(type, getLocation());
    env.update(variable.getName(), Object.class, getLocation());
    for (Statement stmt : block) {
      stmt.validate(env);
    }
  }
}
