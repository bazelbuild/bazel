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
import com.google.devtools.build.lib.syntax.FlowStatement.FlowException;

import java.util.List;

/**
 * Syntax node for a for loop statement.
 */
public final class ForStatement extends Statement {

  private final LValue variable;
  private final Expression collection;
  private final ImmutableList<Statement> block;

  /**
   * Constructs a for loop statement.
   */
  ForStatement(Expression variable, Expression collection, List<Statement> block) {
    this.variable = new LValue(Preconditions.checkNotNull(variable));
    this.collection = Preconditions.checkNotNull(collection);
    this.block = ImmutableList.copyOf(block);
  }

  public LValue getVariable() {
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
    Iterable<?> col = EvalUtils.toIterable(o, getLocation());

    int i = 0;
    for (Object it : ImmutableList.copyOf(col)) {
      variable.assign(env, getLocation(), it);

      try {
        for (Statement stmt : block) {
          stmt.exec(env);
        }
      } catch (FlowException ex) {
        if (ex.mustTerminateLoop()) {
          return;
        }
      }

      i++;
    }
    
    // TODO(bazel-team): This should not happen if every collection is immutable.
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
    if (env.isTopLevel()) {
      throw new EvalException(getLocation(), "'For' is not allowed as a top level statement");
    }
    env.enterLoop();

    try {
      // TODO(bazel-team): validate variable. Maybe make it temporarily readonly.
      collection.validate(env);
      variable.validate(env, getLocation());

      for (Statement stmt : block) {
        stmt.validate(env);
      }
    } finally {
      env.exitLoop(getLocation());
    }
  }
}
