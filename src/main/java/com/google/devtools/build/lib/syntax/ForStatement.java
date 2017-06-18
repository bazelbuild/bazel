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
import com.google.devtools.build.lib.syntax.FlowStatement.FlowException;
import com.google.devtools.build.lib.util.Preconditions;
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
  public ForStatement(LValue variable, Expression collection, List<Statement> block) {
    this.variable = Preconditions.checkNotNull(variable);
    this.collection = Preconditions.checkNotNull(collection);
    this.block = ImmutableList.copyOf(block);
  }

  public LValue getVariable() {
    return variable;
  }

  /**
   * @return The collection we iterate on, e.g. `col` in `for x in col:`
   */
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
  void doExec(Environment env) throws EvalException, InterruptedException {
    Object o = collection.eval(env);
    Iterable<?> col = EvalUtils.toIterable(o, getLocation());
    EvalUtils.lock(o, getLocation());
    try {
      for (Object it : col) {
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
      }
    } finally {
      EvalUtils.unlock(o, getLocation());
    }
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  void validate(ValidationEnvironment env) throws EvalException {
    // TODO(bazel-team): validate variable. Maybe make it temporarily readonly.
    collection.validate(env);
    variable.validate(env, getLocation());

    for (Statement stmt : block) {
      stmt.validate(env);
    }
  }
}
