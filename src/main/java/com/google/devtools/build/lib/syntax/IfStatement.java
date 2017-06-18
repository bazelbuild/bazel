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
import com.google.devtools.build.lib.util.Preconditions;
import java.util.List;

/**
 * Syntax node for an if/else statement.
 */
public final class IfStatement extends Statement {

  /**
   * Syntax node for an [el]if statement.
   */
  public static final class ConditionalStatements extends Statement {

    private final Expression condition;
    private final ImmutableList<Statement> stmts;

    public ConditionalStatements(Expression condition, List<Statement> stmts) {
      this.condition = Preconditions.checkNotNull(condition);
      this.stmts = ImmutableList.copyOf(stmts);
    }

    @Override
    void doExec(Environment env) throws EvalException, InterruptedException {
      for (Statement stmt : stmts) {
        stmt.exec(env);
      }
    }

    @Override
    public String toString() {
      return "[el]if " + condition + ": " + stmts + "\n";
    }

    @Override
    public void accept(SyntaxTreeVisitor visitor) {
      visitor.visit(this);
    }

    public Expression getCondition() {
      return condition;
    }

    public ImmutableList<Statement> getStmts() {
      return stmts;
    }

    @Override
    void validate(ValidationEnvironment env) throws EvalException {
      condition.validate(env);
      validateStmts(env, stmts);
    }
  }

  private final ImmutableList<ConditionalStatements> thenBlocks;
  private final ImmutableList<Statement> elseBlock;

  /**
   * Constructs a if-elif-else statement. The else part is mandatory, but the list may be empty.
   * ThenBlocks has to have at least one element.
   */
  public IfStatement(List<ConditionalStatements> thenBlocks, List<Statement> elseBlock) {
    Preconditions.checkArgument(!thenBlocks.isEmpty());
    this.thenBlocks = ImmutableList.copyOf(thenBlocks);
    this.elseBlock = ImmutableList.copyOf(elseBlock);
  }

  public ImmutableList<ConditionalStatements> getThenBlocks() {
    return thenBlocks;
  }

  public ImmutableList<Statement> getElseBlock() {
    return elseBlock;
  }

  @Override
  public String toString() {
    // TODO(bazel-team): if we want to print the complete statement, the function
    // needs an extra argument to specify indentation level.
    // As guaranteed by the constructor, there must be at least one element in thenBlocks.
    return String.format("if %s:\n", thenBlocks.get(0).getCondition());
  }

  @Override
  void doExec(Environment env) throws EvalException, InterruptedException {
    for (ConditionalStatements stmt : thenBlocks) {
      if (EvalUtils.toBoolean(stmt.getCondition().eval(env))) {
        stmt.exec(env);
        return;
      }
    }
    for (Statement stmt : elseBlock) {
      stmt.exec(env);
    }
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  void validate(ValidationEnvironment env) throws EvalException {
    env.startTemporarilyDisableReadonlyCheckSession();
    for (ConditionalStatements stmts : thenBlocks) {
      stmts.validate(env);
    }
    validateStmts(env, elseBlock);
    env.finishTemporarilyDisableReadonlyCheckSession();
  }

  private static void validateStmts(ValidationEnvironment env, List<Statement> stmts)
      throws EvalException {
    for (Statement stmt : stmts) {
      stmt.validate(env);
    }
    env.finishTemporarilyDisableReadonlyCheckBranch();
  }
}
