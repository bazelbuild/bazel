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
import java.io.IOException;
import java.util.List;

/**
 * Syntax node for an if/else statement.
 */
public final class IfStatement extends Statement {

  /**
   * Syntax node for an [el]if statement.
   *
   * <p>This extends Statement because it implements {@code doExec}, but it is not actually an
   * independent statement in the grammar.
   */
  public static final class ConditionalStatements extends Statement {

    private final Expression condition;
    private final ImmutableList<Statement> statements;

    public ConditionalStatements(Expression condition, List<Statement> statements) {
      this.condition = Preconditions.checkNotNull(condition);
      this.statements = ImmutableList.copyOf(statements);
    }

    @Override
    void doExec(Environment env) throws EvalException, InterruptedException {
      for (Statement stmt : statements) {
        stmt.exec(env);
      }
    }

    // No prettyPrint function; handled directly by IfStatement#prettyPrint.
    @Override
    public void prettyPrint(Appendable buffer, int indentLevel) throws IOException {
      throw new UnsupportedOperationException("Cannot pretty print ConditionalStatements node");
    }

    @Override
    public String toString() {
      return "[el]if " + condition + ": " + statements + "\n";
    }

    @Override
    public void accept(SyntaxTreeVisitor visitor) {
      visitor.visit(this);
    }

    public Expression getCondition() {
      return condition;
    }

    public ImmutableList<Statement> getStatements() {
      return statements;
    }
  }

  /** "if" or "elif" clauses. Must be non-empty. */
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
  public void prettyPrint(Appendable buffer, int indentLevel) throws IOException {
    String clauseWord = "if ";
    for (ConditionalStatements condStmt : thenBlocks) {
      printIndent(buffer, indentLevel);
      buffer.append(clauseWord);
      condStmt.getCondition().prettyPrint(buffer);
      buffer.append(":\n");
      printSuite(buffer, condStmt.getStatements(), indentLevel);
      clauseWord = "elif ";
    }
    if (!elseBlock.isEmpty()) {
      printIndent(buffer, indentLevel);
      buffer.append("else:\n");
      printSuite(buffer, elseBlock, indentLevel);
    }
  }

  @Override
  public String toString() {
    return String.format("if %s: ...\n", thenBlocks.get(0).getCondition());
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
}
