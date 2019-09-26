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
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.Parser.ParseResult;
import java.io.IOException;
import java.util.List;
import javax.annotation.Nullable;

/** Syntax tree for a Starlark file, such as a Bazel BUILD or .bzl file. */
public class StarlarkFile extends Node {

  private final ImmutableList<Statement> statements;

  private final ImmutableList<Comment> comments;

  /**
   * Whether any errors were encountered during scanning or parsing.
   */
  private final boolean containsErrors;

  private final List<Event> stringEscapeEvents;

  @Nullable private final String contentHashCode;

  private StarlarkFile(
      ImmutableList<Statement> statements,
      boolean containsErrors,
      String contentHashCode,
      Location location,
      ImmutableList<Comment> comments,
      List<Event> stringEscapeEvents) {
    this.statements = statements;
    this.containsErrors = containsErrors;
    this.contentHashCode = contentHashCode;
    this.comments = comments;
    this.setLocation(location);
    this.stringEscapeEvents = stringEscapeEvents;
  }

  private static StarlarkFile create(
      List<Statement> preludeStatements,
      ParseResult result,
      String contentHashCode,
      boolean allowImportInternal) {
    ImmutableList.Builder<Statement> statementsbuilder =
        ImmutableList.<Statement>builder().addAll(preludeStatements);

    if (allowImportInternal) {
      for (Statement stmt : result.statements) {
        if (stmt instanceof LoadStatement) {
          statementsbuilder.add(LoadStatement.allowLoadingOfInternalSymbols((LoadStatement) stmt));
        } else {
          statementsbuilder.add(stmt);
        }
      }
    } else {
      statementsbuilder.addAll(result.statements);
    }
    ImmutableList<Statement> statements = statementsbuilder.build();
    return new StarlarkFile(
        statements,
        result.containsErrors,
        contentHashCode,
        result.location,
        ImmutableList.copyOf(result.comments),
        result.stringEscapeEvents);
  }

  /**
   * Extract a subtree containing only statements from {@code firstStatement} (included) up to
   * {@code lastStatement} excluded.
   */
  public StarlarkFile subTree(int firstStatement, int lastStatement) {
    ImmutableList<Statement> statements = this.statements.subList(firstStatement, lastStatement);
    return new StarlarkFile(
        statements,
        containsErrors,
        null,
        this.statements.get(firstStatement).getLocation(),
        ImmutableList.of(),
        stringEscapeEvents);
  }

  /**
   * Returns true if any errors were encountered during scanning or parsing. If
   * set, clients should not rely on the correctness of the AST for builds or
   * BUILD-file editing.
   */
  public boolean containsErrors() {
    return containsErrors;
  }

  /**
   * Returns an (immutable, ordered) list of statements in this BUILD file.
   */
  public ImmutableList<Statement> getStatements() {
    return statements;
  }

  /**
   * Returns an (immutable, ordered) list of comments in this BUILD file.
   */
  public ImmutableList<Comment> getComments() {
    return comments;
  }

  /** Returns true if there was no error event. */
  public boolean replayLexerEvents(StarlarkThread thread, EventHandler eventHandler) {
    if (thread.getSemantics().incompatibleRestrictStringEscapes()
        && !stringEscapeEvents.isEmpty()) {
      Event.replayEventsOn(eventHandler, stringEscapeEvents);
      return false;
    }
    return true;
  }

  /**
   * Executes this build file in a given StarlarkThread.
   *
   * <p>If, for any reason, execution of a statement cannot be completed, exec throws an {@link
   * EvalException}. This exception is caught here and reported through reporter and execution
   * continues on the next statement. In effect, there is a "try/except" block around every top
   * level statement. Such exceptions are not ignored, though: they are visible via the return
   * value. Rules declared in a package containing any error (including loading-phase semantical
   * errors that cannot be checked here) must also be considered "in error".
   *
   * <p>Note that this method will not affect the value of {@link #containsErrors()}; that refers
   * only to lexer/parser errors.
   *
   * @return true if no error occurred during execution.
   */
  // TODO(adonovan): move to EvalUtils.
  public boolean exec(StarlarkThread thread, EventHandler eventHandler)
      throws InterruptedException {
    boolean ok = true;
    for (Statement stmt : statements) {
      if (!execTopLevelStatement(stmt, thread, eventHandler)) {
        ok = false;
      }
    }
    return ok;
  }

  /**
   * Executes top-level statement of this build file in a given StarlarkThread.
   *
   * <p>If, for any reason, execution of a statement cannot be completed, exec throws an {@link
   * EvalException}. This exception is caught here and reported through reporter. In effect, there
   * is a "try/except" block around every top level statement. Such exceptions are not ignored,
   * though: they are visible via the return value. Rules declared in a package containing any error
   * (including loading-phase semantical errors that cannot be checked here) must also be considered
   * "in error".
   *
   * <p>Note that this method will not affect the value of {@link #containsErrors()}; that refers
   * only to lexer/parser errors.
   *
   * @return true if no error occurred during execution.
   */
  public boolean execTopLevelStatement(
      Statement stmt, StarlarkThread thread, EventHandler eventHandler)
      throws InterruptedException {
    try {
      Eval.execToplevelStatement(thread, stmt);
      return true;
    } catch (EvalException e) {
      // Do not report errors caused by a previous parsing error, as it has already been
      // reported.
      if (e.isDueToIncompleteAST()) {
        return false;
      }
      // When the exception is raised from another file, report first the location in the
      // BUILD file (as it is the most probable cause for the error).
      Location exnLoc = e.getLocation();
      Location nodeLoc = stmt.getLocation();
      eventHandler.handle(Event.error(
          (exnLoc == null || !nodeLoc.getPath().equals(exnLoc.getPath())) ? nodeLoc : exnLoc,
          e.getMessage()));
      return false;
    }
  }

  @Override
  public void prettyPrint(Appendable buffer, int indentLevel) throws IOException {
    // Only statements are printed, not comments.
    for (Statement stmt : statements) {
      stmt.prettyPrint(buffer, indentLevel);
    }
  }

  @Override
  public String toString() {
    return "<StarlarkFile with " + statements.size() + " statements>";
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }

  /**
   * Parse the specified file, returning its syntax tree with the preludeStatements inserted at the
   * front of its statement list. All errors during scanning or parsing will be reported to the
   * event handler.
   */
  public static StarlarkFile parseWithPrelude(
      ParserInput input, List<Statement> preludeStatements, EventHandler eventHandler) {
    Parser.ParseResult result = Parser.parseFile(input, eventHandler);
    return create(
        preludeStatements, result, /* contentHashCode= */ null, /*allowImportInternal=*/ false);
  }

  /**
   * Parse the specified build file, returning its AST. All load statements parsed that way will be
   * exempt from visibility restrictions. All errors during scanning or parsing will be reported to
   * the event handler.
   */
  public static StarlarkFile parseVirtualBuildFile(
      ParserInput input, List<Statement> preludeStatements, EventHandler eventHandler) {
    Parser.ParseResult result = Parser.parseFile(input, eventHandler);
    return create(
        preludeStatements, result, /* contentHashCode= */ null, /*allowImportInternal=*/ true);
  }

  public static StarlarkFile parseWithDigest(
      ParserInput input, byte[] digest, EventHandler eventHandler) throws IOException {
    Parser.ParseResult result = Parser.parseFile(input, eventHandler);
    return create(
        /* preludeStatements= */ ImmutableList.of(),
        result,
        HashCode.fromBytes(digest).toString(),
        /* allowImportInternal= */ false);
  }

  public static StarlarkFile parse(ParserInput input, EventHandler eventHandler) {
    Parser.ParseResult result = Parser.parseFile(input, eventHandler);
    return create(
        /* preludeStatements= */ ImmutableList.<Statement>of(),
        result,
        /* contentHashCode= */ null,
        /* allowImportInternal=*/ false);
  }

  /**
   * Parse the specified file but avoid the validation of the imports, returning its AST. All errors
   * during scanning or parsing will be reported to the event handler.
   */
  // TODO(adonovan): redundant; delete.
  public static StarlarkFile parseWithoutImports(ParserInput input, EventHandler eventHandler) {
    ParseResult result = Parser.parseFile(input, eventHandler);
    return new StarlarkFile(
        ImmutableList.copyOf(result.statements),
        result.containsErrors,
        /* contentHashCode= */ null,
        result.location,
        ImmutableList.copyOf(result.comments),
        result.stringEscapeEvents);
  }

  /**
   * Run static checks on the syntax tree.
   *
   * @return a new syntax tree (or the same), with the containsErrors flag updated.
   */
  // TODO(adonovan): eliminate. Most callers need validation because they intend to execute the
  // file, and should be made to use higher-level operations in EvalUtils.
  // rest should skip this step. Called from EvaluationTestCase, ParserTest, ASTFileLookupFunction.
  public StarlarkFile validate(
      StarlarkThread thread, boolean isBuildFile, EventHandler eventHandler) {
    try {
      ValidationEnvironment.validateFile(this, thread, isBuildFile);
      return this;
    } catch (EvalException e) {
      if (!e.isDueToIncompleteAST()) {
        eventHandler.handle(Event.error(e.getLocation(), e.getMessage()));
      }
    }
    if (containsErrors) {
      return this; // already marked as errant
    }
    return new StarlarkFile(
        statements,
        /*containsErrors=*/ true,
        contentHashCode,
        getLocation(),
        comments,
        stringEscapeEvents);
  }

  /**
   * Evaluates the code and return the value of the last statement if it's an Expression or else
   * null.
   */
  // TODO(adonovan): move to EvalUtils. Split into two APIs, eval(expr) and exec(file).
  // (Abolish "statement" and "file+expr" as primary API concepts.)
  // Make callers decide whether they want to execute a file or evaluate an expression.
  @Nullable
  public Object eval(StarlarkThread thread) throws EvalException, InterruptedException {
    List<Statement> stmts = statements;
    Expression expr = null;
    int n = statements.size();
    if (n > 0 && statements.get(n - 1) instanceof ExpressionStatement) {
      stmts = statements.subList(0, n - 1);
      expr = ((ExpressionStatement) statements.get(n - 1)).getExpression();
    }
    Eval.execStatements(thread, stmts);
    return expr == null ? null : Eval.eval(thread, expr);
  }

  /**
   * Parses, resolves and evaluates the input and returns the value of the last statement if it's an
   * Expression or else null. In case of error (either during validation or evaluation), it throws
   * an EvalException. The return value is as for eval(StarlarkThread).
   */
  // Note: uses Starlark (not BUILD) validation semantics.
  // TODO(adonovan): move to EvalUtils; see other eval function.
  @Nullable
  public static Object eval(ParserInput input, StarlarkThread thread)
      throws EvalException, InterruptedException {
    StarlarkFile ast = parseAndValidateSkylark(input, thread);
    return ast.eval(thread);
  }

  /**
   * Parses and validates the input and returns the syntax tree. In case of error during validation,
   * it throws an EvalException. Uses Starlark (not BUILD) validation semantics.
   */
  // TODO(adonovan): move to EvalUtils; see above.
  public static StarlarkFile parseAndValidateSkylark(ParserInput input, StarlarkThread thread)
      throws EvalException {
    StarlarkFile file = parse(input, thread.getEventHandler());
    file.replayLexerEvents(thread, thread.getEventHandler());
    ValidationEnvironment.validateFile(file, thread, /*isBuildFile=*/ false);
    return file;
  }

  /**
   * Returns a hash code calculated from the string content of the source file of this AST.
   */
  @Nullable public String getContentHashCode() {
    return contentHashCode;
  }
}
