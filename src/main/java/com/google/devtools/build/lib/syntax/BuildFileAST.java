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
import com.google.common.collect.ImmutableMap;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.Parser.ParseResult;
import com.google.devtools.build.lib.syntax.SkylarkImport.SkylarkImportSyntaxException;
import com.google.devtools.build.lib.util.Pair;
import java.io.IOException;
import java.util.List;
import javax.annotation.Nullable;

/** Abstract syntax node for an entire BUILD file. */
// TODO(bazel-team): Consider breaking this up into two classes: One that extends Node and does
// not include import info; and one that wraps that object with additional import info but that
// does not itself extend Node. This would help keep the AST minimalistic.
public class BuildFileAST extends Node {

  private final ImmutableList<Statement> statements;

  private final ImmutableList<Comment> comments;

  @Nullable private final ImmutableList<SkylarkImport> imports;

  /**
   * Whether any errors were encountered during scanning or parsing.
   */
  private final boolean containsErrors;

  private final List<Event> stringEscapeEvents;

  @Nullable private final String contentHashCode;

  private BuildFileAST(
      ImmutableList<Statement> statements,
      boolean containsErrors,
      String contentHashCode,
      Location location,
      ImmutableList<Comment> comments,
      @Nullable ImmutableList<SkylarkImport> imports,
      List<Event> stringEscapeEvents) {
    this.statements = statements;
    this.containsErrors = containsErrors;
    this.contentHashCode = contentHashCode;
    this.comments = comments;
    this.setLocation(location);
    this.imports = imports;
    this.stringEscapeEvents = stringEscapeEvents;
  }

  private static BuildFileAST create(
      List<Statement> preludeStatements,
      ParseResult result,
      String contentHashCode,
      ImmutableMap<RepositoryName, RepositoryName> repositoryMapping,
      EventHandler eventHandler,
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
    boolean containsErrors = result.containsErrors;
    Pair<Boolean, ImmutableList<SkylarkImport>> skylarkImports =
        fetchLoads(statements, repositoryMapping, eventHandler);
    containsErrors |= skylarkImports.first;
    return new BuildFileAST(
        statements,
        containsErrors,
        contentHashCode,
        result.location,
        ImmutableList.copyOf(result.comments),
        skylarkImports.second,
        result.stringEscapeEvents);
  }

  private static BuildFileAST create(
      List<Statement> preludeStatements,
      ParseResult result,
      String contentHashCode,
      ImmutableMap<RepositoryName, RepositoryName> repositoryMapping,
      EventHandler eventHandler) {
    return create(
        preludeStatements, result, contentHashCode, repositoryMapping, eventHandler, false);
  }

  /**
   * Extract a subtree containing only statements from {@code firstStatement} (included) up to
   * {@code lastStatement} excluded.
   */
  public BuildFileAST subTree(int firstStatement, int lastStatement) {
    ImmutableList<Statement> statements = this.statements.subList(firstStatement, lastStatement);
    ImmutableList.Builder<SkylarkImport> imports = ImmutableList.builder();
    for (Statement stmt : statements) {
      if (stmt instanceof LoadStatement) {
        String str = ((LoadStatement) stmt).getImport().getValue();
        try {
          imports.add(SkylarkImport.create(str, /* repositoryMapping= */ ImmutableMap.of()));
        } catch (SkylarkImportSyntaxException e) {
          throw new IllegalStateException(
              "Cannot create SkylarkImport for '" + str + "'. This is an internal error.", e);
        }
      }
    }
    return new BuildFileAST(
        statements,
        containsErrors,
        null,
        this.statements.get(firstStatement).getLocation(),
        ImmutableList.of(),
        imports.build(),
        stringEscapeEvents);
  }

  /**
   * Collects all load statements. Returns a pair with a boolean saying if there were errors and the
   * imports that could be resolved.
   */
  private static Pair<Boolean, ImmutableList<SkylarkImport>> fetchLoads(
      List<Statement> statements,
      ImmutableMap<RepositoryName, RepositoryName> repositoryMapping,
      EventHandler eventHandler) {
    ImmutableList.Builder<SkylarkImport> imports = ImmutableList.builder();
    boolean error = false;
    for (Statement stmt : statements) {
      if (stmt instanceof LoadStatement) {
        String importString = ((LoadStatement) stmt).getImport().getValue();
        try {
          imports.add(SkylarkImport.create(importString, repositoryMapping));
        } catch (SkylarkImportSyntaxException e) {
          eventHandler.handle(Event.error(stmt.getLocation(), e.getMessage()));
          error = true;
        }
      }
    }
    return Pair.of(error, imports.build());
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

  /** Returns a list of loads in this BUILD file. */
  public ImmutableList<SkylarkImport> getImports() {
    Preconditions.checkNotNull(imports, "computeImports Should be called in parse* methods");
    return imports;
  }

  /** Returns a list of loads as strings in this BUILD file. */
  public ImmutableList<StringLiteral> getRawImports() {
    ImmutableList.Builder<StringLiteral> imports = ImmutableList.builder();
    for (Statement stmt : statements) {
      if (stmt instanceof LoadStatement) {
        imports.add(((LoadStatement) stmt).getImport());
      }
    }
    return imports.build();
  }

  /** Returns true if there was no error event. */
  public boolean replayLexerEvents(Environment env, EventHandler eventHandler) {
    if (env.getSemantics().incompatibleRestrictStringEscapes() && !stringEscapeEvents.isEmpty()) {
      Event.replayEventsOn(eventHandler, stringEscapeEvents);
      return false;
    }
    return true;
  }

  /**
   * Executes this build file in a given Environment.
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
  public boolean exec(Environment env, EventHandler eventHandler) throws InterruptedException {
    boolean ok = true;
    for (Statement stmt : statements) {
      if (!execTopLevelStatement(stmt, env, eventHandler)) {
        ok = false;
      }
    }
    return ok;
  }

  /**
   * Executes top-level statement of this build file in a given Environment.
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
  public boolean execTopLevelStatement(Statement stmt, Environment env, EventHandler eventHandler)
      throws InterruptedException {
    try {
      Eval.execToplevelStatement(env, stmt);
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
    // Only statements are printed, not comments and processed import data.
    for (Statement stmt : statements) {
      stmt.prettyPrint(buffer, indentLevel);
    }
  }

  @Override
  public String toString() {
    return "<BuildFileAST with " + statements.size() + " statements>";
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
  public static BuildFileAST parseWithPrelude(
      ParserInput input,
      List<Statement> preludeStatements,
      ImmutableMap<RepositoryName, RepositoryName> repositoryMapping,
      EventHandler eventHandler) {
    Parser.ParseResult result = Parser.parseFile(input, eventHandler);
    return create(
        preludeStatements, result, /* contentHashCode= */ null, repositoryMapping, eventHandler);
  }

  /**
   * Parse the specified build file, returning its AST. All load statements parsed that way will be
   * exempt from visibility restrictions. All errors during scanning or parsing will be reported to
   * the event handler.
   */
  public static BuildFileAST parseVirtualBuildFile(
      ParserInput input,
      List<Statement> preludeStatements,
      ImmutableMap<RepositoryName, RepositoryName> repositoryMapping,
      EventHandler eventHandler) {
    Parser.ParseResult result = Parser.parseFile(input, eventHandler);
    return create(
        preludeStatements,
        result,
        /* contentHashCode= */ null,
        repositoryMapping,
        eventHandler,
        true);
  }

  public static BuildFileAST parseWithDigest(
      ParserInput input, byte[] digest, EventHandler eventHandler) throws IOException {
    Parser.ParseResult result = Parser.parseFile(input, eventHandler);
    return create(
        /* preludeStatements= */ ImmutableList.of(),
        result,
        HashCode.fromBytes(digest).toString(),
        /* repositoryMapping= */ ImmutableMap.of(),
        eventHandler);
  }

  public static BuildFileAST parse(ParserInput input, EventHandler eventHandler) {
    Parser.ParseResult result = Parser.parseFile(input, eventHandler);
    return create(
        /* preludeStatements= */ ImmutableList.<Statement>of(),
        result,
        /* contentHashCode= */ null,
        /* repositoryMapping= */ ImmutableMap.of(),
        eventHandler);
  }

  /**
   * Parse the specified file but avoid the validation of the imports, returning its AST. All errors
   * during scanning or parsing will be reported to the event handler.
   */
  public static BuildFileAST parseWithoutImports(ParserInput input, EventHandler eventHandler) {
    ParseResult result = Parser.parseFile(input, eventHandler);
    return new BuildFileAST(
        ImmutableList.copyOf(result.statements),
        result.containsErrors,
        /* contentHashCode= */ null,
        result.location,
        ImmutableList.copyOf(result.comments),
        /* imports= */ null,
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
  public BuildFileAST validate(Environment env, boolean isBuildFile, EventHandler eventHandler) {
    try {
      ValidationEnvironment.validateFile(this, env, isBuildFile);
      return this;
    } catch (EvalException e) {
      if (!e.isDueToIncompleteAST()) {
        eventHandler.handle(Event.error(e.getLocation(), e.getMessage()));
      }
    }
    if (containsErrors) {
      return this; // already marked as errant
    }
    return new BuildFileAST(
        statements,
        /*containsErrors=*/ true,
        contentHashCode,
        getLocation(),
        comments,
        imports,
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
  public Object eval(Environment env) throws EvalException, InterruptedException {
    List<Statement> stmts = statements;
    Expression expr = null;
    int n = statements.size();
    if (n > 0 && statements.get(n - 1) instanceof ExpressionStatement) {
      stmts = statements.subList(0, n - 1);
      expr = ((ExpressionStatement) statements.get(n - 1)).getExpression();
    }
    Eval.execStatements(env, stmts);
    return expr == null ? null : Eval.eval(env, expr);
  }

  /**
   * Parses, resolves and evaluates the input and returns the value of the last statement if it's an
   * Expression or else null. In case of error (either during validation or evaluation), it throws
   * an EvalException. The return value is as for eval(Environment).
   */
  // Note: uses Starlark (not BUILD) validation semantics.
  // TODO(adonovan): move to EvalUtils; see other eval function.
  @Nullable
  public static Object eval(ParserInput input, Environment env)
      throws EvalException, InterruptedException {
    BuildFileAST ast = parseAndValidateSkylark(input, env);
    return ast.eval(env);
  }

  /**
   * Parses and validates the input and returns the syntax tree. In case of error during validation,
   * it throws an EvalException. Uses Starlark (not BUILD) validation semantics.
   */
  // TODO(adonovan): move to EvalUtils; see above.
  public static BuildFileAST parseAndValidateSkylark(ParserInput input, Environment env)
      throws EvalException {
    BuildFileAST file = parse(input, env.getEventHandler());
    file.replayLexerEvents(env, env.getEventHandler());
    ValidationEnvironment.validateFile(file, env, /*isBuildFile=*/ false);
    return file;
  }

  /**
   * Returns a hash code calculated from the string content of the source file of this AST.
   */
  @Nullable public String getContentHashCode() {
    return contentHashCode;
  }
}
