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

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.Parser.ParseResult;
import com.google.devtools.build.lib.syntax.SkylarkImports.SkylarkImportSyntaxException;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Abstract syntax node for an entire BUILD file.
 */
// TODO(bazel-team): Consider breaking this up into two classes: One that extends ASTNode and does
// not include import info; and one that wraps that object with additional import info but that
// does not itself extend ASTNode. This would help keep the AST minimalistic.
public class BuildFileAST extends ASTNode {

  private final ImmutableList<Statement> statements;

  private final ImmutableList<Comment> comments;

  @Nullable private final ImmutableList<SkylarkImport> imports;

  /**
   * Whether any errors were encountered during scanning or parsing.
   */
  private final boolean containsErrors;

  @Nullable private final String contentHashCode;

  private BuildFileAST(
      ImmutableList<Statement> statements,
      boolean containsErrors,
      String contentHashCode,
      Location location,
      ImmutableList<Comment> comments,
      @Nullable ImmutableList<SkylarkImport> imports) {
    this.statements = statements;
    this.containsErrors = containsErrors;
    this.contentHashCode = contentHashCode;
    this.comments = comments;
    this.setLocation(location);
    this.imports = imports;
  }

  private static BuildFileAST create(
      List<Statement> preludeStatements,
      ParseResult result,
      String contentHashCode,
      ImmutableMap<RepositoryName, RepositoryName> repositoryMapping,
      EventHandler eventHandler) {
    ImmutableList<Statement> statements =
        ImmutableList.<Statement>builder()
            .addAll(preludeStatements)
            .addAll(result.statements)
            .build();

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
        skylarkImports.second);
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
          imports.add(SkylarkImports.create(str, /* repositoryMapping= */ ImmutableMap.of()));
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
        imports.build());
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
          imports.add(SkylarkImports.create(importString, repositoryMapping));
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
  /**
   * Executes this build file in a given Environment.
   *
   * <p>If, for any reason, execution of a statement cannot be completed, an {@link EvalException}
   * is thrown by {@link Eval#exec(Statement)}. This exception is caught here and reported
   * through reporter and execution continues on the next statement. In effect, there is a
   * "try/except" block around every top level statement. Such exceptions are not ignored, though:
   * they are visible via the return value. Rules declared in a package containing any error
   * (including loading-phase semantical errors that cannot be checked here) must also be considered
   * "in error".
   *
   * <p>Note that this method will not affect the value of {@link #containsErrors()}; that refers
   * only to lexer/parser errors.
   *
   * @return true if no error occurred during execution.
   */
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
   * Executes tol-level statement of this build file in a given Environment.
   *
   * <p>If, for any reason, execution of a statement cannot be completed, an {@link EvalException}
   * is thrown by {@link Eval#exec(Statement)}. This exception is caught here and reported
   * through reporter. In effect, there is a
   * "try/except" block around every top level statement. Such exceptions are not ignored, though:
   * they are visible via the return value. Rules declared in a package containing any error
   * (including loading-phase semantical errors that cannot be checked here) must also be considered
   * "in error".
   *
   * <p>Note that this method will not affect the value of {@link #containsErrors()}; that refers
   * only to lexer/parser errors.
   *
   * @return true if no error occurred during execution.
   */

  public boolean execTopLevelStatement(Statement stmt, Environment env,
      EventHandler eventHandler) throws InterruptedException {
    try {
      Eval.fromEnvironment(env).exec(stmt);
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
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  /**
   * Parse the specified build file, returning its AST. All errors during scanning or parsing will
   * be reported to the reporter.
   */
  public static BuildFileAST parseBuildFile(
      ParserInputSource input,
      List<Statement> preludeStatements,
      ImmutableMap<RepositoryName, RepositoryName> repositoryMapping,
      EventHandler eventHandler) {
    Parser.ParseResult result = Parser.parseFile(input, eventHandler);
    return create(
            preludeStatements, result, /* contentHashCode= */ null, repositoryMapping, eventHandler)
        .validateBuildFile(eventHandler);
  }

  public static BuildFileAST parseBuildFile(ParserInputSource input, EventHandler eventHandler) {
    Parser.ParseResult result = Parser.parseFile(input, eventHandler);
    return create(
            /* preludeStatements= */ ImmutableList.<Statement>of(),
            result,
            /* contentHashCode= */ null,
            /* repositoryMapping= */ ImmutableMap.of(),
            eventHandler)
        .validateBuildFile(eventHandler);
  }

  public static BuildFileAST parseSkylarkFile(
      byte[] bytes, byte[] digest, PathFragment path, EventHandler eventHandler)
      throws IOException {
    ParserInputSource input = ParserInputSource.create(bytes, path);
    Parser.ParseResult result = Parser.parseFile(input, eventHandler);
    return create(
        /* preludeStatements= */ ImmutableList.of(),
        result,
        HashCode.fromBytes(digest).toString(),
        /* repositoryMapping= */ ImmutableMap.of(),
        eventHandler);
  }

  public static BuildFileAST parseSkylarkFile(ParserInputSource input, EventHandler eventHandler) {
    Parser.ParseResult result = Parser.parseFile(input, eventHandler);
    return create(
        /* preludeStatements= */ ImmutableList.<Statement>of(),
        result,
        /* contentHashCode= */ null,
        /* repositoryMapping= */ ImmutableMap.of(),
        eventHandler);
  }

  /**
   * Parse the specified non-build Skylark file but avoid the validation of the imports, returning
   * its AST. All errors during scanning or parsing will be reported to the reporter.
   *
   * <p>This method should not be used in Bazel code, since it doesn't validate that the imports are
   * syntactically valid.
   */
  public static BuildFileAST parseSkylarkFileWithoutImports(
      ParserInputSource input, EventHandler eventHandler) {
    ParseResult result = Parser.parseFile(input, eventHandler);
    return new BuildFileAST(
        ImmutableList.<Statement>builder()
            .addAll(ImmutableList.<Statement>of())
            .addAll(result.statements)
            .build(),
        result.containsErrors,
        /* contentHashCode= */null,
        result.location,
        ImmutableList.copyOf(result.comments),
        /* imports= */null);
  }

  /**
   * Run static checks on the AST.
   *
   * @return a new AST (or the same), with the containsErrors flag updated.
   */
  public BuildFileAST validate(Environment env, EventHandler eventHandler) {
    boolean valid = ValidationEnvironment.validateAst(env, statements, eventHandler);
    if (valid || containsErrors) {
      return this;
    }
    return new BuildFileAST(statements, true, contentHashCode, getLocation(), comments, imports);
  }

  /**
   * Run static checks for a BUILD file.
   *
   * @return a new AST (or the same), with the containsErrors flag updated.
   */
  public BuildFileAST validateBuildFile(EventHandler eventHandler) {
    boolean valid = ValidationEnvironment.checkBuildSyntax(statements, eventHandler);
    if (valid || containsErrors) {
      return this;
    }
    return new BuildFileAST(statements, true, contentHashCode, getLocation(), comments, imports);
  }

  public static BuildFileAST parseString(EventHandler eventHandler, String... content) {
    String str = Joiner.on("\n").join(content);
    ParserInputSource input = ParserInputSource.create(str, PathFragment.EMPTY_FRAGMENT);
    Parser.ParseResult result = Parser.parseFile(input, eventHandler);
    return create(
        /* preludeStatements= */ ImmutableList.of(),
        result,
        /* contentHashCode= */ null,
        /* repositoryMapping= */ ImmutableMap.of(),
        eventHandler);
  }

  public static BuildFileAST parseBuildString(EventHandler eventHandler, String... content) {
    return parseString(eventHandler, content).validateBuildFile(eventHandler);
  }

  /**
   * Parse the specified build file, without building the AST.
   *
   * @return true if the input file is syntactically valid
   */
  public static boolean checkSyntax(ParserInputSource input, EventHandler eventHandler) {
    Parser.ParseResult result = Parser.parseFile(input, eventHandler);
    return !result.containsErrors;
  }

  /**
   * Evaluates the code and return the value of the last statement if it's an
   * Expression or else null.
   */
  @Nullable public Object eval(Environment env) throws EvalException, InterruptedException {
    Object last = null;
    Eval evaluator = Eval.fromEnvironment(env);
    for (Statement statement : statements) {
      if (statement instanceof ExpressionStatement) {
        last = ((ExpressionStatement) statement).getExpression().eval(env);
      } else {
        evaluator.exec(statement);
        last = null;
      }
    }
    return last;
  }

  /**
   * Evaluates the lines from input and return the value of the last statement if it's an
   * Expression or else null. In case of error (either during validation or evaluation), it
   * throws an EvalException.
   */
  @Nullable
  public static Object eval(Environment env, String... input)
      throws EvalException, InterruptedException {
    BuildFileAST ast = parseAndValidateSkylarkString(env, input);
    return ast.eval(env);
  }

  /**
   * Parses and validates the lines from input and return the the AST
   * In case of error during validation, it throws an EvalException.
   */
  public static BuildFileAST parseAndValidateSkylarkString(Environment env, String[] input)
      throws EvalException {
    BuildFileAST ast = parseString(env.getEventHandler(), input);
    ValidationEnvironment.validateAst(env, ast.getStatements());
    return ast;
  }

  /**
   * Returns a hash code calculated from the string content of the source file of this AST.
   */
  @Nullable public String getContentHashCode() {
    return contentHashCode;
  }
}
