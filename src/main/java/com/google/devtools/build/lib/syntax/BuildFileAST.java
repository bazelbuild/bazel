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
import com.google.devtools.build.lib.vfs.Path;

import java.io.IOException;
import java.util.List;

import javax.annotation.Nullable;

/**
 * Abstract syntax node for an entire BUILD file.
 */
public class BuildFileAST extends ASTNode {

  private final ImmutableList<Statement> stmts;

  private final ImmutableList<Comment> comments;

  private ImmutableList<SkylarkImport> imports;

  /**
   * Whether any errors were encountered during scanning or parsing.
   */
  private final boolean containsErrors;

  private final String contentHashCode;

  private BuildFileAST(List<Statement> preludeStatements, Parser.ParseResult result) {
    this(preludeStatements, result, null);
  }

  private BuildFileAST(List<Statement> preludeStatements,
      Parser.ParseResult result, String contentHashCode) {
    this.stmts = ImmutableList.<Statement>builder()
        .addAll(preludeStatements)
        .addAll(result.statements)
        .build();
    this.comments = ImmutableList.copyOf(result.comments);
    this.containsErrors = result.containsErrors;
    this.contentHashCode = contentHashCode;
    setLocation(result.location);
  }

  private BuildFileAST(
      ImmutableList<Statement> stmts,
      boolean containsErrors,
      String contentHashCode,
      Location location) {
    this.stmts = stmts;
    this.containsErrors = containsErrors;
    this.contentHashCode = contentHashCode;
    this.comments = ImmutableList.of();
    this.setLocation(location);
  }

  /**
   * Extract a subtree containing only statements from {@code firstStatement} (included) up to
   * {@code lastStatement} excluded.
   */
  public BuildFileAST subTree(int firstStatement, int lastStatement) {
    return new BuildFileAST(
        stmts.subList(firstStatement, lastStatement),
        containsErrors,
        null,
        stmts.get(firstStatement).getLocation());
  }

  /** Collects all load statements */
  private ImmutableList<SkylarkImport> fetchLoads(List<Statement> stmts) {
    ImmutableList.Builder<SkylarkImport> imports = new ImmutableList.Builder<>();
    for (Statement stmt : stmts) {
      if (stmt instanceof LoadStatement) {
        SkylarkImport imp = ((LoadStatement) stmt).getImport();
        imports.add(imp);
      }
    }
    return imports.build();
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
    return stmts;
  }

  /**
   * Returns an (immutable, ordered) list of comments in this BUILD file.
   */
  public ImmutableList<Comment> getComments() {
    return comments;
  }

  /**
   * Returns a list of loads in this BUILD file.
   */
  public synchronized ImmutableList<SkylarkImport> getImports() {
    if (imports == null) {
      imports = fetchLoads(stmts);
    }
    return imports;
  }

  /**
   * Executes this build file in a given Environment.
   *
   * <p>If, for any reason, execution of a statement cannot be completed, an
   * {@link EvalException} is thrown by {@link Statement#exec(Environment)}.
   * This exception is caught here and reported through reporter and execution
   * continues on the next statement.  In effect, there is a "try/except" block
   * around every top level statement.  Such exceptions are not ignored, though:
   * they are visible via the return value.  Rules declared in a package
   * containing any error (including loading-phase semantical errors that
   * cannot be checked here) must also be considered "in error".
   *
   * <p>Note that this method will not affect the value of {@link
   * #containsErrors()}; that refers only to lexer/parser errors.
   *
   * @return true if no error occurred during execution.
   */
  public boolean exec(Environment env, EventHandler eventHandler) throws InterruptedException {
    boolean ok = true;
    for (Statement stmt : stmts) {
      try {
        stmt.exec(env);
      } catch (EvalException e) {
        ok = false;
        // Do not report errors caused by a previous parsing error, as it has already been
        // reported.
        if (e.isDueToIncompleteAST()) {
          continue;
        }
        // When the exception is raised from another file, report first the location in the
        // BUILD file (as it is the most probable cause for the error).
        Location exnLoc = e.getLocation();
        Location nodeLoc = stmt.getLocation();
        eventHandler.handle(Event.error(
            (exnLoc == null || !nodeLoc.getPath().equals(exnLoc.getPath())) ? nodeLoc : exnLoc,
            e.getMessage()));
      }
    }
    return ok;
  }

  @Override
  public String toString() {
    return "BuildFileAST" + getStatements();
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  /**
   * Parse the specified build file, returning its AST. All errors during
   * scanning or parsing will be reported to the reporter.
   *
   * @throws IOException if the file cannot not be read.
   */
  public static BuildFileAST parseBuildFile(Path buildFile, EventHandler eventHandler,
                                            boolean parsePython)
      throws IOException {
    return parseBuildFile(buildFile, buildFile.getFileSize(), eventHandler, parsePython);
  }

  public static BuildFileAST parseBuildFile(Path buildFile, long fileSize,
                                            EventHandler eventHandler,
                                            boolean parsePython)
      throws IOException {
    ParserInputSource inputSource = ParserInputSource.create(buildFile, fileSize);
    return parseBuildFile(inputSource, eventHandler, parsePython);
  }

  /**
   * Parse the specified build file, returning its AST. All errors during
   * scanning or parsing will be reported to the reporter.
   */
  public static BuildFileAST parseBuildFile(ParserInputSource input,
                                            List<Statement> preludeStatements,
                                            EventHandler eventHandler,
                                            boolean parsePython) {
    Parser.ParseResult result = Parser.parseFile(input, eventHandler, parsePython);
    return new BuildFileAST(preludeStatements, result);
  }

  public static BuildFileAST parseBuildFile(ParserInputSource input, EventHandler eventHandler,
      boolean parsePython) {
    Parser.ParseResult result = Parser.parseFile(input, eventHandler, parsePython);
    return new BuildFileAST(ImmutableList.<Statement>of(), result);
  }

  /**
   * Parse the specified Skylark file, returning its AST. All errors during
   * scanning or parsing will be reported to the reporter.
   *
   * @throws IOException if the file cannot not be read.
   */
  public static BuildFileAST parseSkylarkFile(Path file, EventHandler eventHandler,
      ValidationEnvironment validationEnvironment) throws IOException {
    return parseSkylarkFile(file, file.getFileSize(), eventHandler,
        validationEnvironment);
  }

  public static BuildFileAST parseSkylarkFile(Path file, long fileSize, EventHandler eventHandler,
      ValidationEnvironment validationEnvironment) throws IOException {
    ParserInputSource input = ParserInputSource.create(file, fileSize);
    Parser.ParseResult result =
        Parser.parseFileForSkylark(input, eventHandler, validationEnvironment);
    return new BuildFileAST(ImmutableList.<Statement>of(), result,
        HashCode.fromBytes(file.getMD5Digest()).toString());
  }

  /**
   * Parse the specified build file, without building the AST.
   *
   * @return true if the input file is syntactically valid
   */
  public static boolean checkSyntax(ParserInputSource input,
                                    EventHandler eventHandler, boolean parsePython) {
    return !parseBuildFile(input, eventHandler, parsePython).containsErrors();
  }

  /**
   * Returns a hash code calculated from the string content of the source file of this AST.
   */
  @Nullable public String getContentHashCode() {
    return contentHashCode;
  }
}
