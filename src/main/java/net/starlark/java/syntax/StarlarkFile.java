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
package net.starlark.java.syntax;

import com.google.common.collect.ImmutableList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Syntax tree for a Starlark file, such as a Bazel BUILD or .bzl file.
 *
 * <p>Call {@link #parse} to parse a file. Parser errors are recorded in the syntax tree (see {@link
 * #errors}), which may be incomplete.
 */
public final class StarlarkFile extends Node {

  private final ImmutableList<Statement> statements;
  private final FileOptions options;
  private final ImmutableList<Comment> comments;
  final List<SyntaxError> errors; // appended to by Resolver
  // Map from global variable name to doc comments. Added to by Resolver.
  final Map<String, DocComments> docCommentsMap = new LinkedHashMap<>();

  // set by resolver
  @Nullable private Resolver.Function resolved;

  @Override
  public int getStartOffset() {
    return 0;
  }

  @Override
  public int getEndOffset() {
    return locs.size();
  }

  private StarlarkFile(
      FileLocations locs,
      ImmutableList<Statement> statements,
      FileOptions options,
      ImmutableList<Comment> comments,
      List<SyntaxError> errors) {
    super(locs);
    this.statements = statements;
    this.options = options;
    this.comments = comments;
    this.errors = errors;
  }

  /**
   * Returns an unmodifiable view of the list of scanner, parser, and (perhaps) resolver errors
   * accumulated in this Starlark file.
   */
  public List<SyntaxError> errors() {
    return Collections.unmodifiableList(errors);
  }

  /** Returns errors().isEmpty(). */
  public boolean ok() {
    return errors.isEmpty();
  }

  /** Returns an (immutable, ordered) list of statements in this BUILD file. */
  public ImmutableList<Statement> getStatements() {
    return statements;
  }

  /** Returns an (immutable, ordered) list of comments in this BUILD file. */
  public ImmutableList<Comment> getComments() {
    return comments;
  }

  @Override
  public String toString() {
    return "<StarlarkFile with " + statements.size() + " statements>";
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }

  void setResolvedFunction(Resolver.Function resolved) {
    this.resolved = resolved;
  }

  /**
   * Returns information about the implicit function containing the top-level statements of the
   * file. Set by the resolver.
   */
  @Nullable
  public Resolver.Function getResolvedFunction() {
    return resolved;
  }

  /**
   * Parse a Starlark file.
   *
   * <p>A syntax tree is always returned, even in case of error. Errors are recorded in the tree.
   * Example usage:
   *
   * <pre>
   * StarlarkFile file = StarlarkFile.parse(input, options);
   * if (!file.ok()) {
   *    Event.replayEventsOn(handler, file.errors());
   *    ...
   * }
   * </pre>
   */
  public static StarlarkFile parse(ParserInput input, FileOptions options) {
    Parser.ParseResult result = Parser.parseFile(input, options);
    return new StarlarkFile(
        result.locs, result.statements, options, result.comments, result.errors);
  }

  /** Parse a Starlark file with default options. */
  public static StarlarkFile parse(ParserInput input) {
    return parse(input, FileOptions.DEFAULT);
  }

  /** Returns the options specified when parsing this file. */
  public FileOptions getOptions() {
    return options;
  }

  /** Returns the name of this file, as specified to the parser. */
  public String getName() {
    return locs.file();
  }

  /** A ParseProfiler records the start and end times of parse operations. */
  public interface ParseProfiler {
    long start();

    void end(long profileStartNanos, String filename);
  }

  /** Installs a global hook that will be notified of parse operations. */
  public static void setParseProfiler(@Nullable ParseProfiler p) {
    Parser.profiler = p;
  }
}
