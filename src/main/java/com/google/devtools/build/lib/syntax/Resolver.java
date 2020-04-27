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
import com.google.devtools.starlark.spelling.SpellChecker;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * The Resolver resolves each identifier in a syntax tree to its binding, and performs other
 * validity checks.
 *
 * <p>When a variable is defined, it is visible in the entire block. For example, a global variable
 * is visible in the entire file; a variable in a function is visible in the entire function block
 * (even on the lines before its first assignment).
 *
 * <p>Resolution is a mutation of the syntax tree, as it attaches scope information to Identifier
 * nodes. (In the future, it will attach additional information to functions to support lexical
 * scope, and even compilation of the trees to bytecode.) Resolution errors are reported in the
 * analogous manner to scan/parse errors: for a StarlarkFile, they are appended to {@code
 * StarlarkFile.errors}; for an expression they are reported by an SyntaxError.Exception exception.
 * It is legal to resolve a file that already contains scan/parse errors, though it may lead to
 * secondary errors.
 */
public final class Resolver extends NodeVisitor {

  enum Scope {
    /** Symbols defined inside a function or a comprehension. */
    Local("local"),
    /** Symbols defined at a module top-level, e.g. functions, loaded symbols. */
    Module("global"),
    /** Predefined symbols (builtins) */
    Universe("builtin");

    private final String qualifier;

    private Scope(String qualifier) {
      this.qualifier = qualifier;
    }

    String getQualifier() {
      return qualifier;
    }
  }

  /**
   * Module is a static abstraction of a Starlark module. It describes the set of variable names for
   * use during name resolution.
   */
  public interface Module {

    // TODO(adonovan): opt: for efficiency, turn this into a predicate, not an enumerable set,
    // and look up bindings as they are needed, not preemptively.
    // Otherwise we must do work proportional to the number of bindings in the
    // environment, not the number of free variables of the file/expression.
    //
    // A single method will then suffice:
    //   Scope resolve(String name) throws Undeclared

    /** Returns the set of names defined by this module. The caller must not modify the set. */
    Set<String> getNames();

    /**
     * Returns (optionally) a more specific error for an undeclared name than the generic message.
     * This hook allows the module to implement flag-enabled names without any knowledge in this
     * file.
     */
    @Nullable
    String getUndeclaredNameError(String name);
  }

  private static final Identifier PREDECLARED; // sentinel for predeclared names

  static {
    try {
      PREDECLARED = (Identifier) Expression.parse(ParserInput.fromLines("PREDECLARED"));
    } catch (SyntaxError.Exception ex) {
      throw new IllegalStateException(ex); // can't happen
    }
  }

  private static class Block {
    private final Map<String, Identifier> variables = new HashMap<>();
    private final Scope scope;
    @Nullable private final Block parent;

    Block(Scope scope, @Nullable Block parent) {
      this.scope = scope;
      this.parent = parent;
    }
  }

  private final List<SyntaxError> errors;
  private final FileOptions options;
  private final Module module;
  private Block block;
  private int loopCount;

  private Resolver(List<SyntaxError> errors, Module module, FileOptions options) {
    this.errors = errors;
    this.module = module;
    this.options = options;
    block = new Block(Scope.Universe, null);
    for (String name : module.getNames()) {
      block.variables.put(name, PREDECLARED);
    }
  }

  // Reports an error at the start of the specified node.
  private void addError(Node node, String message) {
    addError(node.getStartLocation(), message);
  }

  // Reports an error at the specified location.
  private void addError(Location loc, String message) {
    errors.add(new SyntaxError(loc, message));
  }

  /**
   * First pass: add all definitions to the current block. This is done because symbols are
   * sometimes used before their definition point (e.g. a functions are not necessarily declared in
   * order).
   */
  private void collectDefinitions(Iterable<Statement> stmts) {
    for (Statement stmt : stmts) {
      collectDefinitions(stmt);
    }
  }

  private void collectDefinitions(Statement stmt) {
    switch (stmt.kind()) {
      case ASSIGNMENT:
        collectDefinitions(((AssignmentStatement) stmt).getLHS());
        break;
      case IF:
        IfStatement ifStmt = (IfStatement) stmt;
        collectDefinitions(ifStmt.getThenBlock());
        if (ifStmt.getElseBlock() != null) {
          collectDefinitions(ifStmt.getElseBlock());
        }
        break;
      case FOR:
        ForStatement forStmt = (ForStatement) stmt;
        collectDefinitions(forStmt.getVars());
        collectDefinitions(forStmt.getBody());
        break;
      case DEF:
        DefStatement def = (DefStatement) stmt;
        declare(def.getIdentifier());
        break;
      case LOAD:
        LoadStatement load = (LoadStatement) stmt;
        Set<String> names = new HashSet<>();
        for (LoadStatement.Binding b : load.getBindings()) {
          // Reject load('...', '_private').
          Identifier orig = b.getOriginalName();
          if (orig.isPrivate() && !options.allowLoadPrivateSymbols()) {
            addError(orig, "symbol '" + orig.getName() + "' is private and cannot be imported.");
          }

          // The allowToplevelRebinding check is not applied to all files
          // but we apply it to each load statement as a special case,
          // and emit a better error message than the generic check.
          if (!names.add(b.getLocalName().getName())) {
            addError(
                b.getLocalName(),
                String.format(
                    "load statement defines '%s' more than once", b.getLocalName().getName()));
          }
        }

        // TODO(adonovan): support options.loadBindsGlobally().
        // Requires that we open a Local block for each file,
        // as well as its Module block, and select which block
        // to declare it in. See go.starlark.net implementation.

        for (LoadStatement.Binding b : load.getBindings()) {
          declare(b.getLocalName());
        }
        break;
      case EXPRESSION:
      case FLOW:
      case RETURN:
        // nothing to declare
    }
  }

  private void collectDefinitions(Expression lhs) {
    for (Identifier id : Identifier.boundIdentifiers(lhs)) {
      declare(id);
    }
  }

  private void assign(Expression lhs) {
    if (lhs instanceof Identifier) {
      if (options.recordScope()) {
        ((Identifier) lhs).setScope(block.scope);
      }
      // no-op
    } else if (lhs instanceof IndexExpression) {
      visit(lhs);
    } else if (lhs instanceof ListExpression) {
      for (Expression elem : ((ListExpression) lhs).getElements()) {
        assign(elem);
      }
    } else {
      addError(lhs, "cannot assign to '" + lhs + "'");
    }
  }

  @Override
  public void visit(Identifier node) {
    String name = node.getName();
    @Nullable Block b = blockThatDefines(name);
    if (b == null) {
      // The identifier might not exist because it was restricted (hidden) by flags.
      // If this is the case, output a more helpful error message than 'not found'.
      String error = module.getUndeclaredNameError(name);
      if (error == null) {
        // generic error
        error = createInvalidIdentifierException(node.getName(), getAllSymbols());
      }
      addError(node, error);
      return;
    }
    if (options.recordScope()) {
      node.setScope(b.scope);
    }
  }

  private static String createInvalidIdentifierException(String name, Set<String> candidates) {
    if (!Identifier.isValid(name)) {
      // Identifier was created by Parser.makeErrorExpression and contains misparsed text.
      return "contains syntax error(s)";
    }

    String error = getErrorForObsoleteThreadLocalVars(name);
    if (error != null) {
      return error;
    }

    String suggestion = SpellChecker.didYouMean(name, candidates);
    return "name '" + name + "' is not defined" + suggestion;
  }

  // TODO(adonovan): delete this. It's been long enough.
  static String getErrorForObsoleteThreadLocalVars(String name) {
    if (name.equals("PACKAGE_NAME")) {
      return "The value 'PACKAGE_NAME' has been removed in favor of 'package_name()', "
          + "please use the latter ("
          + "https://docs.bazel.build/versions/master/skylark/lib/native.html#package_name). ";
    }
    if (name.equals("REPOSITORY_NAME")) {
      return "The value 'REPOSITORY_NAME' has been removed in favor of 'repository_name()', please"
          + " use the latter ("
          + "https://docs.bazel.build/versions/master/skylark/lib/native.html#repository_name).";
    }
    return null;
  }

  @Override
  public void visit(ReturnStatement node) {
    if (block.scope != Scope.Local) {
      addError(node, "return statements must be inside a function");
    }
    super.visit(node);
  }

  @Override
  public void visit(ForStatement node) {
    if (block.scope != Scope.Local) {
      addError(
          node,
          "for loops are not allowed at the top level. You may move it inside a function "
              + "or use a comprehension, [f(x) for x in sequence]");
    }
    loopCount++;
    visit(node.getCollection());
    assign(node.getVars());
    visitBlock(node.getBody());
    Preconditions.checkState(loopCount > 0);
    loopCount--;
  }

  @Override
  public void visit(LoadStatement node) {
    if (block.scope == Scope.Local) {
      addError(node, "load statement not at top level");
    }
    super.visit(node);
  }

  @Override
  public void visit(FlowStatement node) {
    if (node.getKind() != TokenKind.PASS && loopCount <= 0) {
      addError(node, node.getKind() + " statement must be inside a for loop");
    }
    super.visit(node);
  }

  @Override
  public void visit(DotExpression node) {
    visit(node.getObject());
    // Do not visit the field.
  }

  @Override
  public void visit(Comprehension node) {
    openBlock(Scope.Local);
    for (Comprehension.Clause clause : node.getClauses()) {
      if (clause instanceof Comprehension.For) {
        Comprehension.For forClause = (Comprehension.For) clause;
        collectDefinitions(forClause.getVars());
      }
    }
    // TODO(adonovan): opt: combine loops
    for (Comprehension.Clause clause : node.getClauses()) {
      if (clause instanceof Comprehension.For) {
        Comprehension.For forClause = (Comprehension.For) clause;
        visit(forClause.getIterable());
        assign(forClause.getVars());
      } else {
        Comprehension.If ifClause = (Comprehension.If) clause;
        visit(ifClause.getCondition());
      }
    }
    visit(node.getBody());
    closeBlock();
  }

  @Override
  public void visit(DefStatement node) {
    if (block.scope == Scope.Local) {
      addError(node, "nested functions are not allowed. Move the function to the top level.");
    }
    for (Parameter param : node.getParameters()) {
      if (param instanceof Parameter.Optional) {
        visit(param.getDefaultValue());
      }
    }
    openBlock(Scope.Local);
    for (Parameter param : node.getParameters()) {
      if (param.getIdentifier() != null) {
        declare(param.getIdentifier());
      }
    }
    collectDefinitions(node.getBody());
    visitAll(node.getBody());
    closeBlock();
  }

  @Override
  public void visit(IfStatement node) {
    if (block.scope != Scope.Local) {
      addError(
          node,
          "if statements are not allowed at the top level. You may move it inside a function "
              + "or use an if expression (x if condition else y).");
    }
    super.visit(node);
  }

  @Override
  public void visit(AssignmentStatement node) {
    visit(node.getRHS());

    // Disallow: [e, ...] += rhs
    // Other bad cases are handled in assign.
    if (node.isAugmented() && node.getLHS() instanceof ListExpression) {
      addError(
          node.getOperatorLocation(),
          "cannot perform augmented assignment on a list or tuple expression");
    }

    assign(node.getLHS());
  }

  /** Declare a variable and add it to the environment. */
  private void declare(Identifier id) {
    Identifier prev = block.variables.putIfAbsent(id.getName(), id);

    // Symbols defined in the module scope cannot be reassigned.
    if (prev != null && block.scope == Scope.Module && !options.allowToplevelRebinding()) {
      addError(
          id,
          String.format(
              "cannot reassign global '%s' (read more at"
                  + " https://bazel.build/versions/master/docs/skylark/errors/read-only-variable.html)",
              id.getName()));
      if (prev != PREDECLARED) {
        addError(prev, String.format("'%s' previously declared here", id.getName()));
      }
    }
  }

  /** Returns the nearest Block that defines a symbol. */
  private Block blockThatDefines(String varname) {
    for (Block b = block; b != null; b = b.parent) {
      if (b.variables.containsKey(varname)) {
        return b;
      }
    }
    return null;
  }

  /** Returns the set of all accessible symbols (both local and global) */
  private Set<String> getAllSymbols() {
    Set<String> all = new HashSet<>();
    for (Block b = block; b != null; b = b.parent) {
      all.addAll(b.variables.keySet());
    }
    return all;
  }

  // Report an error if a load statement appears after another kind of statement.
  private void checkLoadAfterStatement(List<Statement> statements) {
    Statement firstStatement = null;

    for (Statement statement : statements) {
      // Ignore string literals (e.g. docstrings).
      if (statement instanceof ExpressionStatement
          && ((ExpressionStatement) statement).getExpression() instanceof StringLiteral) {
        continue;
      }

      if (statement instanceof LoadStatement) {
        if (firstStatement == null) {
          continue;
        }
        addError(statement, "load statements must appear before any other statement");
        addError(firstStatement, "\tfirst non-load statement appears here");
      }

      if (firstStatement == null) {
        firstStatement = statement;
      }
    }
  }

  private void resolveToplevelStatements(List<Statement> statements) {
    // Check that load() statements are on top.
    if (options.requireLoadStatementsFirst()) {
      checkLoadAfterStatement(statements);
    }

    openBlock(Scope.Module);

    // Add each variable defined by statements, not including definitions that appear in
    // sub-scopes of the given statements (function bodies and comprehensions).
    collectDefinitions(statements);

    // Second pass: ensure that all symbols have been defined.
    visitAll(statements);
    closeBlock();
  }

  /**
   * Performs static checks, including resolution of identifiers in {@code file} in the environment
   * defined by {@code module}. The StarlarkFile is mutated. Errors are appended to {@link
   * StarlarkFile#errors}.
   */
  public static void resolveFile(StarlarkFile file, Module module) {
    Resolver r = new Resolver(file.errors, module, file.getOptions());
    r.resolveToplevelStatements(file.getStatements());
    // Check that no closeBlock was forgotten.
    Preconditions.checkState(r.block.parent == null);
  }

  /**
   * Performs static checks, including resolution of identifiers in {@code expr} in the environment
   * defined by {@code module}. This operation mutates the Expression.
   */
  public static void resolveExpr(Expression expr, Module module, FileOptions options)
      throws SyntaxError.Exception {
    List<SyntaxError> errors = new ArrayList<>();
    Resolver r = new Resolver(errors, module, options);

    r.visit(expr);

    if (!errors.isEmpty()) {
      throw new SyntaxError.Exception(errors);
    }
  }

  /** Open a new lexical block that will contain the future declarations. */
  private void openBlock(Scope scope) {
    block = new Block(scope, block);
  }

  /** Close a lexical block (and lose all declarations it contained). */
  private void closeBlock() {
    block = Preconditions.checkNotNull(block.parent);
  }

}
