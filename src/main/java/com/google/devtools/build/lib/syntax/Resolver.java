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
import com.google.devtools.starlark.spelling.SpellChecker;
import com.google.errorprone.annotations.FormatMethod;
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
 * <p>Resolution is a mutation of the syntax tree, as it attaches binding information to Identifier
 * nodes. (In the future, it will attach additional information to functions to support lexical
 * scope, and even compilation of the trees to bytecode.) Resolution errors are reported in the
 * analogous manner to scan/parse errors: for a StarlarkFile, they are appended to {@code
 * StarlarkFile.errors}; for an expression they are reported by an SyntaxError.Exception exception.
 * It is legal to resolve a file that already contains scan/parse errors, though it may lead to
 * secondary errors.
 */
public final class Resolver extends NodeVisitor {

  // TODO(adonovan): use "keyword" (not "named") and "required" (not "mandatory") terminology
  // everywhere, including the spec.

  enum Scope {
    // TODO(adonovan): Add UNIVERSAL, FREE, CELL.
    // (PREDECLARED vs UNIVERSAL allows us to represent the app-dependent and fixed parts of the
    // predeclared environment separately, reducing the amount of copying.)

    /** Binding is local to a function, comprehension, or file (e.g. load). */
    LOCAL,
    /** Binding occurs outside any function or comprehension. */
    GLOBAL,
    /** Binding is predeclared by the core or application. */
    PREDECLARED;

    @Override
    public String toString() {
      return super.toString().toLowerCase();
    }
  }

  // A Binding is a static abstraction of a variable.
  // The Resolver maps each Identifier to a Binding.
  static final class Binding {

    final Scope scope;
    @Nullable final Identifier first; // first binding use, if syntactic
    final int index; // within its block (currently unused)

    private Binding(Scope scope, @Nullable Identifier first, int index) {
      this.scope = scope;
      this.first = first;
      this.index = index;
    }

    @Override
    public String toString() {
      return first == null
          ? scope.toString()
          : String.format(
              "%s[%d] %s @ %s", scope, index, first.getName(), first.getStartLocation());
    }
  }

  /** A Function records information about a resolved function. */
  static final class Function {

    // This class is exposed to Eval in the evaluator build target
    // (which is the same Java package, at least for now).
    // Once we switch to bytecode, it will be exposed only to the compiler.

    // The params and parameterNames fields use "run-time order":
    // non-kwonly, keyword-only, *args, **kwargs.
    // A bare * parameter is dropped.

    final String name;
    final Location location; // of identifier
    final ImmutableList<Parameter> params; // order defined above
    final ImmutableList<Statement> body;
    final boolean hasVarargs;
    final boolean hasKwargs;
    final int numKeywordOnlyParams;
    final ImmutableList<String> parameterNames; // order defined above

    // isToplevel indicates that this is the <toplevel> function containing
    // top-level statements of a file. It causes assignments to unresolved
    // identifiers to update the module, not the lexical frame.
    // TODO(adonovan): remove this hack when identifier resolution is accurate.
    final boolean isToplevel;

    private Function(
        String name,
        Location loc,
        ImmutableList<Parameter> params,
        ImmutableList<Statement> body,
        boolean hasVarargs,
        boolean hasKwargs,
        int numKeywordOnlyParams) {
      this.name = name;
      this.location = loc;
      this.params = params;
      this.body = body;
      this.hasVarargs = hasVarargs;
      this.hasKwargs = hasKwargs;
      this.numKeywordOnlyParams = numKeywordOnlyParams;

      ImmutableList.Builder<String> names = ImmutableList.builderWithExpectedSize(params.size());
      for (Parameter p : params) {
        names.add(p.getName());
      }
      this.parameterNames = names.build();

      this.isToplevel = name.equals("<toplevel>");
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

  private static class Block {
    private final Map<String, Binding> bindings = new HashMap<>();
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

  // Shared binding for all predeclared names.
  private static final Binding PREDECLARED = new Binding(Scope.PREDECLARED, null, 0);

  private Resolver(List<SyntaxError> errors, Module module, FileOptions options) {
    this.errors = errors;
    this.module = module;
    this.options = options;

    this.block = new Block(Scope.PREDECLARED, null);
    for (String name : module.getNames()) {
      block.bindings.put(name, PREDECLARED);
    }
  }

  // Formats and reports an error at the start of the specified node.
  @FormatMethod
  private void errorf(Node node, String format, Object... args) {
    errorf(node.getStartLocation(), format, args);
  }

  // Formats and reports an error at the specified location.
  @FormatMethod
  private void errorf(Location loc, String format, Object... args) {
    errors.add(new SyntaxError(loc, String.format(format, args)));
  }

  /**
   * First pass: add bindings for all variables to the current block. This is done because symbols
   * are sometimes used before their definition point (e.g. functions are not necessarily declared
   * in order).
   */
  // TODO(adonovan): eliminate this first pass by using go.starlark.net one-pass approach.
  private void createBindings(Iterable<Statement> stmts) {
    for (Statement stmt : stmts) {
      createBindings(stmt);
    }
  }

  private void createBindings(Statement stmt) {
    switch (stmt.kind()) {
      case ASSIGNMENT:
        createBindings(((AssignmentStatement) stmt).getLHS());
        break;
      case IF:
        IfStatement ifStmt = (IfStatement) stmt;
        createBindings(ifStmt.getThenBlock());
        if (ifStmt.getElseBlock() != null) {
          createBindings(ifStmt.getElseBlock());
        }
        break;
      case FOR:
        ForStatement forStmt = (ForStatement) stmt;
        createBindings(forStmt.getVars());
        createBindings(forStmt.getBody());
        break;
      case DEF:
        DefStatement def = (DefStatement) stmt;
        bind(def.getIdentifier());
        break;
      case LOAD:
        LoadStatement load = (LoadStatement) stmt;
        Set<String> names = new HashSet<>();
        for (LoadStatement.Binding b : load.getBindings()) {
          // Reject load('...', '_private').
          Identifier orig = b.getOriginalName();
          if (orig.isPrivate() && !options.allowLoadPrivateSymbols()) {
            errorf(orig, "symbol '%s' is private and cannot be imported", orig.getName());
          }

          // The allowToplevelRebinding check is not applied to all files
          // but we apply it to each load statement as a special case,
          // and emit a better error message than the generic check.
          if (!names.add(b.getLocalName().getName())) {
            errorf(
                b.getLocalName(),
                "load statement defines '%s' more than once",
                b.getLocalName().getName());
          }
        }

        // TODO(adonovan): support options.loadBindsGlobally().
        // Requires that we open a LOCAL block for each file,
        // as well as its Module block, and select which block
        // to declare it in. See go.starlark.net implementation.

        for (LoadStatement.Binding b : load.getBindings()) {
          bind(b.getLocalName());
        }
        break;
      case EXPRESSION:
      case FLOW:
      case RETURN:
        // nothing to declare
    }
  }

  private void createBindings(Expression lhs) {
    for (Identifier id : Identifier.boundIdentifiers(lhs)) {
      bind(id);
    }
  }

  private void assign(Expression lhs) {
    if (lhs instanceof Identifier) {
      Identifier id = (Identifier) lhs;
      // Bindings are created by the first pass (createBindings),
      // so there's nothing to do here.
      Preconditions.checkNotNull(block.bindings.get(id.getName()));
    } else if (lhs instanceof IndexExpression) {
      visit(lhs);
    } else if (lhs instanceof ListExpression) {
      for (Expression elem : ((ListExpression) lhs).getElements()) {
        assign(elem);
      }
    } else {
      errorf(lhs, "cannot assign to '%s'", lhs);
    }
  }

  @Override
  public void visit(Identifier id) {
    for (Block b = block; b != null; b = b.parent) {
      Binding bind = b.bindings.get(id.getName());
      if (bind != null) {
        if (options.recordScope()) {
          id.setBinding(bind);
        }
        return;
      }
    }

    // The identifier might not exist because it was restricted (hidden) by flags.
    // If this is the case, output a more helpful error message than 'not found'.
    String error = module.getUndeclaredNameError(id.getName());
    if (error == null) {
      // generic error
      error = createInvalidIdentifierException(id.getName(), getAllSymbols());
    }
    errorf(id, "%s", error);
  }

  private static String createInvalidIdentifierException(String name, Set<String> candidates) {
    if (!Identifier.isValid(name)) {
      // Identifier was created by Parser.makeErrorExpression and contains misparsed text.
      return "contains syntax errors";
    }

    String suggestion = SpellChecker.didYouMean(name, candidates);
    return "name '" + name + "' is not defined" + suggestion;
  }

  @Override
  public void visit(ReturnStatement node) {
    if (block.scope != Scope.LOCAL) {
      errorf(node, "return statements must be inside a function");
    }
    super.visit(node);
  }

  @Override
  public void visit(ForStatement node) {
    if (block.scope != Scope.LOCAL) {
      errorf(
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
    if (block.scope == Scope.LOCAL) {
      errorf(node, "load statement not at top level");
    }
    // Skip super.visit: don't revisit local Identifier as a use.
  }

  @Override
  public void visit(FlowStatement node) {
    if (node.getKind() != TokenKind.PASS && loopCount <= 0) {
      errorf(node, "%s statement must be inside a for loop", node.getKind());
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
    openBlock(Scope.LOCAL);
    for (Comprehension.Clause clause : node.getClauses()) {
      if (clause instanceof Comprehension.For) {
        Comprehension.For forClause = (Comprehension.For) clause;
        createBindings(forClause.getVars());
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
    if (block.scope == Scope.LOCAL) {
      errorf(node, "nested functions are not allowed. Move the function to the top level.");
    }
    node.resolved =
        resolveFunction(
            node.getIdentifier().getName(),
            node.getIdentifier().getStartLocation(),
            node.getParameters(),
            node.getBody());
  }

  private Function resolveFunction(
      String name,
      Location loc,
      ImmutableList<Parameter> parameters,
      ImmutableList<Statement> body) {

    // Resolve defaults in enclosing environment.
    for (Parameter param : parameters) {
      if (param instanceof Parameter.Optional) {
        visit(param.getDefaultValue());
      }
    }

    // Enter function block.
    openBlock(Scope.LOCAL);

    // Check parameter order and convert to run-time order:
    // positionals, keyword-only, *args, **kwargs.
    Parameter.Star star = null;
    Parameter.StarStar starStar = null;
    boolean seenOptional = false;
    int numKeywordOnlyParams = 0;
    // TODO(adonovan): opt: when all Identifiers are resolved to bindings accumulated
    // in the function, params can be a prefix of the function's array of bindings.
    ImmutableList.Builder<Parameter> params =
        ImmutableList.builderWithExpectedSize(parameters.size());
    for (Parameter param : parameters) {
      if (param instanceof Parameter.Mandatory) {
        // e.g. id
        if (starStar != null) {
          errorf(
              param,
              "required parameter %s may not follow **%s",
              param.getName(),
              starStar.getName());
        } else if (star != null) {
          numKeywordOnlyParams++;
        } else if (seenOptional) {
          errorf(
              param,
              "required positional parameter %s may not follow an optional parameter",
              param.getName());
        }
        bindParam(params, param);

      } else if (param instanceof Parameter.Optional) {
        // e.g. id = default
        seenOptional = true;
        if (starStar != null) {
          errorf(param, "optional parameter may not follow **%s", starStar.getName());
        } else if (star != null) {
          numKeywordOnlyParams++;
        }
        bindParam(params, param);

      } else if (param instanceof Parameter.Star) {
        // * or *args
        if (starStar != null) {
          errorf(param, "* parameter may not follow **%s", starStar.getName());
        } else if (star != null) {
          errorf(param, "multiple * parameters not allowed");
        } else {
          star = (Parameter.Star) param;
        }

      } else {
        // **kwargs
        if (starStar != null) {
          errorf(param, "multiple ** parameters not allowed");
        }
        starStar = (Parameter.StarStar) param;
      }
    }

    // * or *args
    if (star != null) {
      if (star.getIdentifier() != null) {
        bindParam(params, star);
      } else if (numKeywordOnlyParams == 0) {
        errorf(star, "bare * must be followed by keyword-only parameters");
      }
    }

    // **kwargs
    if (starStar != null) {
      bindParam(params, starStar);
    }

    createBindings(body);
    visitAll(body);
    closeBlock();

    return new Function(
        name,
        loc,
        params.build(),
        body,
        star != null && star.getIdentifier() != null,
        starStar != null,
        numKeywordOnlyParams);
  }

  private void bindParam(ImmutableList.Builder<Parameter> params, Parameter param) {
    if (bind(param.getIdentifier())) {
      errorf(param, "duplicate parameter: %s", param.getName());
    }
    params.add(param);
  }

  @Override
  public void visit(IfStatement node) {
    if (block.scope != Scope.LOCAL) {
      errorf(
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
      errorf(
          node.getOperatorLocation(),
          "cannot perform augmented assignment on a list or tuple expression");
    }

    assign(node.getLHS());
  }

  /**
   * Process a binding use of a name by adding a binding to the current block if not already bound,
   * and associate the identifier with it. Reports whether the name was already bound in this block.
   */
  private boolean bind(Identifier id) {
    Binding bind = block.bindings.get(id.getName());

    // Already bound in this block?
    if (bind != null) {
      // Symbols defined in the module block cannot be reassigned.
      if (block.scope == Scope.GLOBAL && !options.allowToplevelRebinding()) {
        errorf(
            id,
            "cannot reassign global '%s' (read more at"
                + " https://bazel.build/versions/master/docs/skylark/errors/read-only-variable.html)",
            id.getName());
        if (bind.first != null) {
          errorf(bind.first, "'%s' previously declared here", id.getName());
        }
      }

      if (options.recordScope()) {
        id.setBinding(bind);
      }
      return true;
    }

    // new binding
    // TODO(adonovan): accumulate locals in the enclosing function/file block.
    bind = new Binding(block.scope, id, block.bindings.size());
    block.bindings.put(id.getName(), bind);
    if (options.recordScope()) {
      id.setBinding(bind);
    }
    return false;
  }

  /** Returns the set of all accessible symbols (both local and global) */
  private Set<String> getAllSymbols() {
    Set<String> all = new HashSet<>();
    for (Block b = block; b != null; b = b.parent) {
      all.addAll(b.bindings.keySet());
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
        errorf(statement, "load statements must appear before any other statement");
        errorf(firstStatement, "\tfirst non-load statement appears here");
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

    openBlock(Scope.GLOBAL);

    // Add a binding for each variable defined by statements, not including definitions that appear
    // in sub-scopes of the given statements (function bodies and comprehensions).
    createBindings(statements);

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
    ImmutableList<Statement> stmts = file.getStatements();

    Resolver r = new Resolver(file.errors, module, file.getOptions());
    r.resolveToplevelStatements(stmts);
    // Check that no closeBlock was forgotten.
    Preconditions.checkState(r.block.parent == null);

    // If the final statement is an expression, synthesize a return statement.
    int n = stmts.size();
    if (n > 0 && stmts.get(n - 1) instanceof ExpressionStatement) {
      Expression expr = ((ExpressionStatement) stmts.get(n - 1)).getExpression();
      stmts =
          ImmutableList.<Statement>builder()
              .addAll(stmts.subList(0, n - 1))
              .add(ReturnStatement.make(expr))
              .build();
    }

    // Annotate with resolved information about the toplevel function.
    file.resolved =
        new Function(
            "<toplevel>",
            file.getStartLocation(),
            /*params=*/ ImmutableList.of(),
            /*body=*/ stmts,
            /*hasVarargs=*/ false,
            /*hasKwargs=*/ false,
            /*numKeywordOnlyParams=*/ 0);
  }

  /**
   * Performs static checks, including resolution of identifiers in {@code expr} in the environment
   * defined by {@code module}. This operation mutates the Expression.
   */
  static Function resolveExpr(Expression expr, Module module, FileOptions options)
      throws SyntaxError.Exception {
    List<SyntaxError> errors = new ArrayList<>();
    Resolver r = new Resolver(errors, module, options);

    r.visit(expr);

    if (!errors.isEmpty()) {
      throw new SyntaxError.Exception(errors);
    }

    // Return no-arg function that computes the expression.
    return new Function(
        "<expr>",
        expr.getStartLocation(),
        /*params=*/ ImmutableList.of(),
        ImmutableList.of(ReturnStatement.make(expr)),
        /*hasVarargs=*/ false,
        /*hasKwargs=*/ false,
        /*numKeywordOnlyParams=*/ 0);
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
