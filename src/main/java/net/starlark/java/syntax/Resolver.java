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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.errorprone.annotations.FormatMethod;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.spelling.SpellChecker;

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

  // TODO(adonovan):
  // - use "keyword" (not "named") and "required" (not "mandatory") terminology everywhere,
  //   including the spec.
  // - move the "no if statements at top level" check to bazel's check{Build,*}Syntax
  //   (that's a spec change), or put it behind a FileOptions flag (no spec change).
  // - remove restriction on nested def:
  //   1. use FREE for scope of references to outer LOCALs, which become CELLs.
  //   2. implement closures in eval/.
  // - make loads bind locals by default.

  /** Scope discriminates the scope of a binding: global, local, etc. */
  public enum Scope {
    /** Binding is local to a function, comprehension, or file (e.g. load). */
    LOCAL,
    /** Binding occurs outside any function or comprehension. */
    GLOBAL,
    /** Binding is local to a function, comprehension, or file, but shared with nested functions. */
    CELL, // TODO(adonovan): implement nested def
    /** Binding is an implicit parameter whose value is the CELL of some enclosing function. */
    FREE, // TODO(adonovan): implement nested def
    /** Binding is predeclared by the application (e.g. glob in Bazel). */
    PREDECLARED,
    /** Binding is predeclared by the core (e.g. None). */
    UNIVERSAL;

    @Override
    public String toString() {
      return super.toString().toLowerCase();
    }
  }

  /**
   * A Binding is a static abstraction of a variable. The Resolver maps each Identifier to a
   * Binding.
   */
  public static final class Binding {
    private final Scope scope;
    private final int index; // index within function (LOCAL) or module (GLOBAL)
    @Nullable private final Identifier first; // first binding use, if syntactic

    private Binding(Scope scope, int index, @Nullable Identifier first) {
      this.scope = scope;
      this.index = index;
      this.first = first;
    }

    /** Returns the name of this binding's identifier. */
    @Nullable
    public String getName() {
      return first != null ? first.getName() : null;
    }

    /** Returns the scope of the binding. */
    public Scope getScope() {
      return scope;
    }

    /** Returns the index of a binding within its function (LOCAL) or module (GLOBAL). */
    public int getIndex() {
      return index;
    }

    @Override
    public String toString() {
      return first == null
          ? scope.toString()
          : String.format(
              "%s[%d] %s @ %s", scope, index, first.getName(), first.getStartLocation());
    }
  }

  /** A Resolver.Function records information about a resolved function. */
  public static final class Function {

    private final String name;
    private final Location location;
    private final ImmutableList<Parameter> params;
    private final ImmutableList<Statement> body;
    private final boolean hasVarargs;
    private final boolean hasKwargs;
    private final int numKeywordOnlyParams;
    private final ImmutableList<String> parameterNames;
    private final boolean isToplevel;
    private final ImmutableList<Binding> locals;
    // TODO(adonovan): move this to Program, but that requires communication
    // between resolveFile and compileFile, which depends on use doing the TODO
    // described at Program.compileResolvedFile and eliminating that function.
    private final ImmutableList<String> globals;

    private Function(
        String name,
        Location loc,
        ImmutableList<Parameter> params,
        ImmutableList<Statement> body,
        boolean hasVarargs,
        boolean hasKwargs,
        int numKeywordOnlyParams,
        List<Binding> locals,
        List<String> globals) {
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
      this.locals = ImmutableList.copyOf(locals);
      this.globals = ImmutableList.copyOf(globals);
    }

    /**
     * Returns the name of the function. It may be "<toplevel>" for the implicit function that holds
     * the top-level statements of a file, or "<expr>" for the implicit function that evaluates a
     * single expression.
     */
    public String getName() {
      return name;
    }

    /** Returns the function's local bindings, parameters first. */
    public ImmutableList<Binding> getLocals() {
      return locals;
    }

    /**
     * Returns the list of names of globals referenced by this function. The order matches the
     * indices used in compiled code.
     */
    public ImmutableList<String> getGlobals() {
      return globals;
    }

    /** Returns the location of the function's identifier. */
    public Location getLocation() {
      return location;
    }

    /**
     * Returns the function's parameters, in "run-time order": non-keyword-only parameters,
     * keyword-only parameters, {@code *args}, and finally {@code **kwargs}. A bare {@code *}
     * parameter is dropped.
     */
    public ImmutableList<Parameter> getParameters() {
      return params;
    }

    /**
     * Returns the effective statements of the function's body. (For the implicit function created
     * to evaluate a single standalone expression, this may contain a synthesized Return statement.)
     */
    // TODO(adonovan): eliminate when we switch to compiler.
    public ImmutableList<Statement> getBody() {
      return body;
    }

    /** Reports whether the function has an {@code *args} parameter. */
    public boolean hasVarargs() {
      return hasVarargs;
    }

    /** Reports whether the function has a {@code **kwargs} parameter. */
    public boolean hasKwargs() {
      return hasKwargs;
    }

    /**
     * Returns the number of the function's keyword-only parameters, such as {@code c} in {@code def
     * f(a, *b, c, **d)} or {@code def f(a, *, c, **d)}.
     */
    public int numKeywordOnlyParams() {
      return numKeywordOnlyParams;
    }

    /** Returns the names of the parameters. Order is as for {@link #getParameters}. */
    public ImmutableList<String> getParameterNames() {
      return parameterNames;
    }

    /**
     * isToplevel indicates that this is the <toplevel> function containing top-level statements of
     * a file.
     */
    // TODO(adonovan): remove this when we remove Bazel's "export" hack,
    // or switch to a compiled representation of function bodies.
    public boolean isToplevel() {
      return isToplevel;
    }
  }

  /**
   * A Module is a static abstraction of a Starlark module (see {@link
   * net.starlark.java.eval.Module})). It describes, for the resolver and compiler, the set of
   * variable names that are predeclared, either by the interpreter (UNIVERSAL) or by the
   * application (PREDECLARED), plus the set of pre-defined global names (which is typically empty,
   * except in a REPL or EvaluationTestCase scenario).
   */
  public interface Module {

    /**
     * Resolves a name to a GLOBAL, PREDECLARED, or UNIVERSAL binding.
     *
     * @throws Undefined if the name is not defined.
     */
    Scope resolve(String name) throws Undefined;

    /**
     * An Undefined exception indicates a failure to resolve a top-level name. If {@code candidates}
     * is non-null, it provides the set of accessible top-level names, which, along with local
     * names, will be used as candidates for spelling suggestions.
     */
    final class Undefined extends Exception {
      @Nullable private final Set<String> candidates;

      public Undefined(String message, @Nullable Set<String> candidates) {
        super(message);
        this.candidates = candidates;
      }
    }
  }

  // A simple implementation of the Module for testing.
  // It defines only the predeclared names---no "universal" names (e.g. None)
  // or initially-defined globals (as happens in a REPL).
  // Realistically, most clients will use an eval.Module.
  // TODO(adonovan): move into test/ tree.
  public static Module moduleWithPredeclared(String... names) {
    ImmutableSet<String> predeclared = ImmutableSet.copyOf(names);
    return (name) -> {
      if (predeclared.contains(name)) {
        return Scope.PREDECLARED;
      }
      throw new Resolver.Module.Undefined(
          String.format("name '%s' is not defined", name), predeclared);
    };
  }

  private static class Block {
    @Nullable private final Block parent; // enclosing block, or null for tail of list
    @Nullable Node syntax; // Comprehension, DefStatement, StarlarkFile, or null
    private final ArrayList<Binding> frame; // accumulated locals of enclosing function

    // Bindings for names defined in this block.
    // Also, as an optimization, memoized lookups of enclosing bindings.
    private final Map<String, Binding> bindings = new HashMap<>();

    Block(@Nullable Block parent, @Nullable Node syntax, ArrayList<Binding> frame) {
      this.parent = parent;
      this.syntax = syntax;
      this.frame = frame;
    }
  }

  private final List<SyntaxError> errors;
  private final FileOptions options;
  private final Module module;
  // List whose order defines the numbering of global variables in this program.
  private final ArrayList<String> globals = new ArrayList<>();
  // A cache of PREDECLARED, UNIVERSAL, and GLOBAL bindings queried from the module.
  private final Map<String, Binding> toplevel = new HashMap<>();
  // Linked list of blocks, innermost first, for functions and comprehensions and (finally) file.
  private Block locals;
  private int loopCount;

  private Resolver(List<SyntaxError> errors, Module module, FileOptions options) {
    this.errors = errors;
    this.module = module;
    this.options = options;
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
  private void createBindingsForBlock(Iterable<Statement> stmts) {
    for (Statement stmt : stmts) {
      createBindings(stmt);
    }
  }

  private void createBindings(Statement stmt) {
    switch (stmt.kind()) {
      case ASSIGNMENT:
        createBindingsForLHS(((AssignmentStatement) stmt).getLHS());
        break;
      case IF:
        IfStatement ifStmt = (IfStatement) stmt;
        createBindingsForBlock(ifStmt.getThenBlock());
        if (ifStmt.getElseBlock() != null) {
          createBindingsForBlock(ifStmt.getElseBlock());
        }
        break;
      case FOR:
        ForStatement forStmt = (ForStatement) stmt;
        createBindingsForLHS(forStmt.getVars());
        createBindingsForBlock(forStmt.getBody());
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

  private void createBindingsForLHS(Expression lhs) {
    for (Identifier id : Identifier.boundIdentifiers(lhs)) {
      bind(id);
    }
  }

  private void assign(Expression lhs) {
    if (lhs instanceof Identifier) {
      // Bindings are created by the first pass (createBindings),
      // so there's nothing to do here.
    } else if (lhs instanceof IndexExpression) {
      visit(lhs);
    } else if (lhs instanceof ListExpression) {
      for (Expression elem : ((ListExpression) lhs).getElements()) {
        assign(elem);
      }
    } else if (lhs instanceof DotExpression) {
      visit(((DotExpression) lhs).getObject());
    } else {
      errorf(lhs, "cannot assign to '%s'", lhs);
    }
  }

  @Override
  public void visit(Identifier id) {
    Binding bind = use(id);
    if (bind != null) {
      id.setBinding(bind);
      return;
    }
  }

  // Resolves a non-binding identifier to an existing binding, or null.
  private Binding use(Identifier id) {
    String name = id.getName();

    // local (to function, comprehension, or file)?
    for (Block b = locals; b != null; b = b.parent) {
      Binding bind = b.bindings.get(name);
      if (bind != null) {
        // Optimization: memoize lookup of an outer local
        // in an inner block, to avoid repeated walks.
        if (b != locals) {
          locals.bindings.put(name, bind);
        }
        return bind;
      }
    }

    // toplevel (global, predeclared, universal)?
    Binding bind = toplevel.get(name);
    if (bind != null) {
      return bind;
    }
    Scope scope;
    try {
      scope = module.resolve(name);
    } catch (Resolver.Module.Undefined ex) {
      if (!Identifier.isValid(name)) {
        // If Identifier was created by Parser.makeErrorExpression, it
        // contains misparsed text. Ignore ex and report an appropriate error.
        errorf(id, "contains syntax errors");
      } else if (ex.candidates != null) {
        // Exception provided toplevel candidates.
        // Show spelling suggestions of all symbols in scope,
        String suggestion = SpellChecker.didYouMean(name, getAllSymbols(ex.candidates));
        errorf(id, "%s%s", ex.getMessage(), suggestion);
      } else {
        errorf(id, "%s", ex.getMessage());
      }
      return null;
    }
    switch (scope) {
      case GLOBAL:
        bind = new Binding(scope, globals.size(), id);
        // Accumulate globals in module.
        globals.add(name);
        break;
      case PREDECLARED:
      case UNIVERSAL:
        bind = new Binding(scope, 0, id); // index not used
        break;
      default:
        throw new IllegalStateException("bad scope: " + scope);
    }
    toplevel.put(name, bind);
    return bind;
  }

  @Override
  public void visit(ReturnStatement node) {
    if (locals.syntax instanceof StarlarkFile) {
      errorf(node, "return statements must be inside a function");
    }
    super.visit(node);
  }

  @Override
  public void visit(CallExpression node) {
    // validate call arguments
    boolean seenVarargs = false;
    boolean seenKwargs = false;
    Set<String> keywords = null;
    for (Argument arg : node.getArguments()) {
      if (arg instanceof Argument.Positional) {
        if (seenVarargs) {
          errorf(arg, "positional argument may not follow *args");
        } else if (seenKwargs) {
          errorf(arg, "positional argument may not follow **kwargs");
        } else if (keywords != null) {
          errorf(arg, "positional argument may not follow keyword argument");
        }

      } else if (arg instanceof Argument.Keyword) {
        String keyword = ((Argument.Keyword) arg).getName();
        if (seenVarargs) {
          errorf(arg, "keyword argument %s may not follow *args", keyword);
        } else if (seenKwargs) {
          errorf(arg, "keyword argument %s may not follow **kwargs", keyword);
        }
        if (keywords == null) {
          keywords = new HashSet<>();
        }
        if (!keywords.add(keyword)) {
          errorf(arg, "duplicate keyword argument: %s", keyword);
        }

      } else if (arg instanceof Argument.Star) {
        if (seenKwargs) {
          errorf(arg, "*args may not follow **kwargs");
        } else if (seenVarargs) {
          errorf(arg, "multiple *args not allowed");
        }
        seenVarargs = true;

      } else if (arg instanceof Argument.StarStar) {
        if (seenKwargs) {
          errorf(arg, "multiple **kwargs not allowed");
        }
        seenKwargs = true;
      }
    }

    super.visit(node);
  }

  @Override
  public void visit(ForStatement node) {
    if (locals.syntax instanceof StarlarkFile) {
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
    if (!(locals.syntax instanceof StarlarkFile)) {
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
    ImmutableList<Comprehension.Clause> clauses = node.getClauses();

    // Following Python3, the first for clause is resolved
    // outside the comprehension block. All the other loops
    // are resolved in the scope of their own bindings,
    // permitting forward references.
    Comprehension.For for0 = (Comprehension.For) clauses.get(0);
    visit(for0.getIterable());

    // A comprehension defines a distinct lexical block
    // in the same function's frame.
    pushLocalBlock(node, this.locals.frame);

    for (Comprehension.Clause clause : clauses) {
      if (clause instanceof Comprehension.For) {
        Comprehension.For forClause = (Comprehension.For) clause;
        createBindingsForLHS(forClause.getVars());
      }
    }
    for (int i = 0; i < clauses.size(); i++) {
      Comprehension.Clause clause = clauses.get(i);
      if (clause instanceof Comprehension.For) {
        Comprehension.For forClause = (Comprehension.For) clause;
        if (i > 0) {
          visit(forClause.getIterable());
        }
        assign(forClause.getVars());
      } else {
        Comprehension.If ifClause = (Comprehension.If) clause;
        visit(ifClause.getCondition());
      }
    }
    visit(node.getBody());
    popLocalBlock();
  }

  @Override
  public void visit(DefStatement node) {
    if (!(locals.syntax instanceof StarlarkFile)) {
      errorf(node, "nested functions are not allowed. Move the function to the top level.");
    }
    node.setResolvedFunction(
        resolveFunction(
            node,
            node.getIdentifier().getName(),
            node.getIdentifier().getStartLocation(),
            node.getParameters(),
            node.getBody()));
  }

  private Function resolveFunction(
      DefStatement def,
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
    ArrayList<Binding> frame = new ArrayList<>();
    pushLocalBlock(def, frame);

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

    createBindingsForBlock(body);
    visitAll(body);
    popLocalBlock();

    return new Function(
        name,
        loc,
        params.build(),
        body,
        star != null && star.getIdentifier() != null,
        starStar != null,
        numKeywordOnlyParams,
        frame,
        globals);
  }

  private void bindParam(ImmutableList.Builder<Parameter> params, Parameter param) {
    if (bind(param.getIdentifier())) {
      errorf(param, "duplicate parameter: %s", param.getName());
    }
    params.add(param);
  }

  @Override
  public void visit(IfStatement node) {
    if (locals.syntax instanceof StarlarkFile) {
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
   * and associate the identifier with it. Reports whether the name was already locally bound in
   * this block.
   */
  private boolean bind(Identifier id) {
    String name = id.getName();
    boolean isNew = false;
    Binding bind;

    // outside any function/comprehension? => GLOBAL binding.
    if (locals.syntax instanceof StarlarkFile) {
      // TODO(adonovan): make load statements bind locally.
      // (Will need 'boolean local' param.)
      bind = toplevel.get(name);
      if (bind == null) {
        isNew = true; // new global binding
        bind = new Binding(Scope.GLOBAL, globals.size(), id);
        // Accumulate globals in module.
        globals.add(name);
        toplevel.put(name, bind);
      } else if (!options.allowToplevelRebinding()) {
        // TODO(adonovan): rephrase error in terms of spec.
        errorf(
            id,
            "cannot reassign global '%s' (read more at"
                + " https://bazel.build/versions/master/docs/skylark/errors/read-only-variable.html)",
            name);
        if (bind.first != null) {
          errorf(bind.first, "'%s' previously declared here", name);
        }
      }

    } else {
      // Binding is local to function or comprehension.
      bind = locals.bindings.get(name);
      if (bind == null) {
        isNew = true; // new local binding
        bind = new Binding(Scope.LOCAL, locals.frame.size(), id);
        locals.bindings.put(name, bind);
        // Accumulate locals in frame of enclosing function.
        locals.frame.add(bind);
      }
    }

    id.setBinding(bind);
    return !isNew;
  }

  // Returns the union of accessible local and top-level symbols.
  private Set<String> getAllSymbols(Set<String> predeclared) {
    Set<String> all = new HashSet<>();
    for (Block b = locals; b != null; b = b.parent) {
      all.addAll(b.bindings.keySet());
    }
    all.addAll(predeclared);
    all.addAll(toplevel.keySet());
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

  /**
   * Performs static checks, including resolution of identifiers in {@code file} in the environment
   * defined by {@code module}. The StarlarkFile is mutated. Errors are appended to {@link
   * StarlarkFile#errors}.
   */
  public static void resolveFile(StarlarkFile file, Module module) {
    Resolver r = new Resolver(file.errors, module, file.getOptions());
    ImmutableList<Statement> stmts = file.getStatements();

    // Check that load statements are on top.
    if (r.options.requireLoadStatementsFirst()) {
      r.checkLoadAfterStatement(stmts);
    }

    ArrayList<Binding> frame = new ArrayList<>();
    r.pushLocalBlock(file, frame);

    // First pass: creating bindings for statements in this block.
    r.createBindingsForBlock(stmts);

    // Second pass: visit all references.
    r.visitAll(stmts);

    r.popLocalBlock();

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
    file.setResolvedFunction(
        new Function(
            "<toplevel>",
            file.getStartLocation(),
            /*params=*/ ImmutableList.of(),
            /*body=*/ stmts,
            /*hasVarargs=*/ false,
            /*hasKwargs=*/ false,
            /*numKeywordOnlyParams=*/ 0,
            frame,
            r.globals));
  }

  /**
   * Performs static checks, including resolution of identifiers in {@code expr} in the environment
   * defined by {@code module}. This operation mutates the Expression. Syntax must be resolved
   * before it is evaluated.
   */
  public static Function resolveExpr(Expression expr, Module module, FileOptions options)
      throws SyntaxError.Exception {
    List<SyntaxError> errors = new ArrayList<>();
    Resolver r = new Resolver(errors, module, options);

    ArrayList<Binding> frame = new ArrayList<>();
    r.pushLocalBlock(null, frame); // for bindings in list comprehensions
    r.visit(expr);
    r.popLocalBlock();

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
        /*numKeywordOnlyParams=*/ 0,
        frame,
        r.globals);
  }

  private void pushLocalBlock(Node syntax, ArrayList<Binding> frame) {
    locals = new Block(locals, syntax, frame);
  }

  private void popLocalBlock() {
    locals = locals.parent;
  }
}
