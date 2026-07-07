// Copyright 2025 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.errorprone.annotations.FormatMethod;
import java.util.ArrayDeque;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.spelling.SpellChecker;
import net.starlark.java.syntax.Resolver.Scope;

/**
 * A visitor for tagging the data structures of a resolved file with type information.
 *
 * <p>This populates the function type on the {@link Resolver.Function} objects in the AST and
 * records whether or not a given {@link Resolver.Function} is considered to use static type syntax;
 * populates the variable types on the {@link Resolver.Binding} objects; and populates the Starlark
 * type stored in {@link CastExpression}s. These type fields must all be null prior to running the
 * visitor.
 *
 * <p>The types assigned to the fields are based solely on the type annotations in the program. No
 * type inference is done here.
 *
 * <p>Only a file that has passed the {@code Resolver} without errors should be run through this
 * visitor.
 */
public final class TypeTagger extends NodeVisitor {

  /**
   * An immutable view of a {@code load()} dependency. Provides the exported symbols (in practice:
   * the evaluated module's global variables) and their types.
   *
   * <p>Contrast with {@link Resolver.Module}, which resolves a program's own names during the
   * process of its compilation and type checking. A {@link LoadableModule} and {@link
   * Resolver.Module} in theory need not be objects of the same class (although in practice, they
   * are; see {@link net.starlark.java.eval.Module}).
   */
  public interface LoadableModule {
    /** Returns the symbols (in practice, global variables) exported by this module. */
    Set<String> getExports();

    /** Returns whether the module exports a given symbol. */
    boolean hasExport(String name);

    /**
     * Returns the Starlark type of the specified exported symbol, or null if the export was not
     * assigned a type (in particular, if type tagging for the module was disabled).
     */
    @Nullable
    StarlarkType getExportType(String name);

    /**
     * Returns the Starlark type constructor value of the specified exported symbol, or null if the
     * export does not have a type constructor value.
     */
    @Nullable
    TypeConstructor getExportTypeConstructor(String name);
  }

  /** Returns the named module, or null if not found. */
  @FunctionalInterface
  public interface Loader {
    @Nullable
    LoadableModule load(String importName);
  }

  private final TypeTable typeTable;

  private final Resolver.Module module;

  @Nullable private final Loader loader;

  // Empty if we are tagging a type expression (inside which no function definitions are allowed).
  // Populated and mutated by visitation.
  private final ArrayDeque<Resolver.Function> functionStack = new ArrayDeque<>();

  // Global and file-local symbols of type constructors defined or loaded in this file. Used only
  // for spelling suggestions in error messages. (Note that TypeTable doesn't store names of type
  // constructor symbols.)
  private final LinkedHashSet<String> fileDefinedTypeConstructorNames = new LinkedHashSet<>();

  // Formats and reports an error at the start of the specified node.
  @FormatMethod
  private void errorf(Node node, String format, Object... args) {
    errorf(node.getStartLocation(), format, args);
  }

  // Formats and reports an error at the specified location.
  @FormatMethod
  private void errorf(Location loc, String format, Object... args) {
    typeTable.errors.add(new SyntaxError(loc, String.format(format, args)));
  }

  private TypeTagger(TypeTable typeTable, Resolver.Module module, @Nullable Loader loader) {
    this.typeTable = typeTable;
    this.module = module;
    this.loader = loader;
  }

  private TypeTagger(
      TypeTable typeTable,
      Resolver.Module module,
      @Nullable Loader loader,
      Resolver.Function toplevel) {
    this(typeTable, module, loader);
    functionStack.push(toplevel);
  }

  TypeTable getTypeTable() {
    return typeTable;
  }

  /**
   * Given an identifier denoting a type constructor, obtains the type constructor from the module.
   *
   * <p>If no match, logs an error at the given node and returns null.
   */
  @Nullable
  private TypeConstructor resolveTypeConstructor(Identifier id) {
    String name = id.getName();

    var binding = id.getBinding();
    var scope = binding.getScope();
    @Nullable TypeConstructor constructor = null;
    if (!(scope == Scope.UNIVERSAL || scope == Scope.PREDECLARED || scope == Scope.GLOBAL)) {
      // Non-file-level local names cannot be types. Don't allow `x: Foo` to succeed if Foo is a
      // local shadowing a type name.
      if (binding.isToplevelLocal()) {
        constructor = typeTable.getTypeConstructor(binding);
      }
      if (constructor != null) {
        return constructor;
      } else {
        errorf(id, "local symbol '%s' cannot be used as a type", name);
        return null;
      }
    }

    try {
      constructor = module.getTypeConstructor(name);
      if (constructor == null) {
        errorf(id, "%s symbol '%s' cannot be used as a type", scope, name);
        return null;
      }
      return constructor;
    } catch (Resolver.Module.Undefined ex) {
      LinkedHashSet<String> candidates = new LinkedHashSet<>(fileDefinedTypeConstructorNames);
      if (ex.candidates != null) {
        candidates.addAll(ex.candidates);
      }
      String suggestion = candidates.isEmpty() ? "" : SpellChecker.didYouMean(name, candidates);
      errorf(id, "%s%s", ex.getMessage(), suggestion);
      return null;
    }
  }

  private TypeConstructor.Arg extractArg(Expression expr) {
    switch (expr.kind()) {
      case BINARY_OPERATOR -> {
        // Syntax sugar for union types, i.e. a|b == Union[a,b]
        BinaryOperatorExpression binop = (BinaryOperatorExpression) expr;
        if (binop.getOperator() == TokenKind.PIPE) {
          StarlarkType x = extractType(binop.getX());
          StarlarkType y = extractType(binop.getY());
          return Types.union(x, y);
        }
        errorf(expr, "binary operator '%s' is not supported", binop.getOperator());
        return Types.ANY;
      }
      case TYPE_APPLICATION -> {
        TypeApplication app = (TypeApplication) expr;

        TypeConstructor constructor = resolveTypeConstructor(app.getConstructor());
        if (constructor == null) {
          return Types.ANY;
        }
        ImmutableList<TypeConstructor.Arg> arguments =
            app.getArguments().stream().map(this::extractArg).collect(toImmutableList());

        try {
          return constructor.createStarlarkType(arguments);
        } catch (TypeConstructor.Failure e) {
          errorf(expr, "%s", e.getMessage());
          return Types.ANY;
        }
      }
      case IDENTIFIER -> {
        TypeConstructor constructor = resolveTypeConstructor((Identifier) expr);
        if (constructor == null) {
          return Types.ANY;
        }
        try {
          return constructor.createStarlarkType(ImmutableList.of());
        } catch (TypeConstructor.Failure e) {
          errorf(expr, "%s", e.getMessage());
          return Types.ANY;
        }
      }
      case ELLIPSIS -> {
        return TypeConstructor.Arg.ELLIPSIS;
      }
      case LIST_EXPR -> {
        ListExpression listExpr = (ListExpression) expr;
        if (listExpr.isTuple() && listExpr.getElements().isEmpty()) {
          return TypeConstructor.Arg.EMPTY_TUPLE;
        }
      }
      case DICT_EXPR -> {
        DictExpression dictExpr = (DictExpression) expr;
        LinkedHashMap<String, StarlarkType> types = new LinkedHashMap<>();
        for (DictExpression.Entry entry : dictExpr.getEntries()) {
          if (entry.getKey() instanceof StringLiteral str) {
            String key = str.getValue();
            @Nullable var previous = types.put(key, extractType(entry.getValue()));
            if (previous != null) {
              errorf(str, "dictionary expression has duplicate key: %s", str);
            }
          } else {
            errorf(entry.getKey(), "expected a string literal but got '%s'", entry.getKey());
          }
        }
        return new TypeConstructor.Arg.TypeDict(ImmutableMap.copyOf(types));
      }
      default -> {
        // fall through
      }
    }
    // TODO(ilist@): full evaluation: lists and dicts
    errorf(expr, "unexpected expression '%s'", expr);
    return Types.ANY;
  }

  private StarlarkType extractType(Expression expr) {
    TypeConstructor.Arg arg = extractArg(expr);
    if (!(arg instanceof StarlarkType type)) {
      errorf(expr, "expression '%s' is not a valid type.", expr);
      return Types.ANY;
    }
    return type;
  }

  /**
   * Statically evaluates a type expression to the {@link StarlarkType} it denotes.
   *
   * @param expr a valid type expression with binding information resolved, which must have been
   *     parsed with the appropriate {@link FileOptions} set; see {@link #tagFile}
   * @param exprFunction the resolver function for {@code expr} constructed by {@link
   *     Resolver#resolveExpr()}
   * @param module a static Resolver.Module containing type information for the bindings used in
   *     type expressions
   * @throws SyntaxError.Exception if expr is not a type expression or if it could not be evaluated
   *     to a type.
   */
  static StarlarkType extractType(
      Expression expr, Resolver.Function exprFunction, Resolver.Module module)
      throws SyntaxError.Exception {
    // loader is null because expressions cannot contain load statements.
    TypeTagger r = new TypeTagger(new TypeTable(exprFunction), module, /* loader= */ null);
    StarlarkType result = r.extractType(expr);
    if (!r.getTypeTable().ok()) {
      throw new SyntaxError.Exception(r.getTypeTable().errors());
    }
    return result;
  }

  private Types.CallableType createFunctionType(
      ImmutableList<Parameter> parameters, @Nullable Expression returnTypeExpr) {
    ImmutableList.Builder<String> names = ImmutableList.builder();
    ImmutableList.Builder<StarlarkType> types = ImmutableList.builder();
    ImmutableSet.Builder<String> mandatoryParameters = ImmutableSet.builder();

    int nparams = parameters.size();
    int numPositionalParameters = 0;
    Parameter.Star star = null;
    Parameter.StarStar starStar = null;
    int i;
    for (i = 0; i < nparams; i++) {
      Parameter param = parameters.get(i);
      if (param instanceof Parameter.Star pstar) {
        star = pstar;
        continue;
      }
      if (param instanceof Parameter.StarStar pstarstar) {
        starStar = pstarstar;
        continue;
      }
      if (star == null) {
        numPositionalParameters++;
      }

      String name = param.getName();
      Expression typeExpr = param.getType();

      names.add(name);
      types.add(typeExpr == null ? Types.ANY : extractType(typeExpr));
      if (param instanceof Parameter.Mandatory) {
        mandatoryParameters.add(name);
      }
    }

    StarlarkType varargsType = null;
    if (star != null && star.getIdentifier() != null) {
      Expression typeExpr = star.getType();
      varargsType = typeExpr == null ? Types.ANY : extractType(typeExpr);
    }

    StarlarkType kwargsType = null;
    if (starStar != null) {
      Expression typeExpr = starStar.getType();
      kwargsType = typeExpr == null ? Types.ANY : extractType(typeExpr);
    }

    StarlarkType returnType = Types.ANY;
    if (returnTypeExpr != null) {
      returnType = extractType(returnTypeExpr);
    }

    return Types.callable(
        names.build(),
        types.build(),
        /* numPositionalOnlyParameters= */ 0,
        numPositionalParameters,
        mandatoryParameters.build(),
        varargsType,
        kwargsType,
        returnType);
  }

  /**
   * Sets an identifier's type.
   *
   * <p>The {@code Binding} on the identifier must have already been set by the resolver.
   * (Therefore, this method cannot be called for identifiers that are not symbols, like field names
   * or call site keyword arguments.)
   *
   * <p>Logs an error if the identifier is not the first binding occurrence of the {@code Binding}.
   * In this case, the type is not updated.
   *
   * <p>Throws {@link IllegalArgumentException} if this is the first binding occurrence but somehow
   * the type is already set.
   */
  private void setType(Node node, Identifier id, StarlarkType type) {
    Resolver.Binding binding = id.getBinding();
    checkNotNull(binding, "no binding set on identifier '%s'", id.getName());

    if (binding.getFirst() != id) {
      if (node instanceof DefStatement) {
        // A def statement appearing in typed code constitutes an implicit type annotation on the
        // function identifier's symbol. Even if the signature contains no type annotations, the
        // function identifier is still considered to be marked as a Callable. Therefore, this needs
        // to be the first binding occurrence of the symbol.
        //
        // A consequence of this is that `def f(): ...; f = lambda: ...` is permitted by the
        // type tagger (though the type checker will still require the assignment to be consistent
        // with the def's type signature), even though the opposite statement order is prohibited.
        //
        // When a violation occurs at a def statement, we use a more specific error message to avoid
        // confusing the user.
        errorf(id, "function '%s' was previously declared", id.getName());
      } else {
        errorf(id, "type annotation on '%s' may only appear at its declaration", id.getName());
      }
      if (binding.isSyntactic()) {
        errorf(binding.getFirst(), "'%s' previously declared here", id.getName());
      }
      errorIfTypeConstructor(node, id);
      return;
    }

    @Nullable StarlarkType prevType = typeTable.getType(binding);
    if (prevType != null) {
      throw new IllegalArgumentException(
          String.format("Expected type of binding %s to be null but was %s", binding, prevType));
    }
    typeTable.setDeclaredType(binding, type);
  }

  /**
   * Sets the type constructor value associated with a given binding, making it available for
   * subsequent type tagging and checking.
   */
  private void setTypeConstructor(Node node, Identifier id, TypeConstructor typeConstructor) {
    Resolver.Binding binding = id.getBinding();
    checkNotNull(binding, "no binding set on identifier '%s'", id.getName());
    checkArgument(
        binding.getScope() == Resolver.Scope.GLOBAL || binding.isToplevelLocal(),
        "'%s' must be either a global or a file-level local",
        id.getName());

    if (errorIfTypeConstructor(node, id)) {
      return;
    }

    fileDefinedTypeConstructorNames.add(id.getName());
    typeTable.setTypeConstructor(binding, typeConstructor);
  }

  /**
   * Returns true and logs an error if the given symbol has been associated with a type constructor;
   * otherwise, returns false.
   */
  private boolean errorIfTypeConstructor(Node node, Identifier id) {
    if (typeTable.getTypeConstructor(id.getBinding()) != null) {
      // A type constructor cannot be redeclared, even if allowTopLevelRebinding is set.
      // TODO: #27370 - Allow types to be redeclared in REPL. What we really want to prevent is
      // redeclaration only within the same program (the same set of statements passed to
      // TypeTagger/TypeChecker); but redeclaration in a different program which happens to mutate
      // the same globals should be fine.
      errorf(node, "type '%s' redeclared", id.getName());
      errorf(id.getBinding().getFirst(), "'%s' previously declared here", id.getName());
      return true;
    }
    return false;
  }

  /**
   * Sets a resolved function's type.
   *
   * <p>Throws {@link IllegalArgumentException} if the type is already set.
   */
  private static void setType(
      Resolver.Function resolved, Types.CallableType type, TypeTable typeTable) {
    checkNotNull(resolved);
    @Nullable StarlarkType prevType = typeTable.getType(resolved);
    if (prevType != null) {
      throw new IllegalArgumentException(
          String.format(
              "Expected type of resolved function %s to be null but was %s",
              resolved.getName(), prevType));
    }
    typeTable.setType(resolved, type);
  }

  private void setType(Resolver.Function resolved, Types.CallableType type) {
    setType(resolved, type, typeTable);
  }

  private void visitProgram(Program prog) {
    checkState(
        functionStack.isEmpty(),
        "When tagging a Program, functionStack is expected to be initially empty");
    Resolver.Function toplevel = prog.getResolvedFunction();
    this.functionStack.push(toplevel);
    visitBlock(toplevel.getBody());
    checkState(functionStack.pop().equals(toplevel));
  }

  @Override
  public void visit(StarlarkFile file) {
    checkState(
        functionStack.isEmpty(),
        "When tagging a StarlarkFile, functionStack is expected to be initially empty");
    Resolver.Function toplevel = file.getResolvedFunction();
    this.functionStack.push(toplevel);
    super.visit(file);
    checkState(functionStack.pop().equals(toplevel));
  }

  @Override
  public void visit(AssignmentStatement assignment) {
    if (assignment.getType() != null) {
      setUsesTypeSyntax();
      StarlarkType type = extractType(assignment.getType());
      setType(assignment, (Identifier) assignment.getLHS(), type);
    }

    for (Identifier id : Identifier.boundIdentifiers(assignment.getLHS())) {
      // TODO: #27370 - This is brittle: if loadsBindGlobally and allowToplevelRebinding are both
      // set, the exporting file may break the loading file by changing an exported value to be a
      // TypeConstructor instance. One solution may be to run the check only for symbols which are
      // used by type annotations in this file. (That could also fix the REPL use case.)
      errorIfTypeConstructor(assignment, id);
    }

    // Traverse children; RHS could contain a lambda.
    super.visit(assignment);
  }

  @Override
  public void visit(DefStatement def) {
    Resolver.Function resolvedFunction = def.getResolvedFunction();
    functionStack.push(resolvedFunction);
    Types.CallableType type = createFunctionType(def.getParameters(), def.getReturnType());
    setType(resolvedFunction, type);
    setType(def, def.getIdentifier(), type);
    // Parameter types handled by visit(Parameter).
    if (def.getReturnType() != null || !def.getTypeParameters().isEmpty()) {
      setUsesTypeSyntax();
    }

    super.visit(def);
    checkState(functionStack.pop().equals(resolvedFunction));
  }

  @Override
  public void visit(Parameter param) {
    if (param.getIdentifier() != null) {
      // Default to ANY for unannotated params.
      // This matches the behavior for the Resolver.Function's type.
      StarlarkType type = Types.ANY;
      if (param.getType() != null) {
        setUsesTypeSyntax();
        type = extractType(param.getType());
      }
      setType(param, param.getIdentifier(), type);
    }

    super.visit(param);
  }

  @Override
  public void visit(LoadStatement load) {
    if (loader == null) {
      errorf(load, "load statements are not supported because no module loader has been defined");
      return;
    }
    String importName = load.getImport().getValue();
    @Nullable LoadableModule loadedModule = loader.load(importName);
    if (loadedModule == null) {
      errorf(load, "module '%s' not found", importName);
      return;
    }
    for (LoadStatement.Binding binding : load.getBindings()) {
      String originalName = binding.getOriginalName().getName();
      if (!loadedModule.hasExport(originalName)) {
        errorf(
            binding.getOriginalName(),
            "module '%s' does not contain symbol '%s'%s",
            importName,
            originalName,
            SpellChecker.didYouMean(originalName, loadedModule.getExports()));
        continue;
      }
      setType(load, binding.getLocalName(), loadedModule.getExportType(originalName));
      @Nullable
      TypeConstructor typeConstructor = loadedModule.getExportTypeConstructor(originalName);
      if (typeConstructor != null) {
        setTypeConstructor(load, binding.getLocalName(), typeConstructor);
      }
    }
  }

  @Override
  public void visit(TypeAliasStatement node) {
    setUsesTypeSyntax();
    errorIfTypeConstructor(node, node.getIdentifier());
    super.visit(node);
  }

  @Override
  public void visit(VarStatement var) {
    StarlarkType type = extractType(var.getType());
    setType(var, var.getIdentifier(), type);
    setUsesTypeSyntax();

    // No need to descend into type expression child.
  }

  // TODO: #28325 - Ensure we assign the type of an identifier referencing a universal/predeclared
  // symbol, i.e. with no binding occurrences in the file.

  @Override
  public void visit(CastExpression cast) {
    setUsesTypeSyntax();
    cast.setStarlarkType(extractType(cast.getType()));
    super.visit(cast);
  }

  @Override
  public void visit(LambdaExpression lambda) {
    Types.CallableType type =
        createFunctionType(lambda.getParameters(), /* returnTypeExpr= */ null);
    setType(lambda.getResolvedFunction(), type);

    super.visit(lambda);
  }

  // TODO: #27370 - Figure out the relationship between this visitor and identifiers introduced by
  // type alias statements. I don't think it's quite correct to say that `type A = B` is annotating
  // A's binding with the evaluation of type B. It probably should live in outer logic that
  // determines the type environment.

  private static void checkFileOptions(FileOptions options) {
    checkArgument(
        options.resolveTypeSyntax(), "type tagging requires that resolveTypeSyntax is set");
    checkArgument(
        !options.tolerateInvalidTypeExpressions(),
        "type tagging requires that tolerateInvalidTypeExpressions is not set");
  }

  /**
   * Determines the Starlark types of the {@link Resolver.Function}s and {@link Resolver.Binding}s
   * in the given AST (which must have already been processed by {@link Resolver}), based on the
   * supplied annotations. Returns the resulting {@link TypeTable} for the file.
   *
   * <p>Any errors are appended to the file's list of errors.
   *
   * @throws IllegalArgumentException if the file's {@link FileOptions} don't contain {@link
   *     FileOptions#resolveTypeSyntax()} or do contain {@link
   *     FileOptions#tolerateInvalidTypeExpressions()}.
   * @param loader a {@link Loader} for loading modules via load() statements; may be null if the
   *     file is known to not contain load() statements
   */
  public static TypeTable tagFile(
      StarlarkFile file, Resolver.Module module, @Nullable Loader loader) {
    checkFileOptions(file.getOptions());
    TypeTable typeTable = new TypeTable(file);
    TypeTagger r = new TypeTagger(typeTable, module, loader);
    r.visit(file);
    return typeTable;
  }

  /**
   * Like {@link #tagFile}, but on an already-compiled {@link Program}.
   *
   * <p>The program is *not* mutated. In particular, the pre-existing {@link Program#getTypeTable}
   * (if any) is ignored. Any errors are reported in the returned type table's {@link
   * TypeTable#errors()} list.
   */
  public static TypeTable tagProgram(
      Program prog, Resolver.Module module, @Nullable Loader loader) {
    checkFileOptions(prog.getOptions());
    Resolver.Function toplevel = prog.getResolvedFunction();
    TypeTable typeTable = new TypeTable(toplevel);
    TypeTagger r = new TypeTagger(typeTable, module, loader);
    r.visitProgram(prog);
    return typeTable;
  }

  /**
   * Same as {@link #tagFile}, but for an individual expression.
   *
   * <p>Any errors are thrown as a {@link SyntaxError.Exception}.
   *
   * @param function the {@link Resolver.Function} that the resolver generated to wrap an
   *     expression.
   */
  public static TypeTable tagExpr(
      Expression expr, Resolver.Function function, Resolver.Module module)
      throws SyntaxError.Exception {
    TypeTable typeTable = new TypeTable(function);
    // Use a null loader because load() cannot appear in expressions.
    TypeTagger r = new TypeTagger(typeTable, module, /* loader= */ null, function);

    r.visit(expr);

    if (!typeTable.ok()) {
      throw new SyntaxError.Exception(typeTable.errors());
    }
    return typeTable;
  }

  private void setUsesTypeSyntax() {
    // If anything in the file (or in the expr if TypeTagger is invoked via tagExpr()) uses type
    // syntax, the toplevel is considered to use type syntax.
    typeTable.setUsesTypeSyntax(functionStack.peekLast());
    // If anything nested in the most proximate def statement uses type syntax, the def statement
    // is considered to use type syntax
    typeTable.setUsesTypeSyntax(functionStack.peek());
  }
}
