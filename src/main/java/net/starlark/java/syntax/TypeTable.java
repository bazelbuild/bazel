// Copyright 2026 The Bazel Authors. All rights reserved.
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
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * The static type information for a Starlark file.
 *
 * <p>This information comes from the following sources:
 *
 * <ul>
 *   <li>Declared types: from type annotations in var and assignment statements (for ordinary
 *       variables); from function parameter annotations (for parameter variables); and from the
 *       function signature (possibly annotated) in a def statement (for functions).
 *   <li>Inferred types: computed by the static type checker. Never supersedes a declared type.
 *   <li>Loaded exported types: types of symbols that this module {@code load()}s from other
 *       modules. This is the symbol's declared type in its exporting module; or if there's no
 *       declared type, it's the *dynamic* type of the symbol after the exporting module's
 *       evaluation.
 * </ul>
 *
 * <p>At first glance, one may think this information ought to be directly embedded in a {@link
 * Program}'s AST nodes. But storing it separately has some benefits: for one thing, we can do a
 * second static typechecking pass post-evaluation without having evaluation mutate the AST. And it
 * supports Bazel's model of Starlark evaluation under which compilation (whose output needs to be
 * immutable) is separated from type-checking and evaluation.
 *
 * <p>Initialized by {@link TypeTagger} and further refined by {@link TypeChecker}.
 */
public final class TypeTable {
  // Empty array singleton for use by getAnnotatedFunctionTypes().
  private static final Types.CallableType[] EMPTY_CALLABLE_ARRAY = new Types.CallableType[0];
  // Number of globals in the program; determined by the resolver.
  private final int numGlobals;
  // Total number of functions in the program (including toplevel, nested functions, and lambdas);
  // determined by the resolver. Equals the max function sequence number + 1.
  private final int numFunctions;

  // Declared types of global bindings. Indexed by the binding's global scope index. Elements may be
  // * StarlarkType - for non-function symbols having a declared type, and for functions whose
  //   signature has type annotations;
  // * Resolver.Function - for functions with unannotated signatures;
  // * null - for non-function symbols without a declared type.
  // Lazily initialized by setDeclaredType() / setDeclaredTypeInternal().
  // (Note: unlike for bindingTypes below, here we don't update Resolver.Function values from a
  // getter because getters on the globalsDeclaredTypes operate after type checking has finished and
  // the type table is frozen.)
  @Nullable private Object[] globalsDeclaredTypes;
  // Types of bindings (regardless of scope). Indexed by binding's sequence number. Elements may be
  // * StarlarkType - for non-function symbols having a declared or inferred type, and for functions
  //   whose signature has type annotations.
  // * Resolver.Function - for functions with unannotated signatures; getAndCacheType() may update
  //   the latter to the former to cache the Type.untypedFunction()
  // * null - for non-function variables without static type information.
  private final Object[] bindingTypes;
  // Type constructor values of bindings, indexed by the binding's sequence number. Sparse because
  // most bindings are not type constructors; lazily initialized by setTypeConstructor().
  @Nullable private LinkedHashMap<Integer, TypeConstructor> typeConstructors;
  // Types of functions whose signatures have type annotations. Indexed by function's sequence
  // number. Lazily initialized by setAnnotatedFunctionType(). Contains nulls for functions with
  // unannotated signatures. Contains a subset of information in bindingTypes, but is helpful for
  // retrieval keyed by function. Needed only for StarlarkFunction#getStarlarkType().
  @Nullable private Types.GeneralCallableType[] annotatedFunctionTypes;
  // Whether a function uses type syntax. Indexed by function's sequence number.
  private final boolean[] functionsUsingTypeSyntax;

  private List<SyntaxError> errors = new ArrayList<>();
  // Subset of the above errors list.
  @Nullable private Set<SyntaxError> errorsRelevantOnlyInTypedCode = null;

  /** Constructs a {@link TypeTable} large enough to hold typeable entities in the given file. */
  public TypeTable(StarlarkFile file) {
    this(file.getResolvedFunction());
  }

  /**
   * Constructs a {@link TypeTable} large enough to hold typeable entities in the given toplevel
   * function.
   */
  TypeTable(Resolver.Function toplevel) {
    this(
        toplevel.getGlobals().size(),
        toplevel.getNumBindingsInFile(),
        // toplevel's function sequence number is the max function sequence number in the file.
        toplevel.getFunctionId() + 1);
  }

  private TypeTable(int numGlobals, int numBindings, int numFunctions) {
    this.numGlobals = numGlobals;
    this.numFunctions = numFunctions;
    this.bindingTypes = new Object[numBindings];
    this.functionsUsingTypeSyntax = new boolean[numFunctions];
  }

  /** Records a syntax error in the type table. */
  void addError(SyntaxError error) {
    if (errors == null) {
      errors = new ArrayList<>();
    }
    errors.add(error);
  }

  /** Records a syntax error in the type table. */
  void addErrorRelevantOnlyInTypedCode(SyntaxError error) {
    addError(error);
    if (errorsRelevantOnlyInTypedCode == null) {
      errorsRelevantOnlyInTypedCode = new LinkedHashSet<>();
    }
    errorsRelevantOnlyInTypedCode.add(error);
  }

  /**
   * Returns the list of errors recorded in the type table (excluding errors relevant only in typed
   * code if the file doesn't use type syntax).
   */
  public ImmutableList<SyntaxError> errors() {
    if (errors == null) {
      return ImmutableList.of();
    }
    if (!toplevelUsesTypeSyntax() && !errors.isEmpty() && errorsRelevantOnlyInTypedCode != null) {
      return errors.stream()
          .filter(e -> !errorsRelevantOnlyInTypedCode.contains(e))
          .collect(toImmutableList());
    }
    return ImmutableList.copyOf(errors);
  }

  /**
   * Returns true if no errors were recorded in the type table. If the file doesn't use type syntax,
   * errors relevant only in typed code are ignored.
   *
   * <p>Invariant: {@link ok()} is true iff {@link errors()} is empty.
   */
  public boolean ok() {
    if (errors == null || errors.isEmpty()) {
      return true;
    }
    if (!toplevelUsesTypeSyntax() && errorsRelevantOnlyInTypedCode != null) {
      return errors.stream()
          .filter(e -> !errorsRelevantOnlyInTypedCode.contains(e))
          .findAny()
          .isEmpty();
    }
    return false;
  }

  /** Sets the declared (annotated) type of the given binding. May be called more than once. */
  void setDeclaredType(Resolver.Binding binding, StarlarkType type) {
    setDeclaredTypeInternal(binding, type);
  }

  /**
   * Sets the declared type of the given binding to be the callable type (which will be computed on
   * demand) of the given function with an unannotated signature.
   */
  void setUnannotatedFunctionType(Resolver.Binding binding, Resolver.Function rfn) {
    setDeclaredTypeInternal(binding, rfn);
  }

  /**
   * Sets the declared type of the given binding and its corresponding function whose signature has
   * type annotations.
   */
  void setAnnotatedFunctionType(
      Resolver.Binding binding, Resolver.Function rfn, Types.GeneralCallableType type) {
    setDeclaredTypeInternal(binding, type);
    if (annotatedFunctionTypes == null && type != null) {
      annotatedFunctionTypes = new Types.GeneralCallableType[numFunctions];
    }
    annotatedFunctionTypes[rfn.getFunctionId()] = type;
  }

  /** Clears the declared type of the given binding. */
  void clearDeclaredType(Resolver.Binding binding) {
    setDeclaredTypeInternal(binding, null);
  }

  /**
   * Sets the declared type of the given binding to a {@link StarlarkType} or a {@link
   * Resolver.Function} (for callable types of functions with unannotated signatures), or clears the
   * declared type (if given null).
   */
  private void setDeclaredTypeInternal(Resolver.Binding binding, @Nullable Object typeOrRfn) {
    bindingTypes[binding.getBindingId()] = typeOrRfn;
    if (binding.getScope().equals(Resolver.Scope.GLOBAL)) {
      boolean wantUpdateGlobal = typeOrRfn != null || globalsDeclaredTypes != null;
      if (wantUpdateGlobal) {
        if (globalsDeclaredTypes == null) {
          globalsDeclaredTypes = new Object[numGlobals];
        }
        globalsDeclaredTypes[binding.getIndex()] = typeOrRfn;
      }
    }
  }

  boolean hasBindingType(Resolver.Binding binding) {
    return bindingTypes[binding.getBindingId()] != null;
  }

  /**
   * Sets the inferred type of the given binding. May be called more than once. Null is treated is
   * untyped / Any.
   */
  void setInferredType(Resolver.Binding binding, StarlarkType type) {
    bindingTypes[binding.getBindingId()] = type;
  }

  /**
   * Records the type of a function whose signature has type annotations. Doesn't update the binding
   * type; the caller must do so by calling {@link #setDeclaredType}.
   */
  void setAnnotatedType(Resolver.Function function, Types.GeneralCallableType type) {
    if (annotatedFunctionTypes == null && type != null) {
      annotatedFunctionTypes = new Types.GeneralCallableType[numFunctions];
    }
    annotatedFunctionTypes[function.getFunctionId()] = type;
  }

  /**
   * Sets (or clears, if given a null) the {@link TypeConstructor} value defined in or loaded by
   * this file, so that it can be used when the binding appears in a type expression. May be called
   * more than once.
   *
   * <p>Associating a type constructor with a symbol (e.g. via an assignment) does not by itself
   * make the symbol usable in type expressions.
   *
   * @param binding the global or file-level local binding to associate with a type constructor
   */
  void setTypeConstructor(Resolver.Binding binding, @Nullable TypeConstructor typeConstructor) {
    checkArgument(binding.getScope() == Resolver.Scope.GLOBAL || binding.isToplevelLocal());
    if (typeConstructors == null) {
      if (typeConstructor == null) {
        return;
      }
      typeConstructors = new LinkedHashMap<>();
    }
    typeConstructors.put(binding.getBindingId(), typeConstructor);
  }

  /**
   * Returns the {@link TypeConstructor} value associated with the given binding by {@link
   * #setTypeConstructor}. Null indicates that the binding doesn't (and, perhaps, cannot) have a
   * usable type constructor value that was defined in or loaded by this file.
   */
  @Nullable
  public TypeConstructor getTypeConstructor(Resolver.Binding binding) {
    return typeConstructors == null ? null : typeConstructors.get(binding.getBindingId());
  }

  /**
   * Returns the declared type of the global binding, or null if the global binding doesn't have a
   * declared type.
   *
   * <p>Used in the evaluation phase to set the declared type as the exported type in eval.Module.
   */
  // TODO: #27370 - Instead of having Eval#assignIdentifier read the declared type and set it on the
  // module via the <toplevel> function, pass the list of declared types to eval.Module directly.
  @Nullable
  public StarlarkType getGlobalDeclaredType(Resolver.Binding binding) {
    checkArgument(binding.getScope().equals(Resolver.Scope.GLOBAL));
    if (globalsDeclaredTypes == null) {
      return null;
    }
    int index = binding.getIndex();
    return switch (globalsDeclaredTypes[index]) {
      case null -> null;
      case StarlarkType type -> type;
      case Resolver.Function rfn -> getType(rfn);
      default ->
          throw new IllegalStateException(
              "Unexpected declared type entry: " + globalsDeclaredTypes[index]);
    };
  }

  /**
   * Returns the type of the binding (whether declared or inferred). Null indicates no static type
   * information (during type checking, null means no static type information *yet*).
   */
  @Nullable
  public StarlarkType getType(Resolver.Binding binding) {
    int id = binding.getBindingId();
    return switch (bindingTypes[id]) {
      case null -> null;
      case StarlarkType type -> type;
      case Resolver.Function rfn -> getType(rfn);
      default ->
          throw new IllegalStateException("Unexpected binding type entry: " + bindingTypes[id]);
    };
  }

  /**
   * Returns the type of the function, or null for the program's toplevel. After type tagging is
   * complete, expected to be non-null for non-toplevel functions.
   */
  @Nullable
  public Types.GeneralCallableType getType(Resolver.Function function) {
    if (function.isToplevel()) {
      return null;
    }
    Types.GeneralCallableType annotated = getAnnotatedFunctionType(function);
    return annotated != null ? annotated : Types.allAnyFunction(function);
  }

  /** Returns the return type of a function. */
  public StarlarkType getReturnType(Resolver.Function function) {
    if (function.isToplevel()) {
      return Types.ANY;
    }
    Types.GeneralCallableType annotated = getAnnotatedFunctionType(function);
    return annotated != null ? annotated.getReturnType() : Types.ANY;
  }

  /**
   * Like {@link #getType(Resolver.Binding)}, but caches the type (and therefore, mutates this
   * {@link TypeTable}) if it was computed on demand and not yet cached.
   *
   * <p>Intended to be used by {@link TypeChecker}. Do not call this method after the type table has
   * been stored in skyframe.
   */
  @Nullable
  StarlarkType getAndCacheType(Resolver.Binding binding) {
    int id = binding.getBindingId();
    return switch (bindingTypes[id]) {
      case null -> null;
      case StarlarkType type -> type;
      case Resolver.Function rfn -> {
        checkState(getAnnotatedFunctionType(rfn) == null);
        StarlarkType type = getType(rfn);
        if (type != null) {
          // Cache the type to avoid recomputing Types.allAnyFunction()
          bindingTypes[id] = type;
        }
        yield type;
      }
      default ->
          throw new IllegalStateException("Unexpected binding type entry: " + bindingTypes[id]);
    };
  }

  /**
   * Returns the type of a resolved function whose signature has type annotations, or null if the
   * signature has no type annotations (or, during type tagging, if the function's type was not set
   * yet).
   */
  @Nullable
  Types.GeneralCallableType getAnnotatedFunctionType(Resolver.Function function) {
    return annotatedFunctionTypes == null ? null : annotatedFunctionTypes[function.getFunctionId()];
  }

  /**
   * After type tagging has been performed, returns true if the non-lambda function with this
   * sequence number is considered to use static typing syntax - in other words, type annotations or
   * {@code cast} expressions. Specifically:
   *
   * <ul>
   *   <li>For an ordinary function, returns true if the function's declaration or body (including
   *       any nested lambdas, but <em>not</em> including any ordinary nested {@code def} functions)
   *       uses type syntax.
   *   <li>For a file's toplevel function, returns true if any part of the file uses type syntax.
   *   <li>For a lambda, this bit is never set; callers should instead check {@link #usesTypeSyntax}
   *       for the most proximate enclosing def or toplevel.
   * </ul>
   */
  public boolean usesTypeSyntax(Resolver.Function function) {
    return functionsUsingTypeSyntax[function.getFunctionId()];
  }

  /**
   * After type tagging has been performed, returns true if the file's toplevel function is
   * considered to use static typing syntax (which is true iff any part of the file uses type
   * syntax).
   */
  boolean toplevelUsesTypeSyntax() {
    if (functionsUsingTypeSyntax.length > 0) {
      return functionsUsingTypeSyntax[functionsUsingTypeSyntax.length - 1];
    }
    return false;
  }

  /** Marks the function with the given sequence number as using type syntax. */
  void setUsesTypeSyntax(Resolver.Function function) {
    functionsUsingTypeSyntax[function.getFunctionId()] = true;
  }

  /**
   * Returns an array of function types indexed by their {@link Resolver.Function#getFunctionId}.
   * The size of the array may be smaller than the number of functions in the file; any function
   * whose ID is not represented in the array, or whose entry in the array is null, should be
   * considered to have an unannotated signature (see {@link Types#allAnyFunction}).
   *
   * <p>Can only be called after type tagging has finished.
   */
  public Types.CallableType[] getAnnotatedFunctionTypes() {
    if (annotatedFunctionTypes == null || annotatedFunctionTypes.length == 0) {
      return EMPTY_CALLABLE_ARRAY;
    }
    return Arrays.copyOfRange(
        annotatedFunctionTypes,
        0,
        // Always exclude <toplevel> at index length - 1; it cannot possibly be annotated.
        annotatedFunctionTypes.length - 1,
        // Types.GeneralCallableType is package-private
        Types.CallableType[].class);
  }
}
