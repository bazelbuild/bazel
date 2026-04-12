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

import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/**
 * The static type information for a Starlark file.
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
  // Declared types of global bindings. Indexed by the binding's global scope index.
  private final StarlarkType[] globalsDeclaredTypes;
  // Types of bindings (regardless of scope). Indexed by binding's sequence number.
  private final StarlarkType[] bindingTypes;
  // Types of functions. Indexed by function's sequence number.
  private final Types.CallableType[] functionTypes;
  // Whether a function uses type syntax. Indexed by function's sequence number.
  private final boolean[] functionsUsingTypeSyntax;

  final List<SyntaxError> errors = new ArrayList<>();

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
    this.globalsDeclaredTypes = new StarlarkType[numGlobals];
    this.bindingTypes = new StarlarkType[numBindings];
    this.functionTypes = new Types.CallableType[numFunctions];
    this.functionsUsingTypeSyntax = new boolean[numFunctions];
  }

  /** Returns the list of errors recorded in the type table. */
  public ImmutableList<SyntaxError> errors() {
    return ImmutableList.copyOf(errors);
  }

  /** Returns true if no errors were recorded in the type table. */
  public boolean ok() {
    return errors.isEmpty();
  }

  /**
   * Sets the declared (annotated) type of the given binding. May be called more than once. Null is
   * treated is untyped / Any.
   */
  void setDeclaredType(Resolver.Binding binding, StarlarkType type) {
    bindingTypes[binding.getBindingId()] = type;
    if (binding.getScope().equals(Resolver.Scope.GLOBAL)) {
      globalsDeclaredTypes[binding.getIndex()] = type;
    }
  }

  /**
   * Sets the inferred type of the given binding. May be called more than once. Null is treated is
   * untyped / Any.
   */
  void setInferredType(Resolver.Binding binding, StarlarkType type) {
    bindingTypes[binding.getBindingId()] = type;
  }

  /**
   * Sets the type of the given function. May be called more than once. Null is treated is untyped /
   * Any.
   */
  void setType(Resolver.Function function, Types.CallableType type) {
    functionTypes[function.getFunctionId()] = type;
  }

  /** Returns the declared type of the global binding. Null indicates untyped / Any. */
  @Nullable
  public StarlarkType getGlobalDeclaredType(Resolver.Binding binding) {
    checkArgument(binding.getScope().equals(Resolver.Scope.GLOBAL));
    return globalsDeclaredTypes[binding.getIndex()];
  }

  /**
   * Returns the type of the binding (whether declared or inferred). Null indicates untyped / Any.
   */
  @Nullable
  public StarlarkType getType(Resolver.Binding binding) {
    return bindingTypes[binding.getBindingId()];
  }

  /**
   * Returns the type of the function, or null for the program's toplevel. After type tagging is
   * complete, expected to be non-null for non-toplevel functions.
   */
  @Nullable
  public Types.CallableType getType(Resolver.Function function) {
    return functionTypes[function.getFunctionId()];
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

  /** Marks the function with the given sequence number as using type syntax. */
  void setUsesTypeSyntax(Resolver.Function function) {
    functionsUsingTypeSyntax[function.getFunctionId()] = true;
  }
}
