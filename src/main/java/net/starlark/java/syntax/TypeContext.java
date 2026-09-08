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

import com.google.common.collect.ImmutableList;
import javax.annotation.Nullable;

/**
 * A context for obtaining more detailed information about Starlark types.
 *
 * <p>This is used to inject type information from the {@code eval/} package into the {@code
 * syntax/} package, e.g. the method APIs of {@link StarlarkList}.
 */
public interface TypeContext {
  
  /** Returns the type of the given field of a {@code str} type, or null if no such field exists. */
  @Nullable
  StarlarkType getStrFieldType(String name);

  /**
   * Returns the type of the given field of a {@code list[T]} type, or null if no such field exists.
   */
  @Nullable
  StarlarkType getListFieldType(String name);

  /**
   * Returns the type of the given field of a {@code dict[K, V]} type, or null if no such field
   * exists.
   */
  @Nullable
  StarlarkType getDictFieldType(String name);

  /**
   * Returns the type of the given field of a {@code set[T]} type, or null if no such field exists.
   */
  @Nullable
  StarlarkType getSetFieldType(String name);

  /**
   * Returns the type of the given field of a {@link net.starlark.java.annot.StarlarkBuiltin}
   * annotated class (or a subclass of one), or null if the class is not a @StarlarkBuiltin or has
   * no such field.
   */
  @Nullable
  default StarlarkType getStarlarkBuiltinFieldType(Class<?> clazz, String fieldName) {
    return null;
  }

  /**
   * Returns the supertypes of the auto-generated Starlark type associated with the given {@link
   * net.starlark.java.annot.StarlarkBuiltin} annotated class (or a subclass of one), or null if no
   * such auto-generated type exists.
   */
  @Nullable
  default ImmutableList<StarlarkType> getStarlarkBuiltinAutoTypeSupertypes(Class<?> clazz) {
    return null;
  }

  /**
   * Returns the value type of a {@link Resolver.Scope#PREDECLARED} symbol, or null if there is no
   * such symbol.
   */
  @Nullable
  StarlarkType getPredeclaredSymbolType(String name);

  /**
   * Returns the value type of a {@link Resolver.Scope#UNIVERSAL} symbol, or null if there is no
   * such symbol.
   */
  @Nullable
  StarlarkType getUniversalSymbolType(String name);
}
