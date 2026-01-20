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

/**
 * A factory for creating {@link StarlarkType}s, parameterized by zero or more type arguments.
 *
 * <p>Conceptually, a type constructor corresponds to what the user informally thinks of as "a
 * type": a program symbol, like {@code list}, that can appear within a type expression. The usage
 * of a constructor in a type expression yields an actual type, like {@code list[int]}. In the case
 * of basic types like {@code None} that are not parameterized, there is both a trivial nullary type
 * constructor and an underlying singleton type, where the constructor just wraps the underlying
 * type.
 */
public interface TypeConstructor {

  /** Exception thrown when a {@link TypeConstructor} is invoked with invalid arguments. */
  class Failure extends Exception {
    Failure(String message) {
      super(message);
    }
  }

  /**
   * Returns the result of applying this constructor to the given type arguments
   *
   * @throws Failure if the usage of this constructor is invalid (typically due to a mismatch in the
   *     number of arguments)
   */
  StarlarkType invoke(ImmutableList<?> argsTuple) throws Failure;
}
