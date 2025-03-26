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

package net.starlark.java.eval;

import com.google.common.collect.ImmutableMap;

/**
 * Definitions of types.
 *
 * <p><code>
 *   t ::= None | bool | int | float | str
 * </code>
 */
final class Types {
  // Primitive types
  static final StarlarkType NONE = new None();
  static final StarlarkType BOOL = new Bool();
  static final StarlarkType INT = new Int();
  static final StarlarkType FLOAT = new FloatType();
  static final StarlarkType STR = new Str();

  private Types() {} // uninstantiable

  static final ImmutableMap<String, Object> TYPE_UNIVERSE = makeTypeUniverse();

  private static ImmutableMap<String, Object> makeTypeUniverse() {
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    env //
        .put("None", NONE)
        .put("bool", BOOL)
        .put("int", INT)
        .put("float", FLOAT)
        .put("str", STR);
    return env.buildOrThrow();
  }

  private static final class None extends StarlarkType {
    @Override
    public String toString() {
      return "None";
    }
  }

  private static final class Bool extends StarlarkType {
    @Override
    public String toString() {
      return "bool";
    }
  }

  private static final class Int extends StarlarkType {
    @Override
    public String toString() {
      return "int";
    }
  }

  private static final class FloatType extends StarlarkType { // Float clashes with java.lang.Float
    @Override
    public String toString() {
      return "float";
    }
  }

  private static final class Str extends StarlarkType {
    @Override
    public String toString() {
      return "str";
    }
  }
}
