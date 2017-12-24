// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.serialization.autocodec;

import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.TypeName;
import javax.annotation.processing.ProcessingEnvironment;
import javax.lang.model.type.DeclaredType;

/**
 * Generates serialize and deserialize code fragments for matching types.
 *
 * <p>All methods are logically static and take the {@link ProcessingEnvironment} as a parameter.
 */
interface Marshaller {
  static class Context {
    /** Builder for the method. */
    public final MethodSpec.Builder builder;
    /** Type of {@code name}. */
    public final DeclaredType type;
    /** Name of variable. */
    public final String name;
    /**
     * Recursion depth.
     *
     * <p>Recursion is used to traverse generic types.
     */
    public final int depth;

    Context(MethodSpec.Builder builder, DeclaredType type, String name) {
      this(builder, type, name, 0);
    }

    private Context(MethodSpec.Builder builder, DeclaredType type, String name, int depth) {
      this.builder = builder;
      this.type = type;
      this.name = name;
      this.depth = depth;
    }

    /** Returns a new context with a new type and name at the next recursion depth. */
    Context with(DeclaredType newType, String newName) {
      return new Context(builder, newType, newName, depth + 1);
    }

    TypeName getTypeName() {
      return TypeName.get(type);
    }

    /**
     * Returns a depth-qualified name.
     *
     * <p>This helps to avoid name collisions when recursion reuses symbol names.
     */
    String makeName(String name) {
      return name + depth;
    }
  };

  /** Returns true if `type` is handled by this. */
  boolean matches(DeclaredType type);

  /** Appends code statements to serialize a pre-declared variable. */
  void addSerializationCode(Context context);

  /** Appends code statements to initialize the pre-declared variable with deserialization. */
  void addDeserializationCode(Context context);
}
