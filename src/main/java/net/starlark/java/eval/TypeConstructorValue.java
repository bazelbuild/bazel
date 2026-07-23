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

package net.starlark.java.eval;

import com.google.common.collect.ImmutableList;
import net.starlark.java.syntax.StarlarkType;
import net.starlark.java.syntax.TypeConstructor;
import net.starlark.java.syntax.TypeContext;

/**
 * A {@link StarlarkValue} wrapping a {@link TypeConstructor}. This is used as the runtime value of
 * certain symbols that represent Starlark-defined types, such as type aliases.
 *
 * <p>Not all Starlark values that can be used as type constructors are instances of this class. For
 * example, builtin symbols like {@code list} and {@code dict} are instead instances of {@link
 * BuiltinFunction.BuiltinTypeFunction}.
 */
public sealed class TypeConstructorValue implements StarlarkValue, TypeConstructor
    permits TypeConstructorValue.AllowingNullary {
  public static final StarlarkType TYPE = new Type();

  private final TypeConstructor typeConstructor;

  public static TypeConstructorValue of(TypeConstructor constructor) {
    if (constructor instanceof TypeConstructor.AllowingNullary allowingNullary) {
      return new AllowingNullary(allowingNullary, allowingNullary.createStarlarkType());
    } else {
      return new TypeConstructorValue(constructor);
    }
  }

  private TypeConstructorValue(TypeConstructor typeConstructor) {
    this.typeConstructor = typeConstructor;
  }

  @Override
  public void repr(Printer printer, StarlarkSemantics semantics) {
    printer.append("<Type ").append(typeConstructor.toString()).append(">");
  }

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public StarlarkType getStarlarkType(StarlarkSemantics semantics) {
    return TYPE;
  }

  @Override
  public StarlarkType createStarlarkType(ImmutableList<TypeConstructor.Term> argsTuple)
      throws TypeConstructor.Failure {
    return typeConstructor.createStarlarkType(argsTuple);
  }

  /**
   * A {@link TypeConstructorValue} whose {@link TypeConstructor} may be invoked without type
   * arguments and may be used in {@code isinstance()} checks.
   */
  // TODO: b/536902188 - Make private once we no longer have to worry about OpenJDK 21 in the bazel
  // bootstrap test (https://bugs.openjdk.org/browse/JDK-8284011).
  static final class AllowingNullary extends TypeConstructorValue implements StarlarkTypeValue {
    private final StarlarkType nullaryType;

    private AllowingNullary(TypeConstructor constructor, StarlarkType nullaryType) {
      super(constructor);
      this.nullaryType = nullaryType;
    }

    @Override
    public boolean hasInstance(Object value, StarlarkSemantics semantics, TypeContext typeContext) {
      return StarlarkType.assignableFrom(
          nullaryType, Starlark.getStarlarkType(value, semantics), typeContext);
    }
  }

  /** The type of {@link StarlarkTypeConstructorValue}-s. */
  private static final class Type extends StarlarkType {
    // Singleton.
    private Type() {}

    @Override
    public String toString() {
      return "Type";
    }

    @Override
    public int hashCode() {
      return Type.class.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof Type;
    }
  }
}
