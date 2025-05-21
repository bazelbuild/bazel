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

package net.starlark.java.types;

import static java.util.stream.Collectors.joining;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import javax.annotation.Nullable;

/**
 * Definitions of types.
 *
 * <p><code>
 *   t ::= None | bool | int | float | str | object
 * </code>
 */
public final class Types {
  // Internal type used as a guard for a missing type annotations (for now).
  public static final StarlarkType ANY = new Any();

  // Primitive types
  public static final StarlarkType NONE = new None();
  public static final StarlarkType BOOL = new Bool();
  public static final StarlarkType INT = new Int();
  public static final StarlarkType FLOAT = new FloatType();
  public static final StarlarkType STR = new Str();
  public static final StarlarkType OBJECT = new ObjectType();

  // A frequently used function without parameters, that returns Any.
  public static final CallableType NO_PARAMS_CALLABLE =
      callable(ImmutableList.of(), ImmutableList.of(), 0, ImmutableSet.of(), null, null, ANY);

  private Types() {} // uninstantiable

  public static final ImmutableMap<String, Object> TYPE_UNIVERSE = makeTypeUniverse();

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

  // hashCode and equals implementation is a workaround for serialization code that may duplicate
  // otherwise singletons
  private static final class Any extends StarlarkType {
    @Override
    public String toString() {
      return "Any";
    }

    @Override
    public int hashCode() {
      return Any.class.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof Any;
    }
  }

  private static final class ObjectType extends StarlarkType {
    @Override
    public String toString() {
      return "object";
    }

    @Override
    public int hashCode() {
      return ObjectType.class.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof ObjectType;
    }
  }

  private static final class None extends StarlarkType {
    @Override
    public String toString() {
      return "None";
    }

    @Override
    public int hashCode() {
      return None.class.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof None;
    }
  }

  private static final class Bool extends StarlarkType {
    @Override
    public String toString() {
      return "bool";
    }

    @Override
    public int hashCode() {
      return Bool.class.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof Bool;
    }
  }

  private static final class Int extends StarlarkType {
    @Override
    public String toString() {
      return "int";
    }

    @Override
    public int hashCode() {
      return Int.class.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof Int;
    }
  }

  private static final class FloatType extends StarlarkType { // Float clashes with java.lang.Float
    @Override
    public String toString() {
      return "float";
    }

    @Override
    public int hashCode() {
      return FloatType.class.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof Float;
    }
  }

  private static final class Str extends StarlarkType {
    @Override
    public String toString() {
      return "str";
    }

    @Override
    public int hashCode() {
      return Str.class.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      return obj instanceof Str;
    }
  }

  /** Construct a CallableType representing a Starlark Function */
  public static CallableType callable(
      ImmutableList<String> parameterNames,
      ImmutableList<StarlarkType> parameterTypes,
      int numPositionalParameters,
      ImmutableSet<String> mandatoryParams,
      @Nullable StarlarkType varargsType,
      @Nullable StarlarkType kwargsType,
      StarlarkType returns) {
    return new AutoValue_Types_GeneralCallableType(
        parameterNames,
        parameterTypes,
        numPositionalParameters,
        mandatoryParams,
        varargsType,
        kwargsType,
        returns);
  }

  /**
   * An interface for the general Starlark callable.
   *
   * <p>There are 3 flavours of parameters:
   *
   * <ul>
   *   <li>positional-only (can't be passed with a keyword),
   *   <li>ordinary (can be passed by position or with a keyword) and
   *   <li>keyword-only parameters.
   * </ul>
   *
   * The interface describes them as follows:
   *
   * <ul>
   *   <li>Their types are stored consecutively in <code>parameterTypes</code>.
   *   <li>The list <code>parameterNames</code> is shorter than <code>parameterTypes</code> by the
   *       count of positional-only parameters. (If there are k positional-only params, the (k+i)th
   *       param's name is stored in <code>parameterNames[i]</code>.)
   *   <li><code>numPositionalParameters</code> counts both positional-only and ordinary arguments.
   * </ul>
   *
   * <p>Special parameters {@code *args} and {@code **kwargs} are stored separately. If they are
   * absent, they are set to {@code null}.
   *
   * <p>Mandatory parameters (parameters without default values) are stored as a set.
   */
  public abstract static class CallableType extends StarlarkType {

    public abstract ImmutableList<String> getParameterNames();

    public abstract ImmutableList<StarlarkType> getParameterTypes();

    public abstract int getNumPositionalParameters();

    public abstract ImmutableSet<String> getMandatoryParameters();

    @Nullable
    public abstract StarlarkType getVarargsType();

    @Nullable
    public abstract StarlarkType getKwargsType();

    public abstract StarlarkType getReturnType();

    public StarlarkType getParameterTypeByPos(int i) {
      return getParameterTypes().get(i);
    }

    @Override
    public String toString() {
      // Approximate representation of the type - as much as Callable can do
      return "Callable[["
          + getParameterTypes().stream().map(StarlarkType::toString).collect(joining(", "))
          + "], "
          + getReturnType()
          + "]";
    }

    /** Returns a complete string representation of the type */
    public String toSignatureString() {
      ImmutableList.Builder<String> params = ImmutableList.builder();

      // unnamed positional parameters
      int typeIndex = 0;
      for (; typeIndex < getParameterTypes().size() - getParameterNames().size(); typeIndex++) {
        params.add(getParameterTypeByPos(typeIndex).toString());
      }

      if (typeIndex > 0) { // if there were positional-only parameters, we need to separate them
        params.add("/");
      }

      // named positional parameters
      int nameIndex = 0;
      for (; typeIndex < getNumPositionalParameters(); typeIndex++, nameIndex++) {
        String name = getParameterNames().get(nameIndex);
        StarlarkType type = getParameterTypeByPos(typeIndex);
        if (getMandatoryParameters().contains(name)) {
          params.add(name + ": " + type);
        } else {
          params.add(name + ": [" + type + "]");
        }
      }

      if (getVarargsType() != null) {
        params.add("*args: " + getVarargsType());
      } else if (typeIndex < getParameterTypes().size()) { // if there are going to be kwonly params
        params.add("*");
      }

      // keyword parameters
      for (; typeIndex < getParameterTypes().size(); typeIndex++, nameIndex++) {
        String name = getParameterNames().get(nameIndex);
        String type = getParameterTypeByPos(typeIndex).toString();
        if (getMandatoryParameters().contains(name)) {
          params.add(name + ": " + type);
        } else {
          params.add(name + ": [" + type + "]");
        }
      }

      if (getKwargsType() != null) {
        params.add("**kwargs: " + getKwargsType());
      }

      ImmutableList<String> paramList = params.build();
      return "(" + String.join(", ", paramList) + ") -> " + getReturnType();
    }
  }

  // About 0.1% memory regression may be removed by specializing GeneralCallableType for function
  // without positional-only parameter and by retrieving parameter names from StarlarkFunction
  @AutoValue
  abstract static class GeneralCallableType extends CallableType {}
}
