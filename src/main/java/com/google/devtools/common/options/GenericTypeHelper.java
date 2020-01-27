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
package com.google.devtools.common.options;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.primitives.Primitives;
import com.google.common.reflect.TypeToken;

import java.lang.reflect.Method;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.lang.reflect.TypeVariable;

/**
 * A helper class for {@link OptionsParserImpl} to help checking the return type
 * of a {@link Converter} against the type of a field or the element type of a
 * list.
 *
 * <p>This class has to go through considerable contortion to get the correct result
 * from the Java reflection system, unfortunately. If the generic reflection part
 * had been better designed, some of this would not be necessary.
 */
class GenericTypeHelper {

  /**
   * Returns the raw type of t, if t is either a raw or parameterized type.
   * Otherwise, this method throws an {@link AssertionError}.
   */
  @VisibleForTesting
  static Class<?> getRawType(Type t) {
    if (t instanceof Class<?>) {
      return (Class<?>) t;
    } else if (t instanceof ParameterizedType) {
      return (Class<?>) ((ParameterizedType) t).getRawType();
    } else {
      throw new AssertionError("A known concrete type is not concrete");
    }
  }

  /**
   * If type is a parameterized type, searches the given type variable in the list
   * of declared type variables, and then returns the corresponding actual type.
   * Returns null if the type variable is not defined by type.
   */
  private static Type matchTypeVariable(Type type, TypeVariable<?> variable) {
    if (type instanceof ParameterizedType) {
      Class<?> rawInterfaceType = getRawType(type);
      TypeVariable<?>[] typeParameters = rawInterfaceType.getTypeParameters();
      for (int i = 0; i < typeParameters.length; i++) {
        if (variable.equals(typeParameters[i])) {
          return ((ParameterizedType) type).getActualTypeArguments()[i];
        }
      }
    }
    return null;
  }

  /**
   * Resolves the return type of a method, in particular if the generic return
   * type ({@link Method#getGenericReturnType()}) is a type variable
   * ({@link TypeVariable}), by checking all super-classes and directly
   * implemented interfaces.
   *
   * <p>The method m must be defined by the given type or by its raw class type.
   *
   * @throws AssertionError if the generic return type could not be resolved
   */
  // TODO(bazel-team): also check enclosing classes and indirectly implemented
  // interfaces, which can also contribute type variables. This doesn't happen
  // in the existing use cases.
  public static Type getActualReturnType(Type type, Method method) {
    Type returnType = method.getGenericReturnType();
    if (returnType instanceof Class<?>) {
      return returnType;
    } else if (returnType instanceof ParameterizedType) {
      return returnType;
    } else if (returnType instanceof TypeVariable<?>) {
      TypeVariable<?> variable = (TypeVariable<?>) returnType;
      while (type != null) {
        Type candidate = matchTypeVariable(type, variable);
        if (candidate != null) {
          return candidate;
        }

        Class<?> rawType = getRawType(type);
        for (Type interfaceType : rawType.getGenericInterfaces()) {
          candidate = matchTypeVariable(interfaceType, variable);
          if (candidate != null) {
            return candidate;
          }
        }

        type = rawType.getGenericSuperclass();
      }
    }
    throw new AssertionError("The type " + returnType
        + " is not a Class, ParameterizedType, or TypeVariable");
  }

  /**
   * Determines if a value of a particular type (from) is assignable to a field of
   * a particular type (to). Also allows assigning wrapper types to primitive
   * types.
   *
   * <p>The checks done here should be identical to the checks done by
   * {@link java.lang.reflect.Field#set}. I.e., if this method returns true, a
   * subsequent call to {@link java.lang.reflect.Field#set} should succeed.
   */
  public static boolean isAssignableFrom(Type to, Type from) {
    if (to instanceof Class<?>) {
      Class<?> toClass = (Class<?>) to;
      if (toClass.isPrimitive()) {
        return Primitives.wrap(toClass).equals(from);
      }
    }
    return TypeToken.of(to).isSupertypeOf(from);
  }

  private GenericTypeHelper() {
    // Prevents Java from creating a public constructor.
  }
}
