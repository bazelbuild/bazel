// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.syntax.compiler;

import net.bytebuddy.description.field.FieldDescription;
import net.bytebuddy.description.method.MethodDescription;

import java.util.Arrays;

/**
 * Utilities for reflective access of declared methods.
 */
public class ReflectionUtils {

  /**
   * Get a Byte Buddy {@link MethodDescription} for a constructor of a class.
   *
   * @throws Error when the constructor cannot be found via reflection
   */
  public static MethodDescription.ForLoadedConstructor getConstructor(
      Class<?> clazz, Class<?>... parameterTypes) {
    try {
      return new MethodDescription.ForLoadedConstructor(clazz.getConstructor(parameterTypes));
    } catch (NoSuchMethodException e) {
      throw new Error(
          String.format(
              "Error when reflectively getting a constructor with parameter"
                  + " types %s from class %s",
              Arrays.toString(parameterTypes),
              clazz),
          e);
    }
  }

  /**
   * Get a Byte Buddy {@link MethodDescription} for a method from a class.
   *
   * @throws Error when the method cannot be found via reflection
   */
  public static MethodDescription.ForLoadedMethod getMethod(
      Class<?> clazz, String name, Class<?>... parameterTypes) {
    try {
      return new MethodDescription.ForLoadedMethod(clazz.getMethod(name, parameterTypes));
    } catch (NoSuchMethodException e) {
      throw new Error(
          String.format(
              "Error when reflectively getting method %s with parameter types %s from class %s",
              name,
              Arrays.toString(parameterTypes),
              clazz),
          e);
    }
  }

  /**
   * Get a Byte Buddy {@link FieldDescription} for a field of a class.
   *
   * @throws Error when the field cannot be found via reflection
   */
  public static FieldDescription getField(Class<?> clazz, String name) {
    try {
      return new FieldDescription.ForLoadedField(clazz.getField(name));
    } catch (NoSuchFieldException e) {
      throw new RuntimeException(
          String.format("Error when reflectively getting field %s from class %s", name, clazz), e);
    }
  }
}
