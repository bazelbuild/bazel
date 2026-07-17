// Copyright 2016 The Bazel Authors. All rights reserved.
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

package net.starlark.java.annot;

import java.lang.reflect.Method;
import javax.annotation.Nullable;

/** Utility functions for Starlark annotations. */
public final class StarlarkAnnotations {

  /**
   * Returns the more specific class of two classes. Class x is more specific than class y if x is
   * assignable to y. For example, of String.class and Object.class, String.class is more specific.
   *
   * <p>If either class is null, returns the other class.
   *
   * <p>If the classes are identical, returns the class.
   *
   * @throws IllegalArgumentException if neither class is assignable to the other
   */
  private static Class<?> moreSpecific(Class<?> x, Class<?> y) {
    if (x == null) {
      return y;
    } else if (y == null) {
      return x;
    } else if (x.isAssignableFrom(y)) {
      return y;
    } else if (y.isAssignableFrom(x)) {
      return x;
    } else {
      // If this exception occurs, it indicates the following error scenario:
      //
      // Suppose class A is a subclass of both B and C, where B and C are annotated with
      // @StarlarkBuiltin annotations (and are thus considered "Starlark types"). If B is not a
      // subclass of C (nor vice versa), then it's impossible to resolve whether A is of type
      // B or if A is of type C. It's both! The way to resolve this is usually to have A be its own
      // type (annotated with @StarlarkBuiltin), and thus have the explicit type of A be
      // semantically "B and C".
      throw new IllegalArgumentException(
          String.format("Expected one of %s and %s to be a subclass of the other", x, y));
    }
  }

  /**
   * Information extracted by walking a class's ancestors' {@link StarlarkBuiltin} annotations.
   *
   * @param starlarkBuiltinAncestor the most-specified ancestor annotated with {@link
   *     StarlarkBuiltin}. (It is guaranteed that if two ancestors both define the annotation, one
   *     of them is a subtype of the other.)
   * @param assignableToStructType whether any {@code StarlarkBuiltin}-annotated ancestor has set
   *     {@link StarlarkBuiltin#isStructType} to true.
   */
  private record ClassInfo(Class<?> starlarkBuiltinAncestor, boolean assignableToStructType) {}

  // A map from a class to its ClassInfo.
  private static final ClassValue<ClassInfo> classInfos =
      new ClassValue<ClassInfo>() {
        @Override
        protected ClassInfo computeValue(Class<?> type) {
          return buildClassInfo(type);
        }
      };

  /**
   * Searches a class or interface's class hierarchy for {@link StarlarkBuiltin} annotations.
   *
   * <p>If the {@link StarlarkBuiltin} annotation appears multiple times within the class hierarchy,
   * the returned {@link ClassInfo} will contain the annotation of the most-specified class in the
   * hierarchy; and {@link ClassInfo#assignableToStructType} will be true if any of the annotations
   * have {@code assignableToStructType=true}.
   *
   * @return a {@link ClassInfo} with {@link ClassInfo#starlarkBuiltinAncestor} set to the best-fit
   *     class that declares a {@link StarlarkBuiltin} annotation; returns null if no class in the
   *     hierarchy declares the annotation.
   * @throws IllegalArgumentException if the most-specified class in the hierarchy having the
   *     annotation is not unique
   */
  @Nullable
  private static ClassInfo buildClassInfo(Class<?> classObj) {
    @Nullable Class<?> bestCandidate = null;
    boolean assignable = false;

    if (classObj.isAnnotationPresent(StarlarkBuiltin.class)) {
      bestCandidate = classObj;
      assignable = classObj.getAnnotation(StarlarkBuiltin.class).isStructType();
    }

    // Note that moreSpecific imposes a linear relationship between competing ancestors, so
    // `bestCandidate` and `assignable` are consistent.
    @Nullable Class<?> superclass = classObj.getSuperclass();
    if (superclass != null) {
      @Nullable ClassInfo result = classInfos.get(superclass);
      if (result != null) {
        bestCandidate = moreSpecific(result.starlarkBuiltinAncestor(), bestCandidate);
        assignable |= result.assignableToStructType();
      }
    }
    for (Class<?> interfaceObj : classObj.getInterfaces()) {
      @Nullable ClassInfo result = classInfos.get(interfaceObj);
      if (result != null) {
        bestCandidate = moreSpecific(result.starlarkBuiltinAncestor(), bestCandidate);
        assignable |= result.assignableToStructType();
      }
    }
    return bestCandidate != null ? new ClassInfo(bestCandidate, assignable) : null;
  }

  /**
   * Returns the {@link StarlarkBuiltin} annotation for the given class or interface. If the
   * annotation is not found directly, its ancestor classes and interfaces are searched, and in case
   * multiple annotations are found, the one on the most derived class or interface is used.
   *
   * <p>Returns null if no annotation is found in the class hierarchy.
   */
  @Nullable
  public static StarlarkBuiltin getStarlarkBuiltin(Class<?> classObj) {
    @Nullable ClassInfo classInfo = classInfos.get(classObj);
    return classInfo != null
        ? classInfo.starlarkBuiltinAncestor().getAnnotation(StarlarkBuiltin.class)
        : null;
  }

  /**
   * Returns true if the given classObj or any of its interfaces or supertypes has {@code
   * assignableToStructType} marked in its {@link StarlarkBuiltin} annotation.
   */
  public static boolean isAssignableToStructType(Class<?> classObj) {
    @Nullable ClassInfo classInfo = classInfos.get(classObj);
    return classInfo != null && classInfo.assignableToStructType();
  }

  /**
   * Searches {@code classObj}'s class hierarchy and returns the first superclass or interface that
   * is annotated with {@link StarlarkBuiltin} (including possibly {@code classObj} itself), or null
   * if none is found.
   */
  @Nullable
  public static Class<?> getParentWithStarlarkBuiltin(Class<?> classObj) {
    @Nullable ClassInfo classInfo = classInfos.get(classObj);
    return classInfo != null ? classInfo.starlarkBuiltinAncestor() : null;
  }

  /**
   * Returns the {@link StarlarkMethod} annotation for the given method, if it exists, and null
   * otherwise.
   *
   * <p>Note that the annotation may be defined on a supermethod, rather than directly on the given
   * method.
   *
   * <p>{@code classObj} is the class on which the given method is defined.
   */
  @Nullable
  public static StarlarkMethod getStarlarkMethod(Class<?> classObj, Method method) {
    StarlarkMethod callable = getAnnotationOnClassMatchingSignature(classObj, method);
    if (callable != null) {
      return callable;
    }
    if (classObj.getSuperclass() != null) {
      StarlarkMethod annotation = getStarlarkMethod(classObj.getSuperclass(), method);
      if (annotation != null) {
        return annotation;
      }
    }
    for (Class<?> interfaceObj : classObj.getInterfaces()) {
      StarlarkMethod annotation = getStarlarkMethod(interfaceObj, method);
      if (annotation != null) {
        return annotation;
      }
    }
    return null;
  }

  /**
   * Convenience version of {@code getAnnotationsFromParentClass(Class, Method)} that uses the
   * declaring class of the method.
   */
  @Nullable
  public static StarlarkMethod getStarlarkMethod(Method method) {
    return getStarlarkMethod(method.getDeclaringClass(), method);
  }

  /**
   * Returns the {@code StarlarkMethod} annotation corresponding to the given method of the given
   * class, or null if there is no such annotation.
   *
   * <p>This method checks assignability instead of exact matches for purposes of generics. If Clazz
   * has parameters BarT (extends BarInterface) and BazT (extends BazInterface), then foo(BarT,
   * BazT) should match if the given method signature is foo(BarImpl, BazImpl). The signatures are
   * in inexact match, but an "assignable" match.
   */
  @Nullable
  private static StarlarkMethod getAnnotationOnClassMatchingSignature(
      Class<?> classObj, Method signatureToMatch) {
    // TODO(b/79877079): This method validates several invariants of @StarlarkMethod. These
    // invariants should be verified in annotation processor or in test, and left out of this
    // method.
    Method[] methods = classObj.getDeclaredMethods();
    Class<?>[] paramsToMatch = signatureToMatch.getParameterTypes();

    StarlarkMethod callable = null;

    for (Method method : methods) {
      if (signatureToMatch.getName().equals(method.getName())
          && method.isAnnotationPresent(StarlarkMethod.class)) {
        Class<?>[] paramTypes = method.getParameterTypes();

        if (paramTypes.length == paramsToMatch.length) {
          for (int i = 0; i < paramTypes.length; i++) {
            // This verifies assignability of the method signature to ensure this is not a
            // coincidental overload. We verify assignability instead of matching exact parameter
            // classes in order to match generic methods.
            if (!paramTypes[i].isAssignableFrom(paramsToMatch[i])) {
              throw new IllegalStateException(
                  String.format(
                      "Class %s has an incompatible overload of annotated method %s declared by %s",
                      classObj, signatureToMatch.getName(), signatureToMatch.getDeclaringClass()));
            }
          }
        }
        if (callable == null) {
          callable = method.getAnnotation(StarlarkMethod.class);
        } else {
          throw new IllegalStateException(
              String.format(
                  "Class %s has multiple overloaded methods named '%s' annotated "
                      + "with @StarlarkMethod",
                  classObj, signatureToMatch.getName()));
        }
      }
    }
    return callable;
  }

  private StarlarkAnnotations() {}
}
