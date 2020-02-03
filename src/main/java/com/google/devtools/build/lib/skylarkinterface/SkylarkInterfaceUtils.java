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

package com.google.devtools.build.lib.skylarkinterface;

import java.lang.annotation.Annotation;
import java.lang.reflect.Method;
import javax.annotation.Nullable;

/**
 * Helpers for accessing Skylark interface annotations.
 */
public class SkylarkInterfaceUtils {

  /**
   * Returns the more specific class of two classes. Class x is more specific than class y if x is
   * assignable to y. For example, of Integer.class and Object.class, Integer.class is more
   * specific.
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
      // @SkylarkModule annotations (and are thus considered "skylark types"). If B is not a
      // subclass of C (nor vice versa), then it's impossible to resolve whether A is of type
      // B or if A is of type C. It's both! The way to resolve this is usually to have A be its own
      // type (annotated with @SkylarkModule), and thus have the explicit type of A be semantically
      // "B and C".
      throw new IllegalArgumentException(
          String.format("Expected one of %s and %s to be a subclass of the other", x, y));
    }
  }

  /**
   * Searches a class or interface's class hierarchy for the given class annotation.
   *
   * <p>If the given class annotation appears multiple times within the class hierachy, this chooses
   * the annotation on the most-specified class in the hierarchy.
   *
   * @return the best-fit class that declares the annotation, or null if no class in the hierarchy
   *     declares it
   * @throws IllegalArgumentException if the most-specified class in the hierarchy having the
   *     annotation is not unique
   */
  @Nullable
  private static Class<?> findAnnotatedAncestor(
      Class<?> classObj, Class<? extends Annotation> annotation) {
    if (classObj.isAnnotationPresent(annotation)) {
      return classObj;
    }
    Class<?> bestCandidate = null;
    Class<?> superclass = classObj.getSuperclass();
    if (superclass != null) {
      Class<?> result = findAnnotatedAncestor(superclass, annotation);
      bestCandidate = moreSpecific(result, bestCandidate);
    }
    for (Class<?> interfaceObj : classObj.getInterfaces()) {
      Class<?> result = findAnnotatedAncestor(interfaceObj, annotation);
      bestCandidate = moreSpecific(result, bestCandidate);
    }
    return bestCandidate;
  }

  /**
   * Returns the {@link SkylarkModule} annotation for the given class, if it exists, and
   * null otherwise. The first annotation found will be returned, starting with {@code classObj}
   * and following its base classes and interfaces recursively.
   */
  @Nullable
  public static SkylarkModule getSkylarkModule(Class<?> classObj) {
    Class<?> cls = findAnnotatedAncestor(classObj, SkylarkModule.class);
    return cls == null ? null : cls.getAnnotation(SkylarkModule.class);
  }

  /**
   * Searches {@code classObj}'s class hierarchy and returns the first superclass or interface that
   * is annotated with {@link SkylarkModule} (including possibly {@code classObj} itself), or null
   * if none is found.
   */
  @Nullable
  public static Class<?> getParentWithSkylarkModule(Class<?> classObj) {
    return findAnnotatedAncestor(classObj, SkylarkModule.class);
  }

  /**
   * Searches {@code classObj}'s class hierarchy and for a superclass or interface that
   * is annotated with {@link SkylarkGlobalLibrary} (including possibly {@code classObj} itself),
   * and returns true if one is found.
   */
  public static boolean hasSkylarkGlobalLibrary(Class<?> classObj) {
    return findAnnotatedAncestor(classObj, SkylarkGlobalLibrary.class) != null;
  }

  /**
   * Returns the {@link SkylarkCallable} annotation for the given method, if it exists, and
   * null otherwise.
   *
   * <p>Note that the annotation may be defined on a supermethod, rather than directly on the given
   * method.
   *
   * <p>{@code classObj} is the class on which the given method is defined.
   */
  @Nullable
  public static SkylarkCallable getSkylarkCallable(Class<?> classObj, Method method) {
    SkylarkCallable callable = getCallableOnClassMatchingSignature(classObj, method);
    if (callable != null) {
      return callable;
    }
    if (classObj.getSuperclass() != null) {
      SkylarkCallable annotation = getSkylarkCallable(classObj.getSuperclass(), method);
      if (annotation != null) {
        return annotation;
      }
    }
    for (Class<?> interfaceObj : classObj.getInterfaces()) {
      SkylarkCallable annotation = getSkylarkCallable(interfaceObj, method);
      if (annotation != null) {
        return annotation;
      }
    }
    return null;
  }

  /**
   * Convenience version of {@code getAnnotationsFromParentClass(Class, Method)} that uses
   * the declaring class of the method.
   */
  @Nullable
  public static SkylarkCallable getSkylarkCallable(Method method) {
    return getSkylarkCallable(method.getDeclaringClass(), method);
  }

  /**
   * Returns the {@code SkylarkCallable} annotation corresponding to the given method of the given
   * class, or null if there is no such annotation.
   *
   * <p>This method checks assignability instead of exact matches for purposes of generics. If
   * Clazz has parameters BarT (extends BarInterface) and BazT (extends BazInterface), then
   * foo(BarT, BazT) should match if the given method signature is foo(BarImpl, BazImpl). The
   * signatures are in inexact match, but an "assignable" match.
   */
  @Nullable
  private static SkylarkCallable getCallableOnClassMatchingSignature(
      Class<?> classObj, Method signatureToMatch) {
    // TODO(b/79877079): This method validates several invariants of @SkylarkCallable. These
    // invariants should be verified in annotation processor or in test, and left out of this
    // method.
    Method[] methods = classObj.getDeclaredMethods();
    Class<?>[] paramsToMatch = signatureToMatch.getParameterTypes();

    SkylarkCallable callable = null;

    for (Method method : methods) {
      if (signatureToMatch.getName().equals(method.getName())
          && method.isAnnotationPresent(SkylarkCallable.class)) {
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
          callable = method.getAnnotation(SkylarkCallable.class);
        } else {
          throw new IllegalStateException(
              String.format(
                  "Class %s has multiple overloaded methods named '%s' annotated "
                      + "with @SkylarkCallable",
                  classObj, signatureToMatch.getName()));
        }
      }
    }
    return callable;
  }
}
