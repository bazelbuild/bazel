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

  private static final class ClassWithAnnotation<T extends Annotation> {
    final Class<?> klass;
    final T annotation;

    ClassWithAnnotation(Class<?> klass, T annotation) {
      this.klass = klass;
      this.annotation = annotation;
    }
  }

  /**
   * Returns the more specific class of two classes. Class x is more specific than class y
   * if x is assignable to y. For example, of Integer.class and Object.class, Integer.class is more
   * specific.
   *
   * <p>If either class is null, returns the other class.</p>
   *
   * <p>If the classes are identical, returns the class.</p>
   *
   * @throws IllegalArgumentException if neither class is assignable to the other
   */
  private static <T extends Annotation> ClassWithAnnotation<T> moreSpecificClass(
      ClassWithAnnotation<T> x, ClassWithAnnotation<T> y) {
    if (x == null) {
      return y;
    } else if (y == null) {
      return x;
    }
    Class<?> xClass = x.klass;
    Class<?> yClass = y.klass;
    if (xClass.isAssignableFrom(yClass)) {
      return y;
    } else if (yClass.isAssignableFrom(xClass)) {
      return x;
    } else {
      // If this exception occurs, it indicates the following error scenario:
      //
      // Suppose class A is a subclass of both B and C, where B and C are annotated with
      // @SkylarkModule annotations (and are thus considered "skylark types"). If B is not a
      // subclass of C (nor visa versa), then it's impossible to resolve whether A is of type
      // B or if A is of type C. It's both! The way to resolve this is usually to have A be its own
      // type (annotated with @SkylarkModule), and thus have the explicit type of A be semantically
      // "B and C".
      throw new IllegalArgumentException(String.format(
          "Expected one of %s and %s to be a subclass of the other",
          xClass, yClass));
    }
  }

  /**
   * Searches a class or interface's class hierarchy for the given class annotation.
   *
   * <p>If the given class annotation appears multiple times within the class hierachy, this
   * chooses the annotation on the most-specified class in the hierarchy.</p>
   *
   * @return a {@link ClassWithAnnotation} containing the best-fit annotation and the class
   *     it was declared on
   */
  @Nullable
  private static <T extends Annotation> ClassWithAnnotation<T> searchForClassAnnotation(
      Class<?> classObj,
      Class<T> annotationClass) {
    if (classObj.isAnnotationPresent(annotationClass)) {
      return new ClassWithAnnotation<T>(classObj, classObj.getAnnotation(annotationClass));
    }
    ClassWithAnnotation<T> bestCandidate = null;

    Class<?> superclass = classObj.getSuperclass();
    if (superclass != null) {
      ClassWithAnnotation<T> result = searchForClassAnnotation(superclass, annotationClass);
      bestCandidate = moreSpecificClass(result, bestCandidate);
    }
    for (Class<?> interfaceObj : classObj.getInterfaces()) {
      ClassWithAnnotation<T> result = searchForClassAnnotation(interfaceObj, annotationClass);
      bestCandidate = moreSpecificClass(result, bestCandidate);
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
    ClassWithAnnotation<SkylarkModule> result =
        searchForClassAnnotation(classObj, SkylarkModule.class);
    return result == null ? null : result.annotation;
  }

  /**
   * Searches {@code classObj}'s class hierarchy and returns the first superclass or interface that
   * is annotated with {@link SkylarkModule} (including possibly {@code classObj} itself), or null
   * if none is found.
   */
  @Nullable
  public static Class<?> getParentWithSkylarkModule(Class<?> classObj) {
    ClassWithAnnotation<SkylarkModule> result =
        searchForClassAnnotation(classObj, SkylarkModule.class);
    return result == null ? null : result.klass;
  }

  /**
   * Searches {@code classObj}'s class hierarchy and for a superclass or interface that
   * is annotated with {@link SkylarkGlobalLibrary} (including possibly {@code classObj} itself),
   * and returns true if one is found.
   */
  public static boolean hasSkylarkGlobalLibrary(Class<?> classObj) {
    ClassWithAnnotation<SkylarkGlobalLibrary> result =
        searchForClassAnnotation(classObj, SkylarkGlobalLibrary.class);
    return result != null;
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
