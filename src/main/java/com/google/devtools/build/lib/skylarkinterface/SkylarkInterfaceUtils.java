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

  @Nullable
  private static <T extends Annotation> ClassWithAnnotation<T> searchForClassAnnotation(
      Class<?> classObj,
      Class<T> annotationClass) {
    if (classObj.isAnnotationPresent(annotationClass)) {
      return new ClassWithAnnotation<T>(classObj, classObj.getAnnotation(annotationClass));
    }
    Class<?> superclass = classObj.getSuperclass();
    if (superclass != null) {
      ClassWithAnnotation<T> result = searchForClassAnnotation(superclass, annotationClass);
      if (result != null) {
        return result;
      }
    }
    for (Class<?> interfaceObj : classObj.getInterfaces()) {
      ClassWithAnnotation<T> result = searchForClassAnnotation(interfaceObj, annotationClass);
      if (result != null) {
        return result;
      }
    }
    return null;
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
   * null otherwise. The first annotation of an overridden version of the method that is found
   * will be returned, starting with {@code classObj} and following its base classes and
   * interfaces recursively. This skips any method annotated inside a class that is not
   * marked {@link SkylarkModule} or is not a subclass of a class or interface marked
   * {@link SkylarkModule}.
   */
  @Nullable
  public static SkylarkCallable getSkylarkCallable(Class<?> classObj, Method method) {
    try {
      Method superMethod = classObj.getMethod(method.getName(), method.getParameterTypes());
      boolean classAnnotatedForCallables = getParentWithSkylarkModule(classObj) != null
          || hasSkylarkGlobalLibrary(classObj);
      if (classAnnotatedForCallables
          && superMethod.isAnnotationPresent(SkylarkCallable.class)) {
        return superMethod.getAnnotation(SkylarkCallable.class);
      }
    } catch (NoSuchMethodException e) {
      // The class might not have the specified method, so an exception is OK.
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
}
