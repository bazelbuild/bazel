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

import java.lang.reflect.Method;
import javax.annotation.Nullable;

/**
 * Helpers for accessing Skylark interface annotations.
 */
public class SkylarkInterfaceUtils {

  /**
   * Returns the {@link SkylarkModule} annotation for the given class, if it exists, and
   * null otherwise. The first annotation found will be returned, starting with {@code classObj}
   * and following its base classes and interfaces recursively.
   */
  @Nullable
  public static SkylarkModule getSkylarkModule(Class<?> classObj) {
    if (classObj.isAnnotationPresent(SkylarkModule.class)) {
      return classObj.getAnnotation(SkylarkModule.class);
    }
    Class<?> superclass = classObj.getSuperclass();
    if (superclass != null) {
      SkylarkModule annotation = getSkylarkModule(superclass);
      if (annotation != null) {
        return annotation;
      }
    }
    for (Class<?> interfaceObj : classObj.getInterfaces()) {
      SkylarkModule annotation = getSkylarkModule(interfaceObj);
      if (annotation != null) {
        return annotation;
      }
    }
    return null;
  }

  /**
   * Returns the {@link SkylarkCallable} annotation for the given method, if it exists, and
   * null otherwise. The first annotation of an overridden version of the method that is found
   * will be returned, starting with {@code classObj} and following its base classes and
   * interfaces recursively, skipping any annotation inside a class not marked
   * {@link SkylarkModule}.
   */
  @Nullable
  public static SkylarkCallable getSkylarkCallable(Class<?> classObj, Method method) {
    try {
      Method superMethod = classObj.getMethod(method.getName(), method.getParameterTypes());
      if (classObj.isAnnotationPresent(SkylarkModule.class)
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
