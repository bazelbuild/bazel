// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.docgen;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin;
import com.google.devtools.build.lib.syntax.SkylarkCallable;

import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import javax.annotation.Nullable;

/**
 * A helper class to collect all the Java objects / methods reachable from Skylark.
 */
public class SkylarkJavaInterfaceExplorer {

  /**
   * A class representing a Skylark built-in object with its {@link SkylarkBuiltin} annotation
   * and the {@link SkylarkCallable} methods it might have.
   */
  static final class SkylarkJavaObject implements Comparable<SkylarkJavaObject> {

    private final SkylarkBuiltin annotation;
    @Nullable private final SkylarkBuiltin module;
    private final ImmutableMap<Method, SkylarkCallable> methods;
    private final ImmutableMap<String, SkylarkCallable> extraMethods;

    private SkylarkJavaObject(SkylarkBuiltin annotation,
        ImmutableMap<Method, SkylarkCallable> methods,
        ImmutableMap<String, SkylarkCallable> extraMethods) {
      this.annotation = Preconditions.checkNotNull(annotation);
      this.methods = methods;
      this.extraMethods = extraMethods;
      if (annotation.objectType().isAnnotationPresent(SkylarkBuiltin.class)) {
        module = annotation.objectType().getAnnotation(SkylarkBuiltin.class);
      } else {
        module = null;
      }
    }

    /**
     * Creates a {@link SkylarkJavaObject} from a {@link SkylarkBuiltin} annotation and a map
     * of java Methods and the corresponding {@link SkylarkCallable} annotations. 
     */
    static SkylarkJavaObject ofMethods(
        SkylarkBuiltin annotation, Map<Method, SkylarkCallable> methods) {
      return new SkylarkJavaObject(annotation,
          ImmutableMap.copyOf(methods), ImmutableMap.<String, SkylarkCallable>of());
    }

    /**
     * Creates a {@link SkylarkJavaObject} from a {@link SkylarkBuiltin} annotation and a map
     * of {@link SkylarkCallable} annotations mapped by the name of the methods they refer to.
     * These extraMethods don't refer to actual {@link Method} objects, but other types of
     * Skylark method implementations.
     */
    static SkylarkJavaObject ofExtraMethods(
        SkylarkBuiltin annotation, Map<String, SkylarkCallable> extraMethods) {
      return new SkylarkJavaObject(annotation,
          ImmutableMap.<Method, SkylarkCallable>of(), ImmutableMap.copyOf(extraMethods));
    }

    SkylarkBuiltin getAnnotation() {
      return annotation;
    }

    ImmutableMap<Method, SkylarkCallable> getMethods() {
      return methods;
    }

    ImmutableMap<String, SkylarkCallable> getExtraMethods() {
      return extraMethods;
    }

    @Override
    public int hashCode() {
      int hashcode = annotation.hashCode();
      if (module == null) {
        hashcode ^= module.hashCode();
      }
      return hashcode;
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof SkylarkJavaObject)) {
        return false;
      }
      SkylarkJavaObject o = (SkylarkJavaObject) obj;
      if (this.module != null) {
        if (!this.module.equals(o.module)) {
          return false;
        }
      } else {
        if (o.module != null) {
          return false;
        }
      }
      return annotation.equals(o.annotation);
    }

    @Override
    public int compareTo(SkylarkJavaObject o) {
      return this.name().compareTo(o.name());
    }

    public String name() {
      return (module != null ? module.name() + "." : "") + annotation.name();
    }
  }

  /**
   * Collects and returns all the Java objects reachable in Skylark from (and including)
   * firstClassObject with the corresponding SkylarkBuiltin annotations.
   *
   * <p>Note that the {@link SkylarkBuiltin} annotation for firstClassObject - firstAnnotation -
   * is also an input parameter, because some top level Skylark built-in objects and methods
   * are not annotated on the class, but on a field referencing them.
   */
  Set<SkylarkJavaObject> collect(SkylarkBuiltin firstAnnotation, Class<?> firstClassObject) {
    Set<SkylarkJavaObject> objects = new TreeSet<>();
    Set<Class<?>> processedClasses = new HashSet<>();
    LinkedList<Class<?>> classesToProcess = new LinkedList<>();
    Map<Class<?>, SkylarkBuiltin> annotations = new HashMap<>();

    classesToProcess.addLast(firstClassObject);
    annotations.put(firstClassObject, firstAnnotation);

    while (!classesToProcess.isEmpty()) {
      Class<?> classObject = classesToProcess.removeFirst();
      SkylarkBuiltin annotation = annotations.get(classObject);
      processedClasses.add(classObject);

      Map<Method, SkylarkCallable> methods =
          FuncallExpression.collectSkylarkMethodsWithAnnotation(classObject);
      for (Map.Entry<Method, SkylarkCallable> method : methods.entrySet()) {
        Class<?> returnClass = method.getKey().getReturnType();
        if (returnClass.isAnnotationPresent(SkylarkBuiltin.class)
            && !processedClasses.contains(returnClass)) {
          classesToProcess.addLast(returnClass);
          annotations.put(returnClass, returnClass.getAnnotation(SkylarkBuiltin.class));
        }
      }
      objects.add(SkylarkJavaObject.ofMethods(annotation, methods));
    }

    return objects;
  }
}
