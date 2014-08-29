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
import com.google.devtools.build.lib.syntax.SkylarkModule;
import com.google.devtools.build.lib.util.StringUtilities;

import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * A helper class to collect all the Java objects / methods reachable from Skylark.
 */
public class SkylarkJavaInterfaceExplorer {

  /**
   * A class representing a Skylark built-in object with its {@link SkylarkBuiltin} annotation
   * and the {@link SkylarkCallable} methods it might have.
   */
  static final class SkylarkModuleDoc {

    private final SkylarkModule module;
    private final Class<?> classObject;
    private final Map<String, SkylarkBuiltin> builtin;
    private Map<String, Map.Entry<Method, SkylarkCallable>> methods = null;

    SkylarkModuleDoc(SkylarkModule module, Class<?> classObject) {
      this.module = Preconditions.checkNotNull(module,
          "Class has to be annotated with SkylarkModule: " + classObject);
      this.classObject = classObject;
      this.builtin = new TreeMap<>();
    }

    SkylarkModule getAnnotation() {
      return module;
    }

    Class<?> getClassObject() {
      return classObject;
    }

    private boolean javaMethodsNotCollected() {
      return methods == null;
    }

    private void setJavaMethods(Map<String, Map.Entry<Method, SkylarkCallable>> methods) {
      this.methods = methods;
    }

    Map<String, SkylarkBuiltin> getBuiltinMethods() {
      return builtin;
    }

    Map<String, Map.Entry<Method, SkylarkCallable>> getJavaMethods() {
      return methods;
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
  void collect(SkylarkModule firstModule, Class<?> firstClass,
      Map<String, SkylarkModuleDoc> modules) {
    Set<Class<?>> processedClasses = new HashSet<>();
    LinkedList<Class<?>> classesToProcess = new LinkedList<>();
    Map<Class<?>, SkylarkModule> annotations = new HashMap<>();

    classesToProcess.addLast(firstClass);
    annotations.put(firstClass, firstModule);

    while (!classesToProcess.isEmpty()) {
      Class<?> classObject = classesToProcess.removeFirst();
      SkylarkModule annotation = annotations.get(classObject);
      processedClasses.add(classObject);
      if (!modules.containsKey(annotation.name())) {
        modules.put(annotation.name(), new SkylarkModuleDoc(annotation, classObject));
      }
      SkylarkModuleDoc module = modules.get(annotation.name());

      if (module.javaMethodsNotCollected()) {
        ImmutableMap<Method, SkylarkCallable> methods =
            FuncallExpression.collectSkylarkMethodsWithAnnotation(classObject);
        // Order the methods alphabetically
        Map<String, Map.Entry<Method, SkylarkCallable>> methodMap = new TreeMap<>();
        for (Map.Entry<Method, SkylarkCallable> entry : methods.entrySet()) {
          methodMap.put(getName(entry), entry);
        }

        for (Map.Entry<Method, SkylarkCallable> method : methods.entrySet()) {
          Class<?> returnClass = method.getKey().getReturnType();
          if (returnClass.isAnnotationPresent(SkylarkModule.class)
              && !processedClasses.contains(returnClass)) {
            classesToProcess.addLast(returnClass);
            annotations.put(returnClass, returnClass.getAnnotation(SkylarkModule.class));
          }
        }
        module.setJavaMethods(methodMap);
      }
    }
  }

  private String getName(Map.Entry<Method, SkylarkCallable> method) {
    return method.getValue().name().isEmpty()
        ? StringUtilities.toPythonStyleFunctionName(method.getKey().getName())
        : method.getValue().name();
  }

}
