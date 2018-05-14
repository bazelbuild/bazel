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
package com.google.devtools.build.docgen;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.docgen.skylark.SkylarkBuiltinMethodDoc;
import com.google.devtools.build.docgen.skylark.SkylarkJavaMethodDoc;
import com.google.devtools.build.docgen.skylark.SkylarkModuleDoc;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkGlobalLibrary;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.util.Classpath;
import com.google.devtools.build.lib.util.Classpath.ClassPathException;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import javax.annotation.Nullable;

/**
 * A helper class that collects Skylark module documentation.
 */
final class SkylarkDocumentationCollector {
  @SkylarkModule(
    name = "globals",
    title = "Globals",
    category = SkylarkModuleCategory.TOP_LEVEL_TYPE,
    doc = "Objects, functions and modules registered in the global environment."
  )
  private static final class TopLevelModule {}

  // Common prefix of packages that may contain Skylark modules.
  private static final String MODULES_PACKAGE_PREFIX = "com/google/devtools/build";

  private SkylarkDocumentationCollector() {}

  /**
   * Returns the SkylarkModule annotation for the top-level Skylark module.
   */
  public static SkylarkModule getTopLevelModule() {
    return TopLevelModule.class.getAnnotation(SkylarkModule.class);
  }

  /**
   * Collects the documentation for all Skylark modules and returns a map that maps Skylark module
   * name to the module documentation.
   *
   * <p>WARNING: This method no longer supports the specification of additional module classes via
   * parameters. Instead, all module classes are being picked up automatically.
   */
  public static Map<String, SkylarkModuleDoc> collectModules()
      throws ClassPathException {
    Map<String, SkylarkModuleDoc> modules = new TreeMap<>();
    for (Class<?> candidateClass : Classpath.findClasses(MODULES_PACKAGE_PREFIX)) {
      SkylarkModule annotation = candidateClass.getAnnotation(SkylarkModule.class);
      if (annotation != null) {
        collectJavaObjects(annotation, candidateClass, modules);
      }
      SkylarkGlobalLibrary
          globalNamespaceAnnotation = candidateClass.getAnnotation(SkylarkGlobalLibrary.class);
      if (globalNamespaceAnnotation != null) {
        collectBuiltinMethods(modules, candidateClass);
      }
      collectBuiltinDoc(modules, candidateClass.getDeclaredFields());
    }
    return modules;
  }

  private static SkylarkModuleDoc getTopLevelModuleDoc(Map<String, SkylarkModuleDoc> modules) {
    SkylarkModule annotation = getTopLevelModule();
    modules.computeIfAbsent(
        annotation.name(), (String k) -> new SkylarkModuleDoc(annotation, Object.class));
    return modules.get(annotation.name());
  }

  private static SkylarkModuleDoc getSkylarkModuleDoc(
      Class<?> moduleClass, Map<String, SkylarkModuleDoc> modules) {
    if (moduleClass.equals(Object.class)) {
      return getTopLevelModuleDoc(modules);
    }

    SkylarkModule annotation = Preconditions.checkNotNull(
        Runtime.getSkylarkNamespace(moduleClass).getAnnotation(SkylarkModule.class));
    modules.computeIfAbsent(
        annotation.name(), (String k) -> new SkylarkModuleDoc(annotation, moduleClass));
    return modules.get(annotation.name());
  }

  /**
   * Collects and returns all the Java objects reachable in Skylark from (and including)
   * firstClass with the corresponding SkylarkModule annotation.
   *
   * <p>Note that the {@link SkylarkModule} annotation for firstClass - firstModule -
   * is also an input parameter, because some top level Skylark built-in objects and methods
   * are not annotated on the class, but on a field referencing them.
   */
  @VisibleForTesting
  static void collectJavaObjects(SkylarkModule firstModule, Class<?> firstClass,
      Map<String, SkylarkModuleDoc> modules) {
    Set<Class<?>> done = new HashSet<>();
    Deque<Class<?>> toProcess = new ArrayDeque<>();

    toProcess.addLast(firstClass);

    while (!toProcess.isEmpty()) {
      Class<?> c = toProcess.removeFirst();
      if (done.contains(c)) {
        continue;
      }

      SkylarkModuleDoc module = getSkylarkModuleDoc(c, modules);
      done.add(c);

      if (module.javaMethodsNotCollected()) {
        ImmutableMap<Method, SkylarkCallable> methods =
            FuncallExpression.collectSkylarkMethodsWithAnnotation(c);
        for (Map.Entry<Method, SkylarkCallable> entry : methods.entrySet()) {
          if (entry.getKey().isAnnotationPresent(SkylarkConstructor.class)) {
            collectConstructor(modules, module.getName(), entry.getKey(), entry.getValue());
          } else {
            module.addMethod(
                new SkylarkJavaMethodDoc(module.getName(), entry.getKey(), entry.getValue()));
          }

          Class<?> returnClass = entry.getKey().getReturnType();
          if (returnClass.isAnnotationPresent(SkylarkModule.class)) {
            toProcess.addLast(returnClass);
          } else {
            Map.Entry<Method, SkylarkCallable> selfCallConstructor =
                getSelfCallConstructorMethod(returnClass);
            if (selfCallConstructor != null) {
              // If the class to be processed is not annotated with @SkylarkModule, then its
              // @SkylarkCallable methods are not processed, as it does not have its own
              // documentation page. However, if it is a callable object (has a selfCall method)
              // that is also a constructor for another type, we still want to ensure that method
              // is documented.
              // This is used for builtin providers, which typically are not marked @SkylarkModule,
              // but which have selfCall constructors for their corresponding Info class.

              // For example, the "mymodule" module may return a callable object at mymodule.foo
              // which constructs instances of the Bar class. The type returned by mymodule.foo
              // may have no documentation, but mymodule.foo should be documented as a
              // constructor of Bar objects.
              collectConstructor(modules, module.getName(),
                  selfCallConstructor.getKey(), selfCallConstructor.getValue());
            }
          }
        }
      }
    }
  }

  @Nullable
  private static Map.Entry<Method, SkylarkCallable> getSelfCallConstructorMethod(
      Class<?> objectClass) {
    ImmutableMap<Method, SkylarkCallable> methods =
        FuncallExpression.collectSkylarkMethodsWithAnnotation(objectClass);
    for (Map.Entry<Method, SkylarkCallable> entry : methods.entrySet()) {
      if (entry.getValue().selfCall()
          && entry.getKey().isAnnotationPresent(SkylarkConstructor.class)) {
        // It's illegal, and checked by the interpreter, for there to be more than one method
        // annotated with selfCall. Thus, it's valid to return on the first find.
        return entry;
      }
    }
    return null;
  }

  private static void collectBuiltinDoc(Map<String, SkylarkModuleDoc> modules, Field[] fields) {
    for (Field field : fields) {
      if (field.isAnnotationPresent(SkylarkSignature.class)) {
        SkylarkSignature skylarkSignature = field.getAnnotation(SkylarkSignature.class);
        Class<?> moduleClass = skylarkSignature.objectType();

        SkylarkModuleDoc module = getSkylarkModuleDoc(moduleClass, modules);
        module.addMethod(new SkylarkBuiltinMethodDoc(module, skylarkSignature, field.getType()));
      }
    }
  }

  private static void collectBuiltinMethods(
      Map<String, SkylarkModuleDoc> modules, Class<?> moduleClass) {

    SkylarkModuleDoc topLevelModuleDoc = getTopLevelModuleDoc(modules);

    ImmutableMap<Method, SkylarkCallable> methods =
        FuncallExpression.collectSkylarkMethodsWithAnnotation(moduleClass);
    for (Map.Entry<Method, SkylarkCallable> entry : methods.entrySet()) {
      if (entry.getKey().isAnnotationPresent(SkylarkConstructor.class)) {
        collectConstructor(modules, "", entry.getKey(), entry.getValue());
      } else {
        topLevelModuleDoc.addMethod(new SkylarkJavaMethodDoc("", entry.getKey(), entry.getValue()));
      }
    }
  }

  private static void collectConstructor(Map<String, SkylarkModuleDoc> modules,
      String originatingModuleName, Method method, SkylarkCallable callable) {
    SkylarkConstructor constructorAnnotation =
        Preconditions.checkNotNull(method.getAnnotation(SkylarkConstructor.class));
    Class<?> objectClass = constructorAnnotation.objectType();
    SkylarkModuleDoc module = getSkylarkModuleDoc(objectClass, modules);
    module.setConstructor(new SkylarkJavaMethodDoc(originatingModuleName, method, callable));
  }
}
