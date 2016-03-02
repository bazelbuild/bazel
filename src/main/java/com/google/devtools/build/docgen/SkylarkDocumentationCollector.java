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
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.rules.SkylarkModules;
import com.google.devtools.build.lib.rules.SkylarkRuleContext;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.Jvm;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.MethodLibrary;
import com.google.devtools.build.lib.syntax.Runtime;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;

/**
 * A helper class that collects Skylark module documentation.
 */
final class SkylarkDocumentationCollector {
  @SkylarkModule(name = "globals",
      doc = "Objects, functions and modules registered in the global environment.")
  private static final class TopLevelModule {}

  private SkylarkDocumentationCollector() {}

  /**
   * Returns the SkylarkModule annotation for the top-level Skylark module.
   */
  public static SkylarkModule getTopLevelModule() {
    return TopLevelModule.class.getAnnotation(SkylarkModule.class);
  }

  /**
   * Collects the documentation for all Skylark modules and returns a map that maps Skylark
   * module name to the module documentation.
   */
  public static Map<String, SkylarkModuleDoc> collectModules() {
    Map<String, SkylarkModuleDoc> modules = new TreeMap<>();
    Map<String, SkylarkModuleDoc> builtinModules = collectBuiltinModules();
    Map<SkylarkModule, Class<?>> builtinJavaObjects = collectBuiltinJavaObjects();

    modules.putAll(builtinModules);
    for (SkylarkModuleDoc builtinObject : builtinModules.values()) {
      // Check the return type for built-in functions, it can be a module previously not added.
      for (SkylarkBuiltinMethodDoc builtinMethod : builtinObject.getBuiltinMethods().values()) {
        Class<?> type = builtinMethod.getAnnotation().returnType();
        if (type.isAnnotationPresent(SkylarkModule.class)) {
          collectJavaObjects(type.getAnnotation(SkylarkModule.class), type, modules);
        }
      }
      collectJavaObjects(builtinObject.getAnnotation(), builtinObject.getClassObject(), modules);
    }
    for (Entry<SkylarkModule, Class<?>> builtinModule : builtinJavaObjects.entrySet()) {
      collectJavaObjects(builtinModule.getKey(), builtinModule.getValue(), modules);
    }
    return modules;
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
    Deque<Class<?>> toProcess = new LinkedList<>();
    Map<Class<?>, SkylarkModule> annotations = new HashMap<>();

    toProcess.addLast(firstClass);
    annotations.put(firstClass, firstModule);

    while (!toProcess.isEmpty()) {
      Class<?> c = toProcess.removeFirst();
      SkylarkModule annotation = annotations.get(c);
      done.add(c);
      if (!modules.containsKey(annotation.name())) {
        modules.put(annotation.name(), new SkylarkModuleDoc(annotation, c));
      }
      SkylarkModuleDoc module = modules.get(annotation.name());

      if (module.javaMethodsNotCollected()) {
        ImmutableMap<Method, SkylarkCallable> methods =
            FuncallExpression.collectSkylarkMethodsWithAnnotation(c);
        for (Map.Entry<Method, SkylarkCallable> entry : methods.entrySet()) {
          module.addMethod(new SkylarkJavaMethodDoc(module, entry.getKey(), entry.getValue()));
        }

        for (Map.Entry<Method, SkylarkCallable> method : methods.entrySet()) {
          Class<?> returnClass = method.getKey().getReturnType();
          if (returnClass.isAnnotationPresent(SkylarkModule.class)
              && !done.contains(returnClass)) {
            toProcess.addLast(returnClass);
            annotations.put(returnClass, returnClass.getAnnotation(SkylarkModule.class));
          }
        }
      }
    }
  }

  private static Map<String, SkylarkModuleDoc> collectBuiltinModules() {
    Map<String, SkylarkModuleDoc> modules = new HashMap<>();
    collectBuiltinDoc(modules, Runtime.class.getDeclaredFields());
    collectBuiltinDoc(modules, MethodLibrary.class.getDeclaredFields());
    for (Class<?> moduleClass : SkylarkModules.MODULES) {
      collectBuiltinDoc(modules, moduleClass.getDeclaredFields());
    }
    return modules;
  }

  private static void collectBuiltinDoc(Map<String, SkylarkModuleDoc> modules, Field[] fields) {
    for (Field field : fields) {
      if (field.isAnnotationPresent(SkylarkSignature.class)) {
        SkylarkSignature skylarkSignature = field.getAnnotation(SkylarkSignature.class);
        Class<?> moduleClass = skylarkSignature.objectType();
        SkylarkModule skylarkModule = moduleClass.equals(Object.class)
            ? getTopLevelModule()
            : Runtime.getCanonicalRepresentation(moduleClass).getAnnotation(SkylarkModule.class);
        if (skylarkModule == null) {
          // TODO(bazel-team): we currently have undocumented methods on undocumented data
          // structures, namely java.util.List. Remove this case when we are done.
          Preconditions.checkState(!skylarkSignature.documented());
          Preconditions.checkState(moduleClass == List.class);
        } else {
          if (!modules.containsKey(skylarkModule.name())) {
            modules.put(skylarkModule.name(), new SkylarkModuleDoc(skylarkModule, moduleClass));
          }
          SkylarkModuleDoc module = modules.get(skylarkModule.name());
          module.addMethod(new SkylarkBuiltinMethodDoc(module, skylarkSignature, field.getType()));
        }
      }
    }
  }

  private static Map<SkylarkModule, Class<?>> collectBuiltinJavaObjects() {
    Map<SkylarkModule, Class<?>> modules = new HashMap<>();
    collectBuiltinModule(modules, SkylarkRuleContext.class);
    collectBuiltinModule(modules, TransitiveInfoCollection.class);
    collectBuiltinModule(modules, AppleConfiguration.class);
    collectBuiltinModule(modules, CppConfiguration.class);
    collectBuiltinModule(modules, JavaConfiguration.class);
    collectBuiltinModule(modules, Jvm.class);
    return modules;
  }

  private static void collectBuiltinModule(
      Map<SkylarkModule, Class<?>> modules, Class<?> moduleClass) {
    if (moduleClass.isAnnotationPresent(SkylarkModule.class)) {
      SkylarkModule skylarkModule = moduleClass.getAnnotation(SkylarkModule.class);
      modules.put(skylarkModule, moduleClass);
    }
  }

  /**
   * Returns the top level modules and functions with their documentation in a command-line
   * printable format.
   */
  public static Map<String, String> collectTopLevelModules() {
    Map<String, String> modules = new TreeMap<>();
    for (SkylarkModuleDoc doc : collectBuiltinModules().values()) {
      if (doc.getAnnotation() == getTopLevelModule()) {
        for (Map.Entry<String, SkylarkBuiltinMethodDoc> entry :
            doc.getBuiltinMethods().entrySet()) {
          if (entry.getValue().documented()) {
            modules.put(entry.getKey(),
                DocgenConsts.toCommandLineFormat(entry.getValue().getDocumentation()));
          }
        }
      } else {
        modules.put(doc.getAnnotation().name(),
            DocgenConsts.toCommandLineFormat(doc.getAnnotation().doc()));
      }
    }
    return modules;
  }
}
