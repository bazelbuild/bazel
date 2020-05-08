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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.docgen.starlark.StarlarkBuiltinDoc;
import com.google.devtools.build.docgen.starlark.StarlarkConstructorMethodDoc;
import com.google.devtools.build.docgen.starlark.StarlarkJavaMethodDoc;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkGlobalLibrary;
import com.google.devtools.build.lib.skylarkinterface.StarlarkBuiltin;
import com.google.devtools.build.lib.skylarkinterface.StarlarkDocumentationCategory;
import com.google.devtools.build.lib.syntax.CallUtils;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import java.lang.reflect.Method;
import java.util.Map;
import java.util.TreeMap;
import javax.annotation.Nullable;

/** A helper class that collects Starlark module documentation. */
final class StarlarkDocumentationCollector {
  @StarlarkBuiltin(
      name = "globals",
      title = "Globals",
      category = StarlarkDocumentationCategory.TOP_LEVEL_TYPE,
      doc = "Objects, functions and modules registered in the global environment.")
  private static final class TopLevelModule implements StarlarkValue {}

  private StarlarkDocumentationCollector() {}

  /** Returns the StarlarkBuiltin annotation for the top-level Starlark module. */
  public static StarlarkBuiltin getTopLevelModule() {
    return TopLevelModule.class.getAnnotation(StarlarkBuiltin.class);
  }

  /**
   * Collects the documentation for all Starlark modules comprised of the given classes and returns
   * a map that maps Starlark module name to the module documentation.
   */
  public static Map<String, StarlarkBuiltinDoc> collectModules(Iterable<Class<?>> classes) {
    Map<String, StarlarkBuiltinDoc> modules = new TreeMap<>();
    // The top level module first.
    // (This is a special case of {@link StarlarkBuiltinDoc} as it has no object name).
    StarlarkBuiltin topLevelModule = getTopLevelModule();
    modules.put(
        topLevelModule.name(), new StarlarkBuiltinDoc(topLevelModule, TopLevelModule.class));

    // Creating module documentation is done in three passes.
    // 1. Add all classes/interfaces annotated with @StarlarkBuiltin with documented = true.
    for (Class<?> candidateClass : classes) {
      if (candidateClass.isAnnotationPresent(StarlarkBuiltin.class)) {
        collectSkylarkModule(candidateClass, modules);
      }
    }

    // 2. Add all object methods and global functions.
    for (Class<?> candidateClass : classes) {
      if (candidateClass.isAnnotationPresent(StarlarkBuiltin.class)) {
        collectModuleMethods(candidateClass, modules);
      }
      if (candidateClass.isAnnotationPresent(SkylarkGlobalLibrary.class)) {
        collectGlobalLibraryMethods(candidateClass, modules);
      }
    }

    // 3. Add all constructors.
    for (Class<?> candidateClass : classes) {
      if (candidateClass.isAnnotationPresent(StarlarkBuiltin.class)
          || candidateClass.isAnnotationPresent(SkylarkGlobalLibrary.class)) {
        collectConstructorMethods(candidateClass, modules);
      }
    }

    return modules;
  }

  /**
   * Returns the {@link StarlarkBuiltinDoc} entry representing the collection of top level
   * functions. (This is a special case of {@link StarlarkBuiltinDoc} as it has no object name).
   */
  private static StarlarkBuiltinDoc getTopLevelModuleDoc(Map<String, StarlarkBuiltinDoc> modules) {
    return modules.get(getTopLevelModule().name());
  }

  /**
   * Adds a single {@link StarlarkBuiltinDoc} entry to {@code modules} representing the given {@code
   * moduleClass}, if it is a documented module.
   */
  private static void collectSkylarkModule(
      Class<?> moduleClass, Map<String, StarlarkBuiltinDoc> modules) {
    if (moduleClass.equals(TopLevelModule.class)) {
      // The top level module doc is a special case and is handled separately.
      return;
    }

    StarlarkBuiltin moduleAnnotation =
        Preconditions.checkNotNull(moduleClass.getAnnotation(StarlarkBuiltin.class));

    if (moduleAnnotation.documented()) {
      StarlarkBuiltinDoc previousModuleDoc = modules.get(moduleAnnotation.name());
      if (previousModuleDoc == null) {
        modules.put(moduleAnnotation.name(), new StarlarkBuiltinDoc(moduleAnnotation, moduleClass));
      } else {
        // Handle a strange corner-case: If moduleClass has a subclass which is also
        // annotated with {@link StarlarkBuiltin} with the same name, and also has the same
        // module-level docstring, then the subclass takes precedence.
        // (This is useful if one module is a "common" stable module, and its subclass is
        // an experimental module that also supports all stable methods.)
        validateCompatibleModules(previousModuleDoc.getClassObject(), moduleClass);

        if (previousModuleDoc.getClassObject().isAssignableFrom(moduleClass)) {
          // The new module is a subclass of the old module, so use the subclass.
          modules.put(
              moduleAnnotation.name(), new StarlarkBuiltinDoc(moduleAnnotation, moduleClass));
        }
      }
    }
  }

  /**
   * Validate that it is acceptable that the given module classes with the same module name
   * co-exist.
   */
  private static void validateCompatibleModules(Class<?> one, Class<?> two) {
    StarlarkBuiltin moduleOne = one.getAnnotation(StarlarkBuiltin.class);
    StarlarkBuiltin moduleTwo = two.getAnnotation(StarlarkBuiltin.class);
    if (one.isAssignableFrom(two) || two.isAssignableFrom(one)) {
      if (!moduleOne.doc().equals(moduleTwo.doc())) {
        throw new IllegalStateException(
            String.format(
                "%s and %s are related modules but have mismatching documentation for '%s'",
                one, two, moduleOne.name()));
      }
    } else {
      throw new IllegalStateException(
          String.format(
              "%s and %s are unrelated modules with documentation for '%s'",
              one, two, moduleOne.name()));
    }
  }

  private static void collectModuleMethods(
      Class<?> moduleClass, Map<String, StarlarkBuiltinDoc> modules) {
    StarlarkBuiltin moduleAnnotation =
        Preconditions.checkNotNull(moduleClass.getAnnotation(StarlarkBuiltin.class));

    if (moduleAnnotation.documented()) {
      StarlarkBuiltinDoc moduleDoc =
          Preconditions.checkNotNull(modules.get(moduleAnnotation.name()));

      if (moduleClass == moduleDoc.getClassObject()) {
        ImmutableMap<Method, SkylarkCallable> methods =
            CallUtils.collectSkylarkMethodsWithAnnotation(moduleClass);
        for (Map.Entry<Method, SkylarkCallable> entry : methods.entrySet()) {
          // Only collect methods not annotated with @SkylarkConstructor. Methods with
          // @SkylarkConstructor are added later.
          if (!entry.getKey().isAnnotationPresent(SkylarkConstructor.class)) {
            moduleDoc.addMethod(
                new StarlarkJavaMethodDoc(moduleDoc.getName(), entry.getKey(), entry.getValue()));
          }
        }
      }
    }
  }

  @Nullable
  private static Method getSelfCallConstructorMethod(Class<?> objectClass) {
    Method selfCallMethod = CallUtils.getSelfCallMethod(StarlarkSemantics.DEFAULT, objectClass);
    if (selfCallMethod != null && selfCallMethod.isAnnotationPresent(SkylarkConstructor.class)) {
      return selfCallMethod;
    }
    return null;
  }

  /**
   * Adds {@link StarlarkJavaMethodDoc} entries to the top level module, one for
   * each @SkylarkCallable method defined in the given @SkylarkGlobalLibrary class {@code
   * moduleClass}.
   */
  private static void collectGlobalLibraryMethods(
      Class<?> moduleClass, Map<String, StarlarkBuiltinDoc> modules) {
    Preconditions.checkArgument(moduleClass.isAnnotationPresent(SkylarkGlobalLibrary.class));
    StarlarkBuiltinDoc topLevelModuleDoc = getTopLevelModuleDoc(modules);

    ImmutableMap<Method, SkylarkCallable> methods =
        CallUtils.collectSkylarkMethodsWithAnnotation(moduleClass);
    for (Map.Entry<Method, SkylarkCallable> entry : methods.entrySet()) {
      // Only add non-constructor global library methods. Constructors are added later.
      if (!entry.getKey().isAnnotationPresent(SkylarkConstructor.class)) {
        topLevelModuleDoc.addMethod(
            new StarlarkJavaMethodDoc("", entry.getKey(), entry.getValue()));
      }
    }
  }

  private static void collectConstructor(
      Map<String, StarlarkBuiltinDoc> modules, Class<?> moduleClass, Method method) {
    SkylarkConstructor constructorAnnotation =
        Preconditions.checkNotNull(method.getAnnotation(SkylarkConstructor.class));
    SkylarkCallable callable =
        Preconditions.checkNotNull(method.getAnnotation(SkylarkCallable.class));
    Class<?> objectClass = constructorAnnotation.objectType();
    StarlarkBuiltin objectModule = objectClass.getAnnotation(StarlarkBuiltin.class);
    if (objectModule == null || !objectModule.documented()) {
      // The class of the constructed object type has no documentation, so no place to add
      // constructor information.
      return;
    }
    StarlarkBuiltinDoc module = modules.get(objectModule.name());

    String fullyQualifiedName;
    if (!constructorAnnotation.receiverNameForDoc().isEmpty()) {
      fullyQualifiedName = constructorAnnotation.receiverNameForDoc();
    } else {
      String originatingModuleName = getModuleNameForConstructorPrefix(moduleClass, modules);
      fullyQualifiedName = getFullyQualifiedName(originatingModuleName, callable);
    }

    module.setConstructor(new StarlarkConstructorMethodDoc(fullyQualifiedName, method, callable));
  }

  /**
   * Collect two types of constructor methods:
   *
   * <p>1. Methods that are annotated with @SkylarkConstructor.
   *
   * <p>2. Structfield methods that return an object which itself has a method with selfCall = true,
   * and is annotated with @SkylarkConstructor. (For example, suppose Foo has a structfield method
   * 'bar'. If Foo.bar is itself callable, and is a constructor, then Foo.bar() should be treated
   * like a constructor method.)
   */
  private static void collectConstructorMethods(
      Class<?> moduleClass, Map<String, StarlarkBuiltinDoc> modules) {
    Method selfCallConstructor = getSelfCallConstructorMethod(moduleClass);
    if (selfCallConstructor != null) {
      collectConstructor(modules, moduleClass, selfCallConstructor);
    }

    ImmutableMap<Method, SkylarkCallable> methods =
        CallUtils.collectSkylarkMethodsWithAnnotation(moduleClass);
    for (Map.Entry<Method, SkylarkCallable> entry : methods.entrySet()) {
      if (entry.getKey().isAnnotationPresent(SkylarkConstructor.class)) {
        collectConstructor(modules, moduleClass, entry.getKey());
      }
      Class<?> returnClass = entry.getKey().getReturnType();
      Method returnClassConstructor = getSelfCallConstructorMethod(returnClass);
      if (returnClassConstructor != null) {
        collectConstructor(modules, moduleClass, returnClassConstructor);
      }
    }
  }

  private static String getModuleNameForConstructorPrefix(
      Class<?> moduleClass, Map<String, StarlarkBuiltinDoc> modules) {
    if (moduleClass.isAnnotationPresent(StarlarkBuiltin.class)) {
      String moduleName = moduleClass.getAnnotation(StarlarkBuiltin.class).name();
      StarlarkBuiltinDoc moduleDoc = Preconditions.checkNotNull(modules.get(moduleName));

      if (moduleClass != moduleDoc.getClassObject()) {
        throw new IllegalStateException(
            "Could not determine module name for constructor defined in " + moduleClass);
      }
      return moduleName;
    } else if (moduleClass.isAnnotationPresent(SkylarkGlobalLibrary.class)) {
      return "";
    } else {
      throw new IllegalArgumentException(moduleClass + " has no valid annotation");
    }
  }

  private static String getFullyQualifiedName(
      String objectName, SkylarkCallable callable) {
    String objectDotExpressionPrefix = objectName.isEmpty() ? "" : objectName + ".";
    String methodName = callable.name();
    return objectDotExpressionPrefix + methodName;
  }
}
