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
import com.google.devtools.build.docgen.skylark.SkylarkBuiltinMethodDoc;
import com.google.devtools.build.docgen.skylark.SkylarkConstructorMethodDoc;
import com.google.devtools.build.docgen.skylark.SkylarkJavaMethodDoc;
import com.google.devtools.build.docgen.skylark.SkylarkModuleDoc;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkGlobalLibrary;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Map;
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

  private SkylarkDocumentationCollector() {}

  /**
   * Returns the SkylarkModule annotation for the top-level Skylark module.
   */
  public static SkylarkModule getTopLevelModule() {
    return TopLevelModule.class.getAnnotation(SkylarkModule.class);
  }

  /**
   * Collects the documentation for all Skylark modules comprised of the given classes and returns a
   * map that maps Skylark module name to the module documentation.
   */
  public static Map<String, SkylarkModuleDoc> collectModules(Iterable<Class<?>> classes) {
    Map<String, SkylarkModuleDoc> modules = new TreeMap<>();
    // The top level module first.
    // (This is a special case of {@link SkylarkModuleDoc} as it has no object name).
    SkylarkModule topLevelModule = getTopLevelModule();
    modules.put(topLevelModule.name(), new SkylarkModuleDoc(topLevelModule, TopLevelModule.class));

    // Creating module documentation is done in three passes.
    // 1. Add all classes/interfaces annotated with @SkylarkModule with documented = true.
    for (Class<?> candidateClass : classes) {
      if (candidateClass.isAnnotationPresent(SkylarkModule.class)) {
        collectSkylarkModule(candidateClass, modules);
      }
    }

    // 2. Add all object methods and global functions.
    for (Class<?> candidateClass : classes) {
      if (candidateClass.isAnnotationPresent(SkylarkModule.class)) {
        collectModuleMethods(candidateClass, modules);
      }
      if (candidateClass.isAnnotationPresent(SkylarkGlobalLibrary.class)) {
        collectGlobalLibraryMethods(candidateClass, modules);
      }
      // Use of SkylarkSignature fields is deprecated, but not all uses have been migrated.
      collectSkylarkSignatureFunctions(candidateClass, modules);
    }

    // 3. Add all constructors.
    for (Class<?> candidateClass : classes) {
      if (candidateClass.isAnnotationPresent(SkylarkModule.class)
          || candidateClass.isAnnotationPresent(SkylarkGlobalLibrary.class)) {
        collectConstructorMethods(candidateClass, modules);
      }
    }

    return modules;
  }

  /**
   * Returns the {@link SkylarkModuleDoc} entry representing the collection of top level functions.
   * (This is a special case of {@link SkylarkModuleDoc} as it has no object name).
   */
  private static SkylarkModuleDoc getTopLevelModuleDoc(Map<String, SkylarkModuleDoc> modules) {
    return modules.get(getTopLevelModule().name());
  }

  /**
   * Adds a single {@link SkylarkModuleDoc} entry to {@code modules} representing the given {@code
   * moduleClass}, if it is a documented module.
   */
  private static void collectSkylarkModule(
      Class<?> moduleClass, Map<String, SkylarkModuleDoc> modules) {
    if (moduleClass.equals(TopLevelModule.class)) {
      // The top level module doc is a special case and is handled separately.
      return;
    }

    SkylarkModule moduleAnnotation =
        Preconditions.checkNotNull(moduleClass.getAnnotation(SkylarkModule.class));

    if (moduleAnnotation.documented()) {
      SkylarkModuleDoc previousModuleDoc = modules.get(moduleAnnotation.name());
      if (previousModuleDoc == null) {
        modules.put(moduleAnnotation.name(), new SkylarkModuleDoc(moduleAnnotation, moduleClass));
      } else {
        // Handle a strange corner-case: If moduleClass has a subclass which is also
        // annotated with @SkylarkModule with the same name, and also has the same module-level
        // docstring, then the subclass takes precedence.
        // (This is useful if one module is a "common" stable module, and its subclass is
        // an experimental module that also supports all stable methods.)
        validateCompatibleModules(previousModuleDoc.getClassObject(), moduleClass);

        if (previousModuleDoc.getClassObject().isAssignableFrom(moduleClass)) {
          // The new module is a subclass of the old module, so use the subclass.
          modules.put(moduleAnnotation.name(), new SkylarkModuleDoc(moduleAnnotation, moduleClass));
        }
      }
    }
  }

  /**
   * Validate that it is acceptable that the given module classes with the same module name
   * co-exist.
   */
  private static void validateCompatibleModules(Class<?> one, Class<?> two) {
    SkylarkModule moduleOne = one.getAnnotation(SkylarkModule.class);
    SkylarkModule moduleTwo = two.getAnnotation(SkylarkModule.class);
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
      Class<?> moduleClass, Map<String, SkylarkModuleDoc> modules) {
    SkylarkModule moduleAnnotation =
        Preconditions.checkNotNull(moduleClass.getAnnotation(SkylarkModule.class));

    if (moduleAnnotation.documented()) {
      SkylarkModuleDoc moduleDoc = Preconditions.checkNotNull(modules.get(moduleAnnotation.name()));

      if (moduleClass == moduleDoc.getClassObject()) {
        ImmutableMap<Method, SkylarkCallable> methods =
            FuncallExpression.collectSkylarkMethodsWithAnnotation(moduleClass);
        for (Map.Entry<Method, SkylarkCallable> entry : methods.entrySet()) {
          // Only collect methods not annotated with @SkylarkConstructor. Methods with
          // @SkylarkConstructor are added later.
          if (!entry.getKey().isAnnotationPresent(SkylarkConstructor.class)) {
            moduleDoc.addMethod(
                new SkylarkJavaMethodDoc(moduleDoc.getName(), entry.getKey(), entry.getValue()));
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

  /**
   * Adds {@link SkylarkJavaMethodDoc} entries to the top level module, one for
   * each @SkylarkCallable method defined in the given @SkylarkGlobalLibrary class {@code
   * moduleClass}.
   */
  private static void collectGlobalLibraryMethods(
      Class<?> moduleClass, Map<String, SkylarkModuleDoc> modules) {
    Preconditions.checkArgument(moduleClass.isAnnotationPresent(SkylarkGlobalLibrary.class));
    SkylarkModuleDoc topLevelModuleDoc = getTopLevelModuleDoc(modules);

    ImmutableMap<Method, SkylarkCallable> methods =
        FuncallExpression.collectSkylarkMethodsWithAnnotation(moduleClass);
    for (Map.Entry<Method, SkylarkCallable> entry : methods.entrySet()) {
      // Only add non-constructor global library methods. Constructors are added later.
      if (!entry.getKey().isAnnotationPresent(SkylarkConstructor.class)) {
        topLevelModuleDoc.addMethod(new SkylarkJavaMethodDoc("", entry.getKey(), entry.getValue()));
      }
    }
  }

  /**
   * Adds {@link SkylarkBuiltinMethodDoc} entries to the top level module, one for
   * each @SkylarkSignature-annotated field defined in the given {@code moduleClass}.
   *
   * <p>Note that use of SkylarkSignature fields is deprecated, but not all uses have been migrated.
   */
  private static void collectSkylarkSignatureFunctions(
      Class<?> moduleClass, Map<String, SkylarkModuleDoc> modules) {

    SkylarkModuleDoc topLevelModuleDoc = getTopLevelModuleDoc(modules);

    // Collect any fields annotated with @SkylarkSignature, even if the class isn't
    // annotated.
    for (Field field : moduleClass.getDeclaredFields()) {
      if (field.isAnnotationPresent(SkylarkSignature.class)) {
        SkylarkSignature skylarkSignature = field.getAnnotation(SkylarkSignature.class);
        Preconditions.checkState(skylarkSignature.objectType() == Object.class);

        topLevelModuleDoc.addMethod(
            new SkylarkBuiltinMethodDoc(
                getTopLevelModuleDoc(modules), skylarkSignature, field.getType()));
      }
    }
  }

  private static void collectConstructor(
      Map<String, SkylarkModuleDoc> modules,
      Class<?> moduleClass,
      Method method,
      SkylarkCallable callable) {
    SkylarkConstructor constructorAnnotation =
        Preconditions.checkNotNull(method.getAnnotation(SkylarkConstructor.class));
    Class<?> objectClass = constructorAnnotation.objectType();
    SkylarkModule objectModule = objectClass.getAnnotation(SkylarkModule.class);
    if (objectModule == null || !objectModule.documented()) {
      // The class of the constructed object type has no documentation, so no place to add
      // constructor information.
      return;
    }
    SkylarkModuleDoc module = modules.get(objectModule.name());

    String fullyQualifiedName;
    if (!constructorAnnotation.receiverNameForDoc().isEmpty()) {
      fullyQualifiedName = constructorAnnotation.receiverNameForDoc();
    } else {
      String originatingModuleName = getModuleNameForConstructorPrefix(moduleClass, modules);
      fullyQualifiedName = getFullyQualifiedName(originatingModuleName, callable);
    }

    module.setConstructor(new SkylarkConstructorMethodDoc(fullyQualifiedName, method, callable));
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
      Class<?> moduleClass, Map<String, SkylarkModuleDoc> modules) {

    ImmutableMap<Method, SkylarkCallable> methods =
        FuncallExpression.collectSkylarkMethodsWithAnnotation(moduleClass);
    for (Map.Entry<Method, SkylarkCallable> entry : methods.entrySet()) {
      if (entry.getKey().isAnnotationPresent(SkylarkConstructor.class)) {
        collectConstructor(modules, moduleClass, entry.getKey(), entry.getValue());
      }
      Class<?> returnClass = entry.getKey().getReturnType();
      Map.Entry<Method, SkylarkCallable> selfCallConstructor =
          getSelfCallConstructorMethod(returnClass);
      if (selfCallConstructor != null) {
        collectConstructor(
            modules, moduleClass, selfCallConstructor.getKey(), selfCallConstructor.getValue());
      }
    }
  }

  private static String getModuleNameForConstructorPrefix(
      Class<?> moduleClass, Map<String, SkylarkModuleDoc> modules) {
    if (moduleClass.isAnnotationPresent(SkylarkModule.class)) {
      String moduleName = moduleClass.getAnnotation(SkylarkModule.class).name();
      SkylarkModuleDoc moduleDoc = Preconditions.checkNotNull(modules.get(moduleName));

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
