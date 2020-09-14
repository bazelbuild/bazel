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
import com.google.common.collect.Iterables;
import com.google.devtools.build.docgen.starlark.StarlarkBuiltinDoc;
import com.google.devtools.build.docgen.starlark.StarlarkConstructorMethodDoc;
import com.google.devtools.build.docgen.starlark.StarlarkJavaMethodDoc;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import com.google.devtools.build.lib.util.Classpath;
import com.google.devtools.build.lib.util.Classpath.ClassPathException;
import java.lang.reflect.Method;
import java.util.Map;
import java.util.TreeMap;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkConstructor;
import net.starlark.java.annot.StarlarkDocumentationCategory;
import net.starlark.java.annot.StarlarkGlobalLibrary;
import net.starlark.java.annot.StarlarkMethod;

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

  private static ImmutableMap<String, StarlarkBuiltinDoc> all;

  /** Applies {@link #collectModules} to all Bazel and Starlark classes. */
  static synchronized ImmutableMap<String, StarlarkBuiltinDoc> getAllModules()
      throws ClassPathException {
    if (all == null) {
      all =
          collectModules(
              Iterables.concat(
                  Classpath.findClasses("com/google/devtools/build"), // Bazel
                  Classpath.findClasses("net/starlark/java"))); // Starlark
    }
    return all;
  }

  /**
   * Collects the documentation for all Starlark modules comprised of the given classes and returns
   * a map from the name of each Starlark module to its documentation.
   */
  static ImmutableMap<String, StarlarkBuiltinDoc> collectModules(Iterable<Class<?>> classes) {
    // Force class loading of com.google.devtools.build.lib.syntax.Starlark before we do any of our
    // own processing. Otherwise, we're in trouble since com.google.devtools.build.lib.syntax.Dict
    // happens to be the first class on our classpath that we proccess via #collectModuleMethods,
    // but that entails a logical cycle in
    // com.google.devtools.build.lib.syntax.CallUtils#getCacheValue.
    // TODO(b/161479826): Address this in a less hacky manner.
    @SuppressWarnings("unused")
    Object forceClassLoading = Starlark.UNIVERSE;

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
        collectStarlarkModule(candidateClass, modules);
      }
    }

    // 2. Add all object methods and global functions.
    for (Class<?> candidateClass : classes) {
      if (candidateClass.isAnnotationPresent(StarlarkBuiltin.class)) {
        collectModuleMethods(candidateClass, modules);
      }
      if (candidateClass.isAnnotationPresent(StarlarkGlobalLibrary.class)) {
        collectGlobalLibraryMethods(candidateClass, modules);
      }
    }

    // 3. Add all constructors.
    for (Class<?> candidateClass : classes) {
      if (candidateClass.isAnnotationPresent(StarlarkBuiltin.class)
          || candidateClass.isAnnotationPresent(StarlarkGlobalLibrary.class)) {
        collectConstructorMethods(candidateClass, modules);
      }
    }

    return ImmutableMap.copyOf(modules);
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
  private static void collectStarlarkModule(
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
        for (Map.Entry<Method, StarlarkMethod> entry :
            Starlark.getAnnotatedMethods(moduleClass).entrySet()) {
          // Only collect methods not annotated with @StarlarkConstructor.
          // Methods with @StarlarkConstructor are added later.
          if (!entry.getKey().isAnnotationPresent(StarlarkConstructor.class)) {
            moduleDoc.addMethod(
                new StarlarkJavaMethodDoc(moduleDoc.getName(), entry.getKey(), entry.getValue()));
          }
        }
      }
    }
  }

  @Nullable
  private static Method getSelfCallConstructorMethod(Class<?> objectClass) {
    Method selfCallMethod = Starlark.getSelfCallMethod(StarlarkSemantics.DEFAULT, objectClass);
    if (selfCallMethod != null && selfCallMethod.isAnnotationPresent(StarlarkConstructor.class)) {
      return selfCallMethod;
    }
    return null;
  }

  /**
   * Adds {@link StarlarkJavaMethodDoc} entries to the top level module, one for
   * each @StarlarkMethod method defined in the given @StarlarkGlobalLibrary class {@code
   * moduleClass}.
   */
  private static void collectGlobalLibraryMethods(
      Class<?> moduleClass, Map<String, StarlarkBuiltinDoc> modules) {
    Preconditions.checkArgument(moduleClass.isAnnotationPresent(StarlarkGlobalLibrary.class));
    StarlarkBuiltinDoc topLevelModuleDoc = getTopLevelModuleDoc(modules);

    for (Map.Entry<Method, StarlarkMethod> entry :
        Starlark.getAnnotatedMethods(moduleClass).entrySet()) {
      // Only add non-constructor global library methods. Constructors are added later.
      if (!entry.getKey().isAnnotationPresent(StarlarkConstructor.class)) {
        topLevelModuleDoc.addMethod(
            new StarlarkJavaMethodDoc("", entry.getKey(), entry.getValue()));
      }
    }
  }

  private static void collectConstructor(
      Map<String, StarlarkBuiltinDoc> modules, Class<?> moduleClass, Method method) {
    StarlarkConstructor constructorAnnotation =
        Preconditions.checkNotNull(method.getAnnotation(StarlarkConstructor.class));
    StarlarkMethod callable =
        Preconditions.checkNotNull(method.getAnnotation(StarlarkMethod.class));
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
   * <p>1. Methods that are annotated with @StarlarkConstructor.
   *
   * <p>2. Structfield methods that return an object which itself has a method with selfCall = true,
   * and is annotated with @StarlarkConstructor. (For example, suppose Foo has a structfield method
   * 'bar'. If Foo.bar is itself callable, and is a constructor, then Foo.bar() should be treated
   * like a constructor method.)
   */
  private static void collectConstructorMethods(
      Class<?> moduleClass, Map<String, StarlarkBuiltinDoc> modules) {
    Method selfCallConstructor = getSelfCallConstructorMethod(moduleClass);
    if (selfCallConstructor != null) {
      collectConstructor(modules, moduleClass, selfCallConstructor);
    }

    for (Map.Entry<Method, StarlarkMethod> entry :
        Starlark.getAnnotatedMethods(moduleClass).entrySet()) {
      if (entry.getKey().isAnnotationPresent(StarlarkConstructor.class)) {
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
    } else if (moduleClass.isAnnotationPresent(StarlarkGlobalLibrary.class)) {
      return "";
    } else {
      throw new IllegalArgumentException(moduleClass + " has no valid annotation");
    }
  }

  private static String getFullyQualifiedName(String objectName, StarlarkMethod callable) {
    String objectDotExpressionPrefix = objectName.isEmpty() ? "" : objectName + ".";
    String methodName = callable.name();
    return objectDotExpressionPrefix + methodName;
  }
}
