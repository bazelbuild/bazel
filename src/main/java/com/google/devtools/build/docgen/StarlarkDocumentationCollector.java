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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.docgen.StarlarkDocumentationProcessor.Category;
import com.google.devtools.build.docgen.annot.GlobalMethods;
import com.google.devtools.build.docgen.annot.GlobalMethods.Environment;
import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.docgen.starlark.StarlarkBuiltinDoc;
import com.google.devtools.build.docgen.starlark.StarlarkConstructorMethodDoc;
import com.google.devtools.build.docgen.starlark.StarlarkDocExpander;
import com.google.devtools.build.docgen.starlark.StarlarkDocPage;
import com.google.devtools.build.docgen.starlark.StarlarkGlobalsDoc;
import com.google.devtools.build.docgen.starlark.StarlarkJavaMethodDoc;
import com.google.devtools.build.lib.util.Classpath;
import com.google.devtools.build.lib.util.Classpath.ClassPathException;
import java.lang.reflect.Method;
import java.text.Collator;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import net.starlark.java.annot.StarlarkAnnotations;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;

/** A helper class that collects Starlark module documentation. */
final class StarlarkDocumentationCollector {
  private StarlarkDocumentationCollector() {}

  private static ImmutableMap<Category, ImmutableList<StarlarkDocPage>> all;

  /** Applies {@link #collectDocPages} to all Bazel and Starlark classes. */
  static synchronized ImmutableMap<Category, ImmutableList<StarlarkDocPage>> getAllDocPages(
      StarlarkDocExpander expander) throws ClassPathException {
    if (all == null) {
      all =
          collectDocPages(
              Iterables.concat(
                  /*Bazel*/ Classpath.findClasses("com/google/devtools/build"),
                  /*Starlark*/ Classpath.findClasses("net/starlark/java")),
              expander);
    }
    return all;
  }

  /**
   * Collects the documentation for all Starlark modules comprised of the given classes and returns
   * a map from the name of each Starlark module to its documentation.
   */
  static ImmutableMap<Category, ImmutableList<StarlarkDocPage>> collectDocPages(
      Iterable<Class<?>> classes, StarlarkDocExpander expander) {
    Map<Category, Map<String, StarlarkDocPage>> pages = new EnumMap<>(Category.class);
    for (Category category : Category.values()) {
      pages.put(category, new HashMap<>());
    }

    // 1. Add all classes/interfaces annotated with @StarlarkBuiltin with documented = true.
    for (Class<?> candidateClass : classes) {
      collectStarlarkBuiltin(candidateClass, pages, expander);
    }

    // 2. Add all object methods and global functions.
    //
    //    Also, explicitly process the Starlark interpreter's MethodLibrary
    //    class, which defines None, len, range, etc.
    //    TODO(adonovan): do this without peeking into the implementation,
    //    e.g. by looking at Starlark.UNIVERSE, something like this:
    //
    //    for (Map<String, Object> e : Starlark.UNIVERSE.entrySet()) {
    //      if (e.getValue() instanceof BuiltinFunction) {
    //        BuiltinFunction fn = (BuiltinFunction) e.getValue();
    //        topLevelModuleDoc.addMethod(
    //          new StarlarkJavaMethodDoc("", fn.getJavaMethod(), fn.getAnnotation(), expander));
    //      }
    //    }
    //
    //    Note that BuiltinFunction doesn't actually have getJavaMethod.
    //
    for (Class<?> candidateClass : classes) {
      collectBuiltinMethods(candidateClass, pages, expander);
      collectGlobalMethods(candidateClass, pages, expander);
    }

    // 3. Add all constructors.
    for (Class<?> candidateClass : classes) {
      collectConstructorMethods(candidateClass, pages, expander);
    }

    return ImmutableMap.copyOf(
        Maps.transformValues(
            pages,
            pagesInCategory ->
                ImmutableList.sortedCopyOf(
                    Comparator.comparing(
                        StarlarkDocPage::getTitle, Collator.getInstance(Locale.US)),
                    pagesInCategory.values())));
  }

  /**
   * Adds a single {@link StarlarkDocPage} entry to {@code pages} representing the given {@code
   * builtinClass}, if it is a documented builtin.
   */
  private static void collectStarlarkBuiltin(
      Class<?> builtinClass,
      Map<Category, Map<String, StarlarkDocPage>> pages,
      StarlarkDocExpander expander) {
    StarlarkBuiltin starlarkBuiltin = builtinClass.getAnnotation(StarlarkBuiltin.class);
    if (starlarkBuiltin == null || !starlarkBuiltin.documented()) {
      return;
    }

    Map<String, StarlarkDocPage> pagesInCategory = pages.get(Category.of(starlarkBuiltin));
    StarlarkDocPage existingPage = pagesInCategory.get(starlarkBuiltin.name());
    if (existingPage == null) {
      pagesInCategory.put(
          starlarkBuiltin.name(), new StarlarkBuiltinDoc(starlarkBuiltin, builtinClass, expander));
      return;
    }

    // Handle a strange corner-case: If builtinClass has a subclass which is also
    // annotated with @StarlarkBuiltin with the same name, and also has the same
    // docstring, then the subclass takes precedence.
    // (This is useful if one class is the "common" one with stable methods, and its subclass is
    // an experimental class that also supports all stable methods.)
    Preconditions.checkState(
        existingPage instanceof StarlarkBuiltinDoc,
        "the same name %s is assigned to both a global method environment and a builtin type",
        starlarkBuiltin.name());
    Class<?> clazz = ((StarlarkBuiltinDoc) existingPage).getClassObject();
    validateCompatibleBuiltins(clazz, builtinClass);

    if (clazz.isAssignableFrom(builtinClass)) {
      // The new builtin is a subclass of the old builtin, so use the subclass.
      pagesInCategory.put(
          starlarkBuiltin.name(), new StarlarkBuiltinDoc(starlarkBuiltin, builtinClass, expander));
    }
  }

  /** Validate that it is acceptable that the given builtin classes with the same name co-exist. */
  private static void validateCompatibleBuiltins(Class<?> one, Class<?> two) {
    StarlarkBuiltin builtinOne = one.getAnnotation(StarlarkBuiltin.class);
    StarlarkBuiltin builtinTwo = two.getAnnotation(StarlarkBuiltin.class);
    if (one.isAssignableFrom(two) || two.isAssignableFrom(one)) {
      if (!builtinOne.doc().equals(builtinTwo.doc())) {
        throw new IllegalStateException(
            String.format(
                "%s and %s are related builtins but have mismatching documentation for '%s'",
                one, two, builtinOne.name()));
      }
    } else {
      throw new IllegalStateException(
          String.format(
              "%s and %s are unrelated builtins with documentation for '%s'",
              one, two, builtinOne.name()));
    }
  }

  private static void collectBuiltinMethods(
      Class<?> builtinClass,
      Map<Category, Map<String, StarlarkDocPage>> pages,
      StarlarkDocExpander expander) {
    StarlarkBuiltin starlarkBuiltin = builtinClass.getAnnotation(StarlarkBuiltin.class);

    if (starlarkBuiltin == null || !starlarkBuiltin.documented()) {
      return;
    }
    StarlarkBuiltinDoc builtinDoc =
        (StarlarkBuiltinDoc) pages.get(Category.of(starlarkBuiltin)).get(starlarkBuiltin.name());

    if (builtinClass != builtinDoc.getClassObject()) {
      return;
    }
    for (Map.Entry<Method, StarlarkMethod> entry :
        Starlark.getMethodAnnotations(builtinClass).entrySet()) {
      // Collect methods that aren't directly constructors (i.e. have the @StarlarkConstructor
      // annotation).
      if (entry.getKey().isAnnotationPresent(StarlarkConstructor.class)) {
        continue;
      }
      Method javaMethod = entry.getKey();
      StarlarkMethod starlarkMethod = entry.getValue();
      // Struct fields that return a type that has @StarlarkConstructor are a bit special:
      // they're visited here because they're seen as an attribute of the module, but act more
      // like a reference to the type they construct.
      // TODO(wyv): does this actually happen???
      if (starlarkMethod.structField()) {
        Method selfCall =
            Starlark.getSelfCallMethod(StarlarkSemantics.DEFAULT, javaMethod.getReturnType());
        if (selfCall != null && selfCall.isAnnotationPresent(StarlarkConstructor.class)) {
          javaMethod = selfCall;
        }
      }
      builtinDoc.addMethod(
          new StarlarkJavaMethodDoc(builtinDoc.getName(), javaMethod, starlarkMethod, expander));
    }
  }

  /**
   * Adds {@link StarlarkJavaMethodDoc} entries to the top level module, one for
   * each @StarlarkMethod method defined in the given @GlobalMethods class {@code clazz}.
   */
  private static void collectGlobalMethods(
      Class<?> clazz,
      Map<Category, Map<String, StarlarkDocPage>> pages,
      StarlarkDocExpander expander) {
    GlobalMethods globalMethods = clazz.getAnnotation(GlobalMethods.class);

    if (globalMethods == null && !clazz.getName().equals("net.starlark.java.eval.MethodLibrary")) {
      return;
    }

    Environment[] environments =
        globalMethods == null ? new Environment[] {Environment.ALL} : globalMethods.environment();
    for (Environment environment : environments) {
      StarlarkDocPage page =
          pages
              .get(Category.GLOBAL_FUNCTION)
              .computeIfAbsent(
                  environment.getTitle(), title -> new StarlarkGlobalsDoc(environment, expander));
      for (Map.Entry<Method, StarlarkMethod> entry :
          Starlark.getMethodAnnotations(clazz).entrySet()) {
        // Only add non-constructor global library methods. Constructors are added later.
        // TODO(wyv): add a redirect instead
        if (!entry.getKey().isAnnotationPresent(StarlarkConstructor.class)) {
          page.addMethod(new StarlarkJavaMethodDoc("", entry.getKey(), entry.getValue(), expander));
        }
      }
    }
  }

  private static void collectConstructor(
      Map<Category, Map<String, StarlarkDocPage>> pages,
      Method method,
      StarlarkDocExpander expander) {
    if (!method.isAnnotationPresent(StarlarkConstructor.class)) {
      return;
    }

    StarlarkBuiltin starlarkBuiltin =
        StarlarkAnnotations.getStarlarkBuiltin(method.getReturnType());
    if (starlarkBuiltin == null || !starlarkBuiltin.documented()) {
      // The class of the constructed object type has no documentation, so no place to add
      // constructor information.
      return;
    }
    StarlarkMethod methodAnnot =
        Preconditions.checkNotNull(method.getAnnotation(StarlarkMethod.class));
    StarlarkDocPage doc = pages.get(Category.of(starlarkBuiltin)).get(starlarkBuiltin.name());
    doc.setConstructor(
        new StarlarkConstructorMethodDoc(starlarkBuiltin.name(), method, methodAnnot, expander));
  }

  /**
   * Collect two types of constructor methods:
   *
   * <p>1. The single method with selfCall=true and @StarlarkConstructor (if present)
   *
   * <p>2. Any methods annotated with @StarlarkConstructor
   *
   * <p>Structfield methods that return an object which itself has selfCall=true
   * and @StarlarkConstructor are *not* collected here (collectModuleMethods does that). (For
   * example, supposed Foo has a structfield method named 'Bar', which refers to the Bar type. In
   * Foo's doc, we describe Foo.Bar as an attribute of type Bar and link to the canonical Bar type
   * documentation)
   */
  private static void collectConstructorMethods(
      Class<?> clazz,
      Map<Category, Map<String, StarlarkDocPage>> pages,
      StarlarkDocExpander expander) {
    if (!clazz.isAnnotationPresent(StarlarkBuiltin.class)
        && !clazz.isAnnotationPresent(GlobalMethods.class)) {
      return;
    }
    Method selfCall = Starlark.getSelfCallMethod(StarlarkSemantics.DEFAULT, clazz);
    if (selfCall != null) {
      collectConstructor(pages, selfCall, expander);
    }

    for (Method method : Starlark.getMethodAnnotations(clazz).keySet()) {
      collectConstructor(pages, method, expander);
    }
  }
}
