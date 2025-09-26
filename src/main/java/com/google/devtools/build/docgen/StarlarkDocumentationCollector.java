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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.docgen.StarlarkDocumentationProcessor.Category;
import com.google.devtools.build.docgen.annot.GlobalMethods;
import com.google.devtools.build.docgen.annot.GlobalMethods.Environment;
import com.google.devtools.build.docgen.annot.StarlarkConstructor;
import com.google.devtools.build.docgen.starlark.AnnotStarlarkBuiltinDoc;
import com.google.devtools.build.docgen.starlark.AnnotStarlarkConstructorMethodDoc;
import com.google.devtools.build.docgen.starlark.AnnotStarlarkOrdinaryMethodDoc;
import com.google.devtools.build.docgen.starlark.StardocProtoFunctionDoc;
import com.google.devtools.build.docgen.starlark.StardocProtoProviderDocPage;
import com.google.devtools.build.docgen.starlark.StardocProtoStructDocPage;
import com.google.devtools.build.docgen.starlark.StarlarkDocExpander;
import com.google.devtools.build.docgen.starlark.StarlarkDocPage;
import com.google.devtools.build.docgen.starlark.StarlarkGlobalsDoc;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AspectInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.MacroInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleExtensionInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RepositoryRuleInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RuleInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.StarlarkFunctionInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.StarlarkOtherSymbolInfo;
import com.google.devtools.build.lib.util.Classpath;
import com.google.devtools.build.lib.util.Classpath.ClassPathException;
import com.google.protobuf.ExtensionRegistry;
import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Method;
import java.text.Collator;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import net.starlark.java.annot.StarlarkAnnotations;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * A helper class that collects Starlark module documentation.
 *
 * <p>The documentation comes from {@link StarlarkBuiltin} annotations in Java code or from Stardoc
 * protos produced (via {@code starlark_doc_extract} from specially-structured .bzl files serving as
 * entry points for Starlark APIs. Such an entry point .bzl file is expected to contain only the
 * following documentable entities (whose names must be unique across all .bzl files being
 * processed):
 *
 * <ul>
 *   <li>Providers, defined at global scope;
 *   <li>Structs, defined at global scope, documented using {@code #:}-prefixed doc comments, and
 *       containing only function members.
 * </ul>
 *
 * <p>Notably, .bzl files from which Build Encyclopedia content is extracted have a different,
 * incompatible structure.
 */
final class StarlarkDocumentationCollector {
  private StarlarkDocumentationCollector() {}

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private static ImmutableMap<Category, ImmutableList<StarlarkDocPage>> all;

  /** Applies {@link #collectDocPages} to all Bazel and Starlark classes. */
  static synchronized ImmutableMap<Category, ImmutableList<StarlarkDocPage>> getAllDocPages(
      StarlarkDocExpander expander, ImmutableList<String> apiStardocProtos)
      throws ClassPathException, IOException {
    if (all == null) {
      ImmutableList.Builder<ModuleInfo> parsedApiStardocProtos = ImmutableList.builder();
      for (String filename : apiStardocProtos) {
        parsedApiStardocProtos.add(
            ModuleInfo.parseFrom(
                new FileInputStream(filename), ExtensionRegistry.getEmptyRegistry()));
      }
      all =
          collectDocPages(
              expander,
              Iterables.concat(
                  /*Bazel*/ Classpath.findClasses("com/google/devtools/build"),
                  /*Starlark*/ Classpath.findClasses("net/starlark/java")),
              parsedApiStardocProtos.build());
    }
    return all;
  }

  /**
   * Collects the documentation for all Starlark modules comprised of the given classes and returns
   * a map from the name of each Starlark module to its documentation.
   */
  static ImmutableMap<Category, ImmutableList<StarlarkDocPage>> collectDocPages(
      StarlarkDocExpander expander,
      Iterable<Class<?>> classes,
      ImmutableList<ModuleInfo> apiStardocProtos) {
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

    // 4. Add docs from .bzl files.
    HashMap<String, ModuleInfo> bzlStructPages = new HashMap<>();
    for (ModuleInfo moduleInfo : apiStardocProtos) {
      collectFromStardocProto(moduleInfo, pages, bzlStructPages, expander);
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
          starlarkBuiltin.name(),
          new AnnotStarlarkBuiltinDoc(starlarkBuiltin, builtinClass, expander));
      return;
    }

    // Handle a strange corner-case: If builtinClass has a subclass which is also
    // annotated with @StarlarkBuiltin with the same name, and also has the same
    // docstring, then the subclass takes precedence.
    // (This is useful if one class is the "common" one with stable methods, and its subclass is
    // an experimental class that also supports all stable methods.)
    Preconditions.checkState(
        existingPage instanceof AnnotStarlarkBuiltinDoc,
        "the same name %s is assigned to both a global method environment and a builtin type",
        starlarkBuiltin.name());
    Class<?> clazz = ((AnnotStarlarkBuiltinDoc) existingPage).getClassObject();
    validateCompatibleBuiltins(clazz, builtinClass);

    if (clazz.isAssignableFrom(builtinClass)) {
      // The new builtin is a subclass of the old builtin, so use the subclass.
      pagesInCategory.put(
          starlarkBuiltin.name(),
          new AnnotStarlarkBuiltinDoc(starlarkBuiltin, builtinClass, expander));
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
    AnnotStarlarkBuiltinDoc builtinDoc =
        (AnnotStarlarkBuiltinDoc)
            pages.get(Category.of(starlarkBuiltin)).get(starlarkBuiltin.name());

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
      builtinDoc.addMember(
          new AnnotStarlarkOrdinaryMethodDoc(
              builtinDoc.getName(), javaMethod, starlarkMethod, expander));
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
          page.addMember(
              new AnnotStarlarkOrdinaryMethodDoc("", entry.getKey(), entry.getValue(), expander));
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
        new AnnotStarlarkConstructorMethodDoc(
            starlarkBuiltin.name(), method, methodAnnot, expander));
  }

  /**
   * Parses a Starlark API proto to produce {@link StardocProtoStructDocPage} and {@link
   * StardocProtoProviderDocPage} pages, inserting them into the appropriate categories of {@code
   * pages}.
   *
   * @param moduleInfo a Stardoc proto for a .bzl file serving as an entry point for Starlark APIs
   * @param pages the categorized map of documentation pages; added to by this method
   * @param bzlStructPages a map from names of structs whose documentation has been collected to the
   *     Stardoc protos defining them; added to by this method
   * @param expander the expander to use for links
   */
  private static void collectFromStardocProto(
      ModuleInfo moduleInfo,
      Map<Category, Map<String, StarlarkDocPage>> pages,
      Map<String, ModuleInfo> bzlStructPages,
      StarlarkDocExpander expander) {
    // For now, support only the following:
    // - structs containing only functions (classified as TOP_LEVEL_MODULE)
    // - providers not contained in a struct (classified as PROVIDER)
    Map<String, StarlarkDocPage> pagesInCategory = pages.get(Category.TOP_LEVEL_MODULE);
    for (StarlarkOtherSymbolInfo symbolInfo : moduleInfo.getStarlarkOtherSymbolInfoList()) {
      if (symbolInfo.getTypeName().equals("struct")) {
        String structName = symbolInfo.getName();
        if (structName.contains(".")) {
          // Skip nested structs.
          continue;
        }
        if (pagesInCategory.containsKey(structName)) {
          checkState(
              !bzlStructPages.containsKey(structName),
              "Conflicting documentation for struct '%s' defined in Starlark files %s and %s",
              structName,
              moduleInfo.getFile(),
              bzlStructPages.get(structName).getFile());
          logger.atWarning().log(
              "Documentation for struct %s defined in %s overrides previously-seen documentation"
                  + " for module %s implemented in Java",
              structName, moduleInfo.getFile(), structName);
        }
        pagesInCategory.put(
            structName, new StardocProtoStructDocPage(expander, moduleInfo, symbolInfo));
      }
    }

    for (StarlarkFunctionInfo functionInfo : moduleInfo.getFuncInfoList()) {
      String functionName = functionInfo.getFunctionName();
      checkState(
          functionName.contains("."),
          "Function %s defined in %s must be namespaced inside a struct",
          functionName,
          moduleInfo.getFile());
      String structName = Splitter.on('.').splitToList(functionName).getFirst();
      checkState(
          moduleInfo.getStarlarkOtherSymbolInfoList().stream()
              .anyMatch(symbolInfo -> symbolInfo.getName().equals(structName)),
          "Struct %s defined in %s must be documented with '#:'-prefixed doc comments",
          structName,
          moduleInfo.getFile());
      StarlarkDocPage page = checkNotNull(pagesInCategory.get(structName));
      page.addMember(new StardocProtoFunctionDoc(expander, moduleInfo, structName, functionInfo));
    }

    for (ProviderInfo providerInfo : moduleInfo.getProviderInfoList()) {
      String providerName = providerInfo.getProviderName();
      // TODO(arostovtsev): Add a stub for namespaced providers which will be resolved to a link to
      // the real provider doc.
      checkState(
          !providerName.contains("."),
          "Provider %s in %s: provider members in structs are not supported yet",
          providerName,
          moduleInfo.getFile());
      pages
          .get(Category.PROVIDER)
          .put(providerName, new StardocProtoProviderDocPage(expander, moduleInfo, providerInfo));
    }

    // TODO(arostovtsev): What about other types of members in structs? Need changes to
    // starlark_doc_extract to check for their presence.
    verifyDoNotExist(
        moduleInfo,
        "aspects",
        moduleInfo.getAspectInfoList().stream()
            .map(AspectInfo::getAspectName)
            .collect(toImmutableList()));
    verifyDoNotExist(
        moduleInfo,
        "macros",
        moduleInfo.getMacroInfoList().stream()
            .map(MacroInfo::getMacroName)
            .collect(toImmutableList()));
    verifyDoNotExist(
        moduleInfo,
        "module extesions",
        moduleInfo.getModuleExtensionInfoList().stream()
            .map(ModuleExtensionInfo::getExtensionName)
            .collect(toImmutableList()));
    verifyDoNotExist(
        moduleInfo,
        "repository rules",
        moduleInfo.getRepositoryRuleInfoList().stream()
            .map(RepositoryRuleInfo::getRuleName)
            .collect(toImmutableList()));
    verifyDoNotExist(
        moduleInfo,
        "rules",
        moduleInfo.getRuleInfoList().stream()
            .map(RuleInfo::getRuleName)
            .collect(toImmutableList()));
  }

  private static void verifyDoNotExist(ModuleInfo moduleInfo, String what, List<String> badNames) {
    checkState(
        badNames.isEmpty(),
        "Starlark and BUILD language API entry point %s is expected not to contain %s;"
            + " found %s",
        moduleInfo.getFile(),
        what,
        badNames);
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
