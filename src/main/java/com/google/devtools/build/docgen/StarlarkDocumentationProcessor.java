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
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.docgen.starlark.StarlarkBuiltinDoc;
import com.google.devtools.build.docgen.starlark.StarlarkDocUtils;
import com.google.devtools.build.docgen.starlark.StarlarkMethodDoc;
import com.google.devtools.build.lib.util.Classpath;
import com.google.devtools.build.lib.util.Classpath.ClassPathException;
import java.io.File;
import java.io.IOException;
import java.text.Collator;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import net.starlark.java.annot.StarlarkDocumentationCategory;

/** A class to assemble documentation for Starlark. */
public final class StarlarkDocumentationProcessor {

  private static final ImmutableList<StarlarkDocumentationCategory> GLOBAL_CATEGORIES =
      ImmutableList.<StarlarkDocumentationCategory>of(
          StarlarkDocumentationCategory.NONE, StarlarkDocumentationCategory.TOP_LEVEL_TYPE);

  // Common prefix of packages that may contain Starlark modules.
  @VisibleForTesting static final String MODULES_PACKAGE_PREFIX = "com/google/devtools/build";

  private StarlarkDocumentationProcessor() {}

  /** Generates the Starlark documentation to the given output directory. */
  public static void generateDocumentation(String outputDir, String... args)
      throws IOException, ClassPathException {
    parseOptions(args);

    Map<String, StarlarkBuiltinDoc> modules =
        StarlarkDocumentationCollector.collectModules(
            Classpath.findClasses(MODULES_PACKAGE_PREFIX));

    // Generate the top level module first in the doc
    StarlarkBuiltinDoc topLevelModule =
        modules.remove(StarlarkDocumentationCollector.getTopLevelModule().name());
    writePage(outputDir, topLevelModule);

    // Use a LinkedHashMap to preserve ordering of categories, as the output iterates over
    // this map's entry set to determine category ordering.
    Map<StarlarkDocumentationCategory, List<StarlarkBuiltinDoc>> modulesByCategory =
        new LinkedHashMap<>();
    for (StarlarkDocumentationCategory c : StarlarkDocumentationCategory.values()) {
      modulesByCategory.put(c, new ArrayList<StarlarkBuiltinDoc>());
    }

    modulesByCategory.get(topLevelModule.getAnnotation().category()).add(topLevelModule);

    for (StarlarkBuiltinDoc module : modules.values()) {
      if (module.getAnnotation().documented()) {
        writePage(outputDir, module);
        modulesByCategory.get(module.getAnnotation().category()).add(module);
      }
    }
    Collator us = Collator.getInstance(Locale.US);
    for (List<StarlarkBuiltinDoc> module : modulesByCategory.values()) {
      Collections.sort(module, (doc1, doc2) -> us.compare(doc1.getTitle(), doc2.getTitle()));
    }
    writeCategoryPage(
        StarlarkDocumentationCategory.CONFIGURATION_FRAGMENT, outputDir, modulesByCategory);
    writeCategoryPage(StarlarkDocumentationCategory.BUILTIN, outputDir, modulesByCategory);
    writeCategoryPage(StarlarkDocumentationCategory.PROVIDER, outputDir, modulesByCategory);
    writeNavPage(outputDir, modulesByCategory.get(StarlarkDocumentationCategory.TOP_LEVEL_TYPE));

    // In the code, there are two SkylarkModuleCategory instances that have no heading:
    // TOP_LEVEL_TYPE and NONE.

    // TOP_LEVEL_TYPE also contains the "global" module.
    // We remove both categories and the "global" module from the map and display them manually:
    // - Methods in the "global" module are displayed under "Global Methods and Constants".
    // - Modules in both categories are displayed under "Global Modules" (except for the global
    // module itself).
    List<String> globalFunctions = new ArrayList<>();
    List<String> globalConstants = new ArrayList<>();
    StarlarkBuiltinDoc globalModule = findGlobalModule(modulesByCategory);
    for (StarlarkMethodDoc method : globalModule.getMethods()) {
      if (method.documented()) {
        if (method.isCallable()) {
          globalFunctions.add(method.getName());
        } else {
          globalConstants.add(method.getName());
        }
      }
    }

    List<StarlarkBuiltinDoc> globalModules = new ArrayList<>();
    for (StarlarkDocumentationCategory globalCategory : GLOBAL_CATEGORIES) {
      for (StarlarkBuiltinDoc module : modulesByCategory.remove(globalCategory)) {
        if (!module.getName().equals(globalModule.getName())) {
          globalModules.add(module);
        }
      }
    }

    Collections.sort(globalModules, (doc1, doc2) -> us.compare(doc1.getName(), doc2.getName()));
    writeOverviewPage(
        outputDir,
        globalModule.getName(),
        globalFunctions,
        globalConstants,
        globalModules,
        modulesByCategory);
  }

  private static StarlarkBuiltinDoc findGlobalModule(
      Map<StarlarkDocumentationCategory, List<StarlarkBuiltinDoc>> modulesByCategory) {
    List<StarlarkBuiltinDoc> topLevelModules =
        modulesByCategory.get(StarlarkDocumentationCategory.TOP_LEVEL_TYPE);
    String globalModuleName = StarlarkDocumentationCollector.getTopLevelModule().name();
    for (StarlarkBuiltinDoc module : topLevelModules) {
      if (module.getName().equals(globalModuleName)) {
        return module;
      }
    }

    throw new IllegalStateException("No globals module in the top level category.");
  }

  private static void writePage(String outputDir, StarlarkBuiltinDoc module) throws IOException {
    File skylarkDocPath = new File(outputDir + "/" + module.getName() + ".html");
    Page page = TemplateEngine.newPage(DocgenConsts.STARLARK_LIBRARY_TEMPLATE);
    page.add("module", module);
    page.write(skylarkDocPath);
  }

  private static void writeCategoryPage(
      StarlarkDocumentationCategory category,
      String outputDir,
      Map<StarlarkDocumentationCategory, List<StarlarkBuiltinDoc>> modules)
      throws IOException {
    File skylarkDocPath = new File(String.format("%s/skylark-%s.html",
        outputDir, category.getTemplateIdentifier()));
    Page page = TemplateEngine.newPage(DocgenConsts.STARLARK_MODULE_CATEGORY_TEMPLATE);
    page.add("category", category);
    page.add("modules", modules.get(category));
    page.add("description", StarlarkDocUtils.substituteVariables(category.getDescription()));
    page.write(skylarkDocPath);
  }

  private static void writeNavPage(String outputDir, List<StarlarkBuiltinDoc> navModules)
      throws IOException {
    File navFile = new File(outputDir + "/skylark-nav.html");
    Page page = TemplateEngine.newPage(DocgenConsts.STARLARK_NAV_TEMPLATE);
    page.add("modules", navModules);
    page.write(navFile);
  }

  private static void writeOverviewPage(
      String outputDir,
      String globalModuleName,
      List<String> globalFunctions,
      List<String> globalConstants,
      List<StarlarkBuiltinDoc> globalModules,
      Map<StarlarkDocumentationCategory, List<StarlarkBuiltinDoc>> modulesPerCategory)
      throws IOException {
    File skylarkDocPath = new File(outputDir + "/skylark-overview.html");
    Page page = TemplateEngine.newPage(DocgenConsts.STARLARK_OVERVIEW_TEMPLATE);
    page.add("global_name", globalModuleName);
    page.add("global_functions", globalFunctions);
    page.add("global_constants", globalConstants);
    page.add("global_modules", globalModules);
    page.add("modules", modulesPerCategory);
    page.write(skylarkDocPath);
  }

  private static void parseOptions(String... args) {
    for (String arg : args) {
      if (arg.startsWith("--be_root=")) {
        DocgenConsts.BeDocsRoot = arg.split("--be_root=", 2)[1];
      }
      if (arg.startsWith("--doc_extension=")) {
        DocgenConsts.documentationExtension = arg.split("--doc_extension=", 2)[1];
      }
    }
  }
}
