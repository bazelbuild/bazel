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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.docgen.starlark.StarlarkBuiltinDoc;
import com.google.devtools.build.docgen.starlark.StarlarkDocExpander;
import com.google.devtools.build.docgen.starlark.StarlarkMethodDoc;
import com.google.devtools.build.lib.util.Classpath.ClassPathException;
import java.io.File;
import java.io.IOException;
import java.text.Collator;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import net.starlark.java.annot.StarlarkBuiltin;

/** A class to assemble documentation for Starlark. */
public final class StarlarkDocumentationProcessor {

  private static final ImmutableList<Category> GLOBAL_CATEGORIES =
      ImmutableList.<Category>of(Category.NONE, Category.TOP_LEVEL_TYPE);

  private StarlarkDocumentationProcessor() {}

  /** Generates the Starlark documentation to the given output directory. */
  public static void generateDocumentation(String outputDir, String... args)
      throws IOException, ClassPathException {
    Map<String, String> options = parseOptions(args);

    String docsRoot = options.get("--starlark_docs_root");
    if (docsRoot != null) {
      DocgenConsts.starlarkDocsRoot = docsRoot;
    }

    String linkMapPath = options.get("--link_map_path");
    if (linkMapPath == null) {
      throw new IllegalArgumentException("Required option '--link_map_path' is missing.");
    }

    DocLinkMap linkMap = DocLinkMap.createFromFile(linkMapPath);
    StarlarkDocExpander expander =
        new StarlarkDocExpander(new RuleLinkExpander(/*singlePage*/ false, linkMap));

    Map<String, StarlarkBuiltinDoc> modules =
        new TreeMap<>(StarlarkDocumentationCollector.getAllModules(expander));

    // Generate the top level module first in the doc
    StarlarkBuiltinDoc topLevelModule =
        modules.remove(StarlarkDocumentationCollector.getTopLevelModule().name());
    writePage(outputDir, topLevelModule);

    // Use a LinkedHashMap to preserve ordering of categories, as the output iterates over
    // this map's entry set to determine category ordering.
    Map<Category, List<StarlarkBuiltinDoc>> modulesByCategory = new LinkedHashMap<>();
    for (Category c : Category.values()) {
      modulesByCategory.put(c, new ArrayList<StarlarkBuiltinDoc>());
    }

    modulesByCategory.get(Category.of(topLevelModule.getAnnotation())).add(topLevelModule);

    for (StarlarkBuiltinDoc module : modules.values()) {
      if (module.getAnnotation().documented()) {
        writePage(outputDir, module);
        modulesByCategory.get(Category.of(module.getAnnotation())).add(module);
      }
    }
    Collator us = Collator.getInstance(Locale.US);
    for (List<StarlarkBuiltinDoc> module : modulesByCategory.values()) {
      Collections.sort(module, (doc1, doc2) -> us.compare(doc1.getTitle(), doc2.getTitle()));
    }
    writeCategoryPage(Category.CORE, outputDir, modulesByCategory, expander);
    writeCategoryPage(Category.CONFIGURATION_FRAGMENT, outputDir, modulesByCategory, expander);
    writeCategoryPage(Category.BUILTIN, outputDir, modulesByCategory, expander);
    writeCategoryPage(Category.PROVIDER, outputDir, modulesByCategory, expander);
    writeNavPage(outputDir, modulesByCategory.get(Category.TOP_LEVEL_TYPE));

    // In the code, there are two StarlarkModuleCategory instances that have no heading:
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
    for (Category globalCategory : GLOBAL_CATEGORIES) {
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
      Map<Category, List<StarlarkBuiltinDoc>> modulesByCategory) {
    List<StarlarkBuiltinDoc> topLevelModules = modulesByCategory.get(Category.TOP_LEVEL_TYPE);
    String globalModuleName = StarlarkDocumentationCollector.getTopLevelModule().name();
    for (StarlarkBuiltinDoc module : topLevelModules) {
      if (module.getName().equals(globalModuleName)) {
        return module;
      }
    }

    throw new IllegalStateException("No globals module in the top level category.");
  }

  private static void writePage(String outputDir, StarlarkBuiltinDoc module) throws IOException {
    File starlarkDocPath = new File(outputDir + "/" + module.getName() + ".html");
    Page page = TemplateEngine.newPage(DocgenConsts.STARLARK_LIBRARY_TEMPLATE);
    page.add("module", module);
    page.write(starlarkDocPath);
  }

  private static void writeCategoryPage(
      Category category,
      String outputDir,
      Map<Category, List<StarlarkBuiltinDoc>> modules,
      StarlarkDocExpander expander)
      throws IOException {
    File starlarkDocPath =
        new File(String.format("%s/starlark-%s.html", outputDir, category.getTemplateIdentifier()));
    Page page = TemplateEngine.newPage(DocgenConsts.STARLARK_MODULE_CATEGORY_TEMPLATE);
    page.add("category", category);
    page.add("modules", modules.get(category));
    page.add("description", expander.expand(category.description));
    page.write(starlarkDocPath);
  }

  private static void writeNavPage(String outputDir, List<StarlarkBuiltinDoc> navModules)
      throws IOException {
    File navFile = new File(outputDir + "/starlark-nav.html");
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
      Map<Category, List<StarlarkBuiltinDoc>> modulesPerCategory)
      throws IOException {
    File starlarkDocPath = new File(outputDir + "/starlark-overview.html");
    Page page = TemplateEngine.newPage(DocgenConsts.STARLARK_OVERVIEW_TEMPLATE);
    page.add("global_name", globalModuleName);
    page.add("global_functions", globalFunctions);
    page.add("global_constants", globalConstants);
    page.add("global_modules", globalModules);
    page.add("modules", modulesPerCategory);
    page.write(starlarkDocPath);
  }

  private static Map<String, String> parseOptions(String... args) {
    Map<String, String> options = new HashMap<>();
    for (String arg : args) {
      if (arg.startsWith("--")) {
        String[] parts = arg.split("=", 2);
        options.put(parts[0], parts.length > 1 ? parts[1] : null);
      }
    }
    return options;
  }

  /**
   * An enumeration of categories used to organize the API index. Instances of this class are
   * accessed by templates, using reflection.
   */
  public enum Category {
    CONFIGURATION_FRAGMENT(
        "Configuration Fragments",
        "Configuration fragments give rules access to "
            + "language-specific parts of <a href=\"configuration.html\">"
            + "configuration</a>. "
            + "<p>Rule implementations can get them using "
            + "<code><a href=\"ctx.html#fragments\">ctx."
            + "fragments</a>.<i>[fragment name]</i></code>"),

    PROVIDER(
        "Providers",
        "This section lists providers available on built-in rules. See the <a"
            + " href='$STARLARK_DOCS_ROOT/rules.html#providers'>Rules page</a> for more on"
            + " providers."),

    BUILTIN("Built-in Types", "This section lists types of Starlark objects."),

    // Used for top-level modules of functions in the global namespace. Such modules will always
    // be usable solely by accessing their members, via modulename.funcname() or
    // modulename.constantname.
    // Examples: attr, cc_common, config, java_common
    TOP_LEVEL_TYPE(null, null),

    CORE(
        "Core Starlark data types",
        "This section lists the data types of the <a"
            + " href='https://github.com/bazelbuild/starlark/blob/master/spec.md#built-in-constants-and-functions'>Starlark"
            + " core language</a>."),

    // Legacy uncategorized type; these are treated like TOP_LEVEL_TYPE in documentation.
    NONE(null, null);

    // Maps (essentially free-form) strings in annotations to permitted categories.
    static Category of(StarlarkBuiltin annot) {
      switch (annot.category()) {
        case DocCategory.CONFIGURATION_FRAGMENT:
          return CONFIGURATION_FRAGMENT;
        case DocCategory.PROVIDER:
          return PROVIDER;
        case DocCategory.BUILTIN:
          return BUILTIN;
        case DocCategory.TOP_LEVEL_TYPE:
          return TOP_LEVEL_TYPE;
        case DocCategory.NONE:
          return NONE;
        case "core": // interpreter built-ins (e.g. int)
        case "core.lib": // Starlark standard modules (e.g. json)
          return CORE;
        case "": // no annotation
          return TOP_LEVEL_TYPE;
        default:
          throw new IllegalStateException(
              String.format(
                  "docgen does not recognize DocCategory '%s' for StarlarkBuiltin '%s'",
                  annot.category(), annot.name()));
      }
    }

    private Category(String title, String description) {
      this.title = title;
      this.description = description;
    }

    private final String title;
    private final String description;

    public String getTitle() {
      return title;
    }

    public String getDescription() {
      return description;
    }

    public String getTemplateIdentifier() {
      return name().toLowerCase().replace("_", "-");
    }
  }
}
