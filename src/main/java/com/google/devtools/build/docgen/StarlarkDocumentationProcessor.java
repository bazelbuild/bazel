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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.docgen.starlark.StarlarkDocExpander;
import com.google.devtools.build.docgen.starlark.StarlarkDocPage;
import com.google.devtools.build.lib.util.Classpath.ClassPathException;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import net.starlark.java.annot.StarlarkBuiltin;

/** A class to assemble documentation for Starlark. */
public final class StarlarkDocumentationProcessor {

  private StarlarkDocumentationProcessor() {}

  /** Generates the Starlark documentation to the given output directory. */
  public static void generateDocumentation(String outputDir, StarlarkDocumentationOptions options)
      throws IOException, ClassPathException {
    if (options.starlarkDocsRoot != null) {
      DocgenConsts.starlarkDocsRoot = options.starlarkDocsRoot;
    }

    DocLinkMap linkMap = DocLinkMap.createFromFile(checkNotNull(options.linkMapPath));
    StarlarkDocExpander expander =
        new StarlarkDocExpander(new RuleLinkExpander(/* singlePage= */ false, linkMap));

    ImmutableMap<Category, ImmutableList<StarlarkDocPage>> allPages =
        StarlarkDocumentationCollector.getAllDocPages(
            expander, ImmutableList.copyOf(options.apiStardocProtos));

    for (var categoryAndPages : allPages.entrySet()) {
      writeCategoryPage(
          categoryAndPages.getKey(), outputDir, categoryAndPages.getValue(), expander);
      for (StarlarkDocPage page : categoryAndPages.getValue()) {
        writePage(outputDir, categoryAndPages.getKey(), page);
      }
    }

    writeOverviewPage(outputDir, allPages);
    if (options.createToc) {
      writeTableOfContents(outputDir, allPages);
    }
  }

  private static void writePage(String outputDir, Category category, StarlarkDocPage docPage)
      throws IOException {
    File starlarkDocPath =
        new File(String.format("%s/%s/%s.html", outputDir, category.getPath(), docPage.getName()));
    Page page = TemplateEngine.newPage(DocgenConsts.STARLARK_LIBRARY_TEMPLATE);
    page.add("page", docPage);
    page.write(starlarkDocPath);
  }

  private static void writeCategoryPage(
      Category category,
      String outputDir,
      ImmutableList<StarlarkDocPage> allPages,
      StarlarkDocExpander expander)
      throws IOException {
    Files.createDirectories(Path.of(outputDir, category.getPath()));

    File starlarkDocPath = new File(String.format("%s/%s.html", outputDir, category.getPath()));
    Page page = TemplateEngine.newPage(DocgenConsts.STARLARK_MODULE_CATEGORY_TEMPLATE);
    page.add("category", category);
    page.add("allPages", allPages);
    page.add("description", expander.expand(category.description));
    page.write(starlarkDocPath);
  }

  private static void writeOverviewPage(
      String outputDir, ImmutableMap<Category, ImmutableList<StarlarkDocPage>> allPages)
      throws IOException {
    File starlarkDocPath = new File(outputDir + "/overview.html");
    Page page = TemplateEngine.newPage(DocgenConsts.STARLARK_OVERVIEW_TEMPLATE);
    page.add("allPages", allPages);
    page.write(starlarkDocPath);
  }

  private static void writeTableOfContents(
      String outputDir, ImmutableMap<Category, ImmutableList<StarlarkDocPage>> allPages)
      throws IOException {
    File starlarkDocPath = new File(outputDir + "/_toc.yaml");
    Page page = TemplateEngine.newPage(DocgenConsts.STARLARK_TOC_TEMPLATE);
    page.add("allPages", allPages);
    page.write(starlarkDocPath);
  }

  /**
   * An enumeration of categories used to organize the API index. Instances of this class are
   * accessed by templates, using reflection.
   */
  public enum Category {
    GLOBAL_FUNCTION(
        "Global functions",
        "globals",
        "This section lists the global functions available in Starlark. The list of available"
            + " functions differs depending on the file type (whether a BUILD file, or a .bzl file,"
            + " etc)."),

    CONFIGURATION_FRAGMENT(
        "Configuration Fragments",
        "fragments",
        "Configuration fragments give rules access to "
            + "language-specific parts of <a href=\"builtins/configuration.html\">"
            + "configuration</a>. "
            + "<p>Rule implementations can get them using "
            + "<code><a href=\"builtins/ctx.html#fragments\">ctx."
            + "fragments</a>.<i>[fragment name]</i></code>"),

    PROVIDER(
        "Providers",
        "providers",
        "This section lists providers available on built-in rules. See the <a"
            + " href='https://bazel.build/extending/rules#providers'>Rules page</a> for more on"
            + " providers. These symbols are available only in .bzl files."),

    BUILTIN(
        "Built-in Types",
        "builtins",
        "This section lists types of Starlark objects. With some exceptions, these type names are"
            + " not valid Starlark symbols; instances of them may be acquired through different"
            + " means."),

    // Used for top-level modules of functions in the global namespace. Such modules will always
    // be usable solely by accessing their members, via modulename.funcname() or
    // modulename.constantname.
    // Examples: attr, cc_common, config, java_common
    TOP_LEVEL_MODULE(
        "Top-level Modules",
        "toplevel",
        "This section lists top-level modules. These symbols are available only in .bzl files."),

    CORE(
        "Core Starlark data types",
        "core",
        "This section lists the data types of the <a"
            + " href='https://github.com/bazelbuild/starlark/blob/master/spec.md#built-in-constants-and-functions'>Starlark"
            + " core language</a>. With some exceptions, these type names are not valid Starlark"
            + " symbols; instances of them may be acquired through different means.");

    // Maps (essentially free-form) strings in annotations to permitted categories.
    public static Category of(StarlarkBuiltin annot) {
      return switch (annot.category()) {
        case DocCategory.CONFIGURATION_FRAGMENT -> CONFIGURATION_FRAGMENT;
        case DocCategory.PROVIDER -> PROVIDER;
        case DocCategory.BUILTIN -> BUILTIN;
        case DocCategory.TOP_LEVEL_MODULE -> TOP_LEVEL_MODULE;
        case "core", "core.lib" ->
            // interpreter built-ins (e.g. int)
            // Starlark standard modules (e.g. json)
            CORE;
        default ->
            throw new IllegalStateException(
                String.format(
                    "docgen does not recognize DocCategory '%s' for StarlarkBuiltin '%s'",
                    annot.category(), annot.name()));
      };
    }

    Category(String title, String path, String description) {
      this.title = title;
      this.path = path;
      this.description = description;
    }

    private final String title;
    private final String path;
    private final String description;

    public String getTitle() {
      return title;
    }

    public String getDescription() {
      return description;
    }

    public String getPath() {
      return path;
    }
  }
}
