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

import com.google.devtools.build.docgen.skylark.SkylarkBuiltinMethodDoc;
import com.google.devtools.build.docgen.skylark.SkylarkJavaMethodDoc;
import com.google.devtools.build.docgen.skylark.SkylarkModuleDoc;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A class to assemble documentation for Skylark.
 */
public final class SkylarkDocumentationProcessor {
  private SkylarkDocumentationProcessor() {}

  /**
   * Generates the Skylark documentation to the given output directory.
   */
  public static void generateDocumentation(String outputDir) throws IOException,
      BuildEncyclopediaDocException {
    Map<String, SkylarkModuleDoc> modules = SkylarkDocumentationCollector.collectModules();
    List<SkylarkModuleDoc> navModules = new ArrayList<>();

    // Generate the top level module first in the doc
    SkylarkModuleDoc topLevelModule = modules.remove(
        SkylarkDocumentationCollector.getTopLevelModule().name());
    topLevelModule.setTitle("Globals");
    writePage(outputDir, topLevelModule);
    navModules.add(topLevelModule);

    for (SkylarkModuleDoc module : modules.values()) {
      if (module.getAnnotation().documented()) {
        writePage(outputDir, module);
        navModules.add(module);
      }
    }
    writeNavPage(outputDir, navModules);
  }

  private static void writePage(String outputDir, SkylarkModuleDoc module) throws IOException {
    File skylarkDocPath = new File(outputDir + "/" + module.getName() + ".html");
    Page page = TemplateEngine.newPage(DocgenConsts.SKYLARK_LIBRARY_TEMPLATE);
    page.add("module", module);
    page.write(skylarkDocPath);
  }

  private static void writeNavPage(String outputDir, List<SkylarkModuleDoc> navModules)
      throws IOException {
    File navFile = new File(outputDir + "/" + "skylark-nav.html");
    Page page = TemplateEngine.newPage(DocgenConsts.SKYLARK_NAV_TEMPLATE);
    page.add("modules", navModules);
    page.write(navFile);
  }

  /**
   * Returns the API doc for the specified Skylark object in a command line printable format,
   * params[0] identifies either a module or a top-level object, the optional params[1] identifies a
   * method in the module.<br>
   * Returns null if no Skylark object is found.
   */
  public static String getCommandLineAPIDoc(String[] params) {
    Map<String, SkylarkModuleDoc> modules = SkylarkDocumentationCollector.collectModules();
    SkylarkModuleDoc toplevelModuleDoc = modules.get(
        SkylarkDocumentationCollector.getTopLevelModule().name());
    if (modules.containsKey(params[0])) {
      // Top level module
      SkylarkModuleDoc module = modules.get(params[0]);
      if (params.length == 1) {
        String moduleName = module.getAnnotation().name();
        StringBuilder sb = new StringBuilder();
        sb.append(moduleName).append("\n\t").append(module.getAnnotation().doc()).append("\n");
        // Print the signature of all built-in methods
        for (SkylarkBuiltinMethodDoc method : module.getBuiltinMethods().values()) {
          printBuiltinFunctionDoc(moduleName, method, sb);
        }
        // Print all Java methods
        for (SkylarkJavaMethodDoc method : module.getJavaMethods()) {
          printJavaFunctionDoc(method, sb);
        }
        return DocgenConsts.toCommandLineFormat(sb.toString());
      } else {
        return getFunctionDoc(module.getAnnotation().name(), params[1], module);
      }
    } else if (toplevelModuleDoc.getBuiltinMethods().containsKey(params[0])){
      // Top level object / function
      return getFunctionDoc(null, params[0], toplevelModuleDoc);
    }
    return null;
  }

  private static String getFunctionDoc(String moduleName, String methodName,
      SkylarkModuleDoc module) {
    if (module.getBuiltinMethods().containsKey(methodName)) {
      // Create the doc for the built-in function
      SkylarkBuiltinMethodDoc method = module.getBuiltinMethods().get(methodName);
      StringBuilder sb = new StringBuilder();
      printBuiltinFunctionDoc(moduleName, method, sb);
      sb.append(method.getParams());
      return DocgenConsts.removeDuplicatedNewLines(DocgenConsts.toCommandLineFormat(sb.toString()));
    } else {
      // Search if there are matching Java functions
      StringBuilder sb = new StringBuilder();
      boolean foundMatchingMethod = false;
      for (SkylarkJavaMethodDoc method : module.getJavaMethods()) {
        if (method.getName().equals(methodName)) {
          printJavaFunctionDoc(method, sb);
          foundMatchingMethod = true;
        }
      }
      if (foundMatchingMethod) {
        return DocgenConsts.toCommandLineFormat(sb.toString());
      }
    }
    return null;
  }

  private static void printBuiltinFunctionDoc(String moduleName, SkylarkBuiltinMethodDoc method,
      StringBuilder sb) {
    if (moduleName != null) {
      sb.append(moduleName).append(".");
    }
    sb.append(method.getName()).append("\n\t").append(method.getDocumentation()).append("\n");
  }

  private static void printJavaFunctionDoc(SkylarkJavaMethodDoc method, StringBuilder sb) {
    sb.append(method.getSignature())
      .append("\t").append(method.getDocumentation()).append("\n");
  }
}
