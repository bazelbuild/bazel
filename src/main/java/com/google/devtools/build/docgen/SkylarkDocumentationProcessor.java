// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.docgen.SkylarkJavaInterfaceExplorer.SkylarkBuiltinMethod;
import com.google.devtools.build.docgen.SkylarkJavaInterfaceExplorer.SkylarkJavaMethod;
import com.google.devtools.build.docgen.SkylarkJavaInterfaceExplorer.SkylarkModuleDoc;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.packages.MethodLibrary;
import com.google.devtools.build.lib.rules.SkylarkModules;
import com.google.devtools.build.lib.rules.SkylarkRuleContext;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Environment.NoneType;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin.Param;
import com.google.devtools.build.lib.syntax.SkylarkCallable;
import com.google.devtools.build.lib.syntax.SkylarkModule;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;

/**
 * A class to assemble documentation for Skylark.
 */
public class SkylarkDocumentationProcessor {

  private static final String TOP_LEVEL_ID = "_top_level";

  private static final boolean USE_TEMPLATE = false;

  @SkylarkModule(name = "Global objects, functions and modules",
      doc = "Objects, functions and modules registered in the global environment.")
  private static final class TopLevelModule {}

  static SkylarkModule getTopLevelModule() {
    return TopLevelModule.class.getAnnotation(SkylarkModule.class);
  }

  /**
   * Generates the Skylark documentation to the given output directory.
   */
  public void generateDocumentation(String outputPath) throws IOException,
      BuildEncyclopediaDocException {
    BufferedWriter bw = null;
    File skylarkDocPath = new File(outputPath);
    try {
      bw = new BufferedWriter(new FileWriter(skylarkDocPath));
      if (USE_TEMPLATE) {
        bw.write(SourceFileReader.readTemplateContents(DocgenConsts.SKYLARK_BODY_TEMPLATE,
            ImmutableMap.<String, String>of(
                DocgenConsts.VAR_SECTION_SKYLARK_BUILTIN, generateAllBuiltinDoc())));
      } else {
        bw.write(generateAllBuiltinDoc());
      }
      System.out.println("Skylark documentation generated: " + skylarkDocPath.getAbsolutePath());
    } finally {
      if (bw != null) {
        bw.close();
      }
    }
  }

  @VisibleForTesting
  Map<String, SkylarkModuleDoc> collectModules() {
    Map<String, SkylarkModuleDoc> modules = new TreeMap<>();
    Map<String, SkylarkModuleDoc> builtinModules = collectBuiltinModules();
    Map<SkylarkModule, Class<?>> builtinJavaObjects = collectBuiltinJavaObjects();

    modules.putAll(builtinModules);
    SkylarkJavaInterfaceExplorer explorer = new SkylarkJavaInterfaceExplorer();
    for (SkylarkModuleDoc builtinObject : builtinModules.values()) {
      // Check the return type for built-in functions, it can be a module previously not added.
      for (SkylarkBuiltinMethod builtinMethod : builtinObject.getBuiltinMethods().values()) {
        Class<?> type = builtinMethod.annotation.returnType();
        if (type.isAnnotationPresent(SkylarkModule.class)) {
          explorer.collect(type.getAnnotation(SkylarkModule.class), type, modules);
        }
      }
      explorer.collect(builtinObject.getAnnotation(), builtinObject.getClassObject(), modules);
    }
    for (Entry<SkylarkModule, Class<?>> builtinModule : builtinJavaObjects.entrySet()) {
      explorer.collect(builtinModule.getKey(), builtinModule.getValue(), modules);
    }
    return modules;
  }

  private String generateAllBuiltinDoc() {
    Map<String, SkylarkModuleDoc> modules = collectModules();

    StringBuilder sb = new StringBuilder();
    // Generate the top level module first in the doc
    SkylarkModuleDoc topLevelModule = modules.remove(getTopLevelModule().name());
    generateModuleDoc(topLevelModule, sb);
    for (SkylarkModuleDoc module : modules.values()) {
      if (!module.getAnnotation().hidden()) {
        sb.append("<hr>");
        generateModuleDoc(module, sb);
      }
    }
    return sb.toString();
  }

  private void generateModuleDoc(SkylarkModuleDoc module, StringBuilder sb) {
    SkylarkModule annotation = module.getAnnotation();
    sb.append(String.format("<h2 id=\"modules.%s\">%s</h2>\n",
          getModuleId(annotation),
          annotation.name()))
      .append(annotation.doc())
      .append("\n");
    sb.append("<ul>");
    // Sort Java and SkylarkBuiltin methods together. The map key is only used for sorting.
    TreeMap<String, Object> methodMap = new TreeMap<>();
    for (SkylarkJavaMethod method : module.getJavaMethods()) {
      methodMap.put(method.name + method.method.getParameterTypes().length, method);
    }
    for (SkylarkBuiltinMethod builtin : module.getBuiltinMethods().values()) {
      methodMap.put(builtin.annotation.name(), builtin);
    }
    for (Object object : methodMap.values()) {
      if (object instanceof SkylarkJavaMethod) {
        SkylarkJavaMethod method = (SkylarkJavaMethod) object;
        generateDirectJavaMethodDoc(annotation.name(), method.name, method.method,
            method.callable, sb);
      }
      if (object instanceof SkylarkBuiltinMethod) {
        generateBuiltinItemDoc(getModuleId(annotation), (SkylarkBuiltinMethod) object, sb);
      }
    }
    sb.append("</ul>");
  }

  private String getModuleId(SkylarkModule annotation) {
    if (annotation == getTopLevelModule()) {
      return TOP_LEVEL_ID;
    } else {
      return annotation.name();
    }
  }

  private void generateBuiltinItemDoc(
      String moduleId, SkylarkBuiltinMethod method, StringBuilder sb) {
    SkylarkBuiltin annotation = method.annotation;
    if (annotation.hidden()) {
      return;
    }
    sb.append(String.format("<li><h3 id=\"modules.%s.%s\">%s</h3>\n",
          moduleId,
          annotation.name(),
          annotation.name()));

    if (com.google.devtools.build.lib.syntax.Function.class.isAssignableFrom(method.fieldClass)) {
      sb.append(getSignature(moduleId, annotation));
    } else {
      if (!annotation.returnType().equals(Object.class)) {
        sb.append("<code>" + getTypeAnchor(annotation.returnType()) + "</code><br>");
      }
    }

    sb.append(annotation.doc() + "\n");
    printParams(moduleId, annotation, sb);
  }

  private void printParams(String moduleId, SkylarkBuiltin annotation, StringBuilder sb) {
    if (annotation.mandatoryParams().length + annotation.optionalParams().length > 0) {
      sb.append("<h4>Parameters</h4>\n");
      printParams(moduleId, annotation.name(), annotation.mandatoryParams(), sb);
      printParams(moduleId, annotation.name(), annotation.optionalParams(), sb);
    } else {
      sb.append("<br/>\n");
    }
  }

  private void generateDirectJavaMethodDoc(String objectName, String methodName,
      Method method, SkylarkCallable annotation, StringBuilder sb) {
    if (annotation.hidden()) {
      return;
    }

    sb.append(String.format("<li><h3 id=\"modules.%s.%s\">%s</h3>\n%s\n",
            objectName,
            methodName,
            methodName,
            getSignature(objectName, methodName, method)))
        .append(annotation.doc())
        .append(getReturnTypeExtraMessage(annotation))
        .append("\n");
  }

  private String getReturnTypeExtraMessage(SkylarkCallable annotation) {
    if (annotation.allowReturnNones()) {
      return " May return <code>None</code>.\n";
    }
    return "";
  }

  private String getSignature(String objectName, String methodName, Method method) {
    String args = method.getAnnotation(SkylarkCallable.class).structField()
        ? "" : "(" + getParameterString(method) + ")";

    return String.format("<code>%s %s.%s%s</code><br>",
        getTypeAnchor(method.getReturnType()), objectName, methodName, args);
  }

  private String getSignature(String objectName, SkylarkBuiltin method) {
    List<String> argList = new ArrayList<>();
    for (Param param : method.mandatoryParams()) {
      argList.add(param.name());
    }
    for (Param param : method.optionalParams()) {
      argList.add(param.name() + "?");
    }
    String args = "(" + Joiner.on(", ").join(argList) + ")";
    if (!objectName.equals(TOP_LEVEL_ID)) {
      return String.format("<code>%s %s.%s%s</code><br>\n",
          getTypeAnchor(method.returnType()), objectName, method.name(), args);
    } else {
      return String.format("<code>%s %s%s</code><br>\n",
          getTypeAnchor(method.returnType()), method.name(), args);
    }
  }

  private String getTypeAnchor(Class<?> returnType, Class<?> generic1) {
    return getTypeAnchor(returnType) + " of " + getTypeAnchor(generic1) + "s";
  }

  private String getTypeAnchor(Class<?> returnType) {
    if (returnType.equals(String.class)) {
      return "<a class=\"anchor\" href=\"#modules.string\">string</a>";
    } else if (Map.class.isAssignableFrom(returnType)) {
      return "<a class=\"anchor\" href=\"#modules.dict\">dict</a>";
    } else if (List.class.isAssignableFrom(returnType)) {
      // Annotated Java methods can return simple java.util.Lists (which get auto-converted).
      return "<a class=\"anchor\" href=\"#modules.list\">list</a>";
    } else if (returnType.equals(Void.TYPE) || returnType.equals(NoneType.class)) {
      return "<a class=\"anchor\" href=\"#modules." + TOP_LEVEL_ID + ".None\">None</a>";
    } else if (returnType.isAnnotationPresent(SkylarkModule.class)) {
      // TODO(bazel-team): this can produce dead links for types don't show up in the doc.
      // The correct fix is to generate those types (e.g. SkylarkFileType) too.
      String module = returnType.getAnnotation(SkylarkModule.class).name();
      return "<a class=\"anchor\" href=\"#modules." + module + "\">" + module + "</a>";
    } else {
      return EvalUtils.getDataTypeNameFromClass(returnType);
    }
  }

  private String getParameterString(Method method) {
    return Joiner.on(", ").join(Iterables.transform(
        ImmutableList.copyOf(method.getParameterTypes()), new Function<Class<?>, String>() {
          @Override
          public String apply(Class<?> input) {
            return getTypeAnchor(input);
          }
        }));
  }

  private void printParams(String moduleId, String methodName,
      Param[] params, StringBuilder sb) {
    if (params.length > 0) {
      sb.append("<ul>\n");
      for (Param param : params) {
        String paramType = param.type().equals(Object.class) ? ""
            : (param.generic1().equals(Object.class)
                ? " (" + getTypeAnchor(param.type()) + ")"
                : " (" + getTypeAnchor(param.type(), param.generic1()) + ")");
        sb.append(String.format("\t<li id=\"modules.%s.%s.%s\"><code>%s%s</code>: ",
            moduleId,
            methodName,
            param.name(),
            param.name(),
            paramType))
          .append(param.doc())
          .append("\n\t</li>\n");
      }
      sb.append("</ul>\n");
    }
  }

  private Map<String, SkylarkModuleDoc> collectBuiltinModules() {
    Map<String, SkylarkModuleDoc> modules = new HashMap<>();
    collectBuiltinDoc(modules, Environment.class.getDeclaredFields());
    collectBuiltinDoc(modules, MethodLibrary.class.getDeclaredFields());
    for (Class<?> moduleClass : SkylarkModules.MODULES) {
      collectBuiltinDoc(modules, moduleClass.getDeclaredFields());
    }
    return modules;
  }

  private Map<SkylarkModule, Class<?>> collectBuiltinJavaObjects() {
    Map<SkylarkModule, Class<?>> modules = new HashMap<>();
    collectBuiltinModule(modules, SkylarkRuleContext.class);
    collectBuiltinModule(modules, TransitiveInfoCollection.class);
    return modules;
  }

  /**
   * Returns the top level modules and functions with their documentation in a command-line
   * printable format.
   */
  public Map<String, String> collectTopLevelModules() {
    Map<String, String> modules = new TreeMap<>();
    for (SkylarkModuleDoc doc : collectBuiltinModules().values()) {
      if (doc.getAnnotation() == getTopLevelModule()) {
        for (Map.Entry<String, SkylarkBuiltinMethod> entry : doc.getBuiltinMethods().entrySet()) {
          if (!entry.getValue().annotation.hidden()) {
            modules.put(entry.getKey(),
                DocgenConsts.toCommandLineFormat(entry.getValue().annotation.doc()));
          }
        }
      } else {
        modules.put(doc.getAnnotation().name(),
            DocgenConsts.toCommandLineFormat(doc.getAnnotation().doc()));
      }
    }
    return modules;
  }

  /**
   * Returns the API doc for the specified Skylark object in a command line printable format,
   * params[0] identifies either a module or a top-level object, the optional params[1] identifies a
   * method in the module.<br>
   * Returns null if no Skylark object is found.
   */
  public String getCommandLineAPIDoc(String[] params) {
    Map<String, SkylarkModuleDoc> modules = collectModules();
    SkylarkModuleDoc toplevelModuleDoc = modules.get(getTopLevelModule().name());
    if (modules.containsKey(params[0])) {
      // Top level module
      SkylarkModuleDoc module = modules.get(params[0]);
      if (params.length == 1) {
        String moduleName = module.getAnnotation().name();
        StringBuilder sb = new StringBuilder();
        sb.append(moduleName).append("\n\t").append(module.getAnnotation().doc()).append("\n");
        // Print the signature of all built-in methods
        for (SkylarkBuiltinMethod method : module.getBuiltinMethods().values()) {
          printBuiltinFunctionDoc(moduleName, method.annotation, sb);
        }
        // Print all Java methods
        for (SkylarkJavaMethod method : module.getJavaMethods()) {
          printJavaFunctionDoc(moduleName, method, sb);
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

  private String getFunctionDoc(String moduleName, String methodName, SkylarkModuleDoc module) {
    if (module.getBuiltinMethods().containsKey(methodName)) {
      // Create the doc for the built-in function
      SkylarkBuiltinMethod method = module.getBuiltinMethods().get(methodName);
      StringBuilder sb = new StringBuilder();
      printBuiltinFunctionDoc(moduleName, method.annotation, sb);
      printParams(moduleName, method.annotation, sb);
      return DocgenConsts.removeDuplicatedNewLines(DocgenConsts.toCommandLineFormat(sb.toString()));
    } else {
      // Search if there are matching Java functions
      StringBuilder sb = new StringBuilder();
      boolean foundMatchingMethod = false;
      for (SkylarkJavaMethod method : module.getJavaMethods()) {
        if (method.name.equals(methodName)) {
          printJavaFunctionDoc(moduleName, method, sb);
          foundMatchingMethod = true;
        }
      }
      if (foundMatchingMethod) {
        return DocgenConsts.toCommandLineFormat(sb.toString()); 
      }
    }
    return null;
  }

  private void printBuiltinFunctionDoc(
      String moduleName, SkylarkBuiltin annotation, StringBuilder sb) {
    if (moduleName != null) {
      sb.append(moduleName).append(".");
    }
    sb.append(annotation.name()).append("\n\t").append(annotation.doc()).append("\n");
  }

  private void printJavaFunctionDoc(String moduleName, SkylarkJavaMethod method, StringBuilder sb) {
    sb.append(getSignature(moduleName, method.name, method.method))
      .append("\t").append(method.callable.doc()).append("\n");
  }

  private void collectBuiltinModule(
      Map<SkylarkModule, Class<?>> modules, Class<?> moduleClass) {
    if (moduleClass.isAnnotationPresent(SkylarkModule.class)) {
      SkylarkModule skylarkModule = moduleClass.getAnnotation(SkylarkModule.class);
      modules.put(skylarkModule, moduleClass);
    }
  }

  private void collectBuiltinDoc(Map<String, SkylarkModuleDoc> modules, Field[] fields) {
    for (Field field : fields) {
      if (field.isAnnotationPresent(SkylarkBuiltin.class)) {
        SkylarkBuiltin skylarkBuiltin = field.getAnnotation(SkylarkBuiltin.class);
        Class<?> moduleClass = skylarkBuiltin.objectType();
        SkylarkModule skylarkModule = moduleClass.equals(Object.class)
            ? getTopLevelModule()
            : moduleClass.getAnnotation(SkylarkModule.class);
        if (!modules.containsKey(skylarkModule.name())) {
          modules.put(skylarkModule.name(), new SkylarkModuleDoc(skylarkModule, moduleClass));
        }
        modules.get(skylarkModule.name()).getBuiltinMethods()
            .put(skylarkBuiltin.name(), new SkylarkBuiltinMethod(skylarkBuiltin, field.getType()));
      }
    }
  }
}
