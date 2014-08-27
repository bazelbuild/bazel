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
import com.google.devtools.build.docgen.SkylarkJavaInterfaceExplorer.SkylarkJavaObject;
import com.google.devtools.build.lib.packages.MethodLibrary;
import com.google.devtools.build.lib.rules.SkylarkAttr;
import com.google.devtools.build.lib.rules.SkylarkCommandLine;
import com.google.devtools.build.lib.rules.SkylarkRuleClassFunctions;
import com.google.devtools.build.lib.rules.SkylarkRuleContext;
import com.google.devtools.build.lib.rules.SkylarkRuleImplementationFunctions;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin.Param;
import com.google.devtools.build.lib.syntax.SkylarkCallable;
import com.google.devtools.build.lib.util.StringUtilities;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * A class to assemble documentation for Skylark.
 */
public class SkylarkDocumentationProcessor {

  /**
   * Generates the Skylark documentation to the given output directory.
   */
  public void generateDocumentation(String outputRootDir) throws IOException,
      BuildEncyclopediaDocException {
    BufferedWriter bw = null;
    File skylarkDocPath = new File(outputRootDir + File.separator + DocgenConsts.SKYLARK_DOC_NAME);
    try {
      bw = new BufferedWriter(new FileWriter(skylarkDocPath));
      bw.write(SourceFileReader.readTemplateContents(DocgenConsts.SKYLARK_BODY_TEMPLATE,
          ImmutableMap.<String, String>of(
              DocgenConsts.VAR_SECTION_SKYLARK_BUILTIN, generateAllBuiltinDoc())));
      System.out.println("Skylark documentation generated: " + skylarkDocPath.getAbsolutePath());
    } finally {
      if (bw != null) {
        bw.close();
      }
    }
  }

  private String generateAllBuiltinDoc() {
    Set<SkylarkJavaObject> reachableObjects = new TreeSet<>();
    StringBuilder sb = new StringBuilder();

    reachableObjects.addAll(collectMethodLibraryDocs());

    Map<SkylarkBuiltin, Class<?>> builtinDoc = collectBuiltinObjects();
    SkylarkJavaInterfaceExplorer explorer = new SkylarkJavaInterfaceExplorer();
    for (Map.Entry<SkylarkBuiltin, Class<?>> annotation : builtinDoc.entrySet()) {
      reachableObjects.addAll(explorer.collect(annotation.getKey(), annotation.getValue()));
    }

    for (SkylarkJavaObject object : reachableObjects) {
      if (!object.getAnnotation().hidden()) {
        sb.append(generateBuiltinItemDoc(object));
      }
    }
    return sb.toString();
  }

  private String getName(SkylarkCallable callable, String javaName) {
    return callable.name().isEmpty()
        ? StringUtilities.toPythonStyleFunctionName(javaName)
        : callable.name();
  }

  private String generateBuiltinItemDoc(SkylarkJavaObject object) {
    SkylarkBuiltin annotation = object.getAnnotation();
    StringBuilder sb = new StringBuilder()
        .append(String.format("<h3 id=\"objects.%s\">%s</h3>\n",
            object.name(),
            object.name()))
        .append(annotation.doc())
        .append("\n");

    printParams("Mandatory parameters", annotation.name(), annotation.mandatoryParams(), sb);
    printParams("Optional parameters", annotation.name(), annotation.optionalParams(), sb);

    for (Map.Entry<Method, SkylarkCallable> method : object.getMethods().entrySet()) {
      sb.append(generateDirectJavaMethodDoc(
          annotation.name(), getName(method.getValue(), method.getKey().getName()),
          method.getKey(), method.getValue()));
    }
    for (Map.Entry<String, SkylarkCallable> method : object.getExtraMethods().entrySet()) {
      sb.append(generateDirectJavaMethodDoc(
          annotation.name(), getName(method.getValue(), method.getKey()),
          null, method.getValue()));
    }

    return sb.toString();
  }

  private String generateDirectJavaMethodDoc(
      String objectName, String methodName, Method method, SkylarkCallable annotation) {
    if (annotation.hidden()) {
      return "";
    }

    StringBuilder sb = new StringBuilder();
    sb.append(String.format("<h4 id=\"objects.%s.%s\">%s</h4>\n%s\n",
            objectName,
            methodName,
            methodName,
            getSignature(objectName, methodName, method)))
        .append(annotation.doc())
        .append("\n");
    return sb.toString();
  }

  private String getSignature(String objectName, String methodName, Method method) {
    if (method == null) {
      return "";
    }

    String args = getParameterString(method);
    if (!args.isEmpty()) {
      args = "(" + args + ")";
    }

    return String.format("<code>%s %s.%s%s</code><br>",
        EvalUtils.getDataTypeNameFromClass(method.getReturnType()), objectName, methodName, args);
  }

  private String getParameterString(Method method) {
    return Joiner.on(", ").join(Iterables.transform(
        ImmutableList.copyOf(method.getParameterTypes()), new Function<Class<?>, String>() {
          @Override
          public String apply(Class<?> input) {
            return EvalUtils.getDataTypeNameFromClass(input);
          }
        }));
  }

  private void printParams(String title, String objectName, Param[] params, StringBuilder sb) {
    if (params.length > 0) {
      sb.append(String.format("<h4>%s</h4>\n", title));
      sb.append("<ul>\n");
      for (Param param : params) {
        sb.append(String.format("\t<li id=\"objects.%s.%s\"><code>%s</code>: ",
            objectName,
            param.name(),
            param.name()))
          .append(param.doc())
          .append("\n\t</li>\n");
      }
      sb.append("</ul>\n");
    }
  }

  @VisibleForTesting
  Map<SkylarkBuiltin, Class<?>> collectBuiltinObjects() {
    ImmutableMap.Builder<SkylarkBuiltin, Class<?>> builder = ImmutableMap.builder();
    collectBuiltinDoc(builder, Environment.class.getDeclaredFields());
    collectBuiltinDoc(builder, MethodLibrary.class.getDeclaredFields());
    collectBuiltinDoc(builder, SkylarkRuleClassFunctions.class.getDeclaredFields());
    collectBuiltinDoc(builder, SkylarkRuleImplementationFunctions.class.getDeclaredFields());
    collectBuiltinDoc(builder, SkylarkAttr.class.getDeclaredFields());
    collectBuiltinDoc(builder, SkylarkCommandLine.class.getDeclaredFields());
    builder.put(
        SkylarkRuleContext.class.getAnnotation(SkylarkBuiltin.class), SkylarkRuleContext.class);
    for (Object obj : SkylarkRuleImplementationFunctions.JAVA_OBJECTS_TO_EXPOSE.values()) {
      if (obj instanceof Class<?>) {
        Class<?> classObj = (Class<?>) obj;
        if (classObj.isAnnotationPresent(SkylarkBuiltin.class)) {
          SkylarkBuiltin skylarkBuiltin = classObj.getAnnotation(SkylarkBuiltin.class);
          builder.put(skylarkBuiltin, classObj);
        }
      }
    }
    return builder.build();
  }

  private void collectBuiltinDoc(ImmutableMap.Builder<SkylarkBuiltin, Class<?>> builder,
      Field[] fields) {
    for (Field field : fields) {
      if (field.isAnnotationPresent(SkylarkBuiltin.class)) {
        SkylarkBuiltin skylarkBuiltin = field.getAnnotation(SkylarkBuiltin.class);
        builder.put(skylarkBuiltin, field.getClass());
      }
    }
  }

  private List<SkylarkJavaObject> collectMethodLibraryDocs() {
    return ImmutableList.<SkylarkJavaObject>builder()
        .add(SkylarkJavaObject.ofExtraMethods(
            getMethodLibraryAnnotation("stringFunctions"),
            collectMethodLibraryDoc(MethodLibrary.stringFunctions.keySet())))
        .add(SkylarkJavaObject.ofExtraMethods(
            getMethodLibraryAnnotation("dictFunctions"),
            collectMethodLibraryDoc(MethodLibrary.dictFunctions.keySet())))
        .build();
  }

  @VisibleForTesting
  ImmutableMap<String, SkylarkCallable> collectMethodLibraryDocMap() {
    return ImmutableMap.<String, SkylarkCallable>builder()
        .putAll(collectMethodLibraryDoc(MethodLibrary.listFunctions))
        .putAll(collectMethodLibraryDoc(MethodLibrary.stringFunctions.keySet()))
        .build();
  }

  private SkylarkBuiltin getMethodLibraryAnnotation(String fieldName) {
    try {
      return MethodLibrary.class.getDeclaredField(fieldName)
          .getAnnotation(SkylarkBuiltin.class);
    } catch (NoSuchFieldException | SecurityException e) {
      throw new RuntimeException(e);
    }
  }

  private Map<String, SkylarkCallable> collectMethodLibraryDoc(
      Iterable<com.google.devtools.build.lib.syntax.Function> functions) {
    Map<String, SkylarkCallable> methods = new HashMap<>();
    for (com.google.devtools.build.lib.syntax.Function function : functions) {
      try {
        // Replacing the '$' character in the name of hidden methods
        String functionName = function.getName().replace("$", "");
        Field field = MethodLibrary.class.getDeclaredField(functionName);
        methods.put(function.getName(), field.getAnnotation(SkylarkCallable.class));
      } catch (NoSuchFieldException | SecurityException e) {
        throw new RuntimeException(e);
      }
    }
    return methods;
  }
}
