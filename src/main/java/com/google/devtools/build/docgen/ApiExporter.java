// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.docgen.builtin.BuiltinProtos.Builtins;
import com.google.devtools.build.docgen.builtin.BuiltinProtos.Callable;
import com.google.devtools.build.docgen.builtin.BuiltinProtos.Param;
import com.google.devtools.build.docgen.builtin.BuiltinProtos.Type;
import com.google.devtools.build.docgen.builtin.BuiltinProtos.Value;
import com.google.devtools.build.docgen.skylark.SkylarkConstructorMethodDoc;
import com.google.devtools.build.docgen.skylark.SkylarkMethodDoc;
import com.google.devtools.build.docgen.skylark.SkylarkModuleDoc;
import com.google.devtools.build.docgen.skylark.SkylarkParamDoc;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.util.Classpath.ClassPathException;
import com.google.devtools.common.options.OptionsParser;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Collections;
import java.util.Map;

/** The main class for the Skylark documentation generator. */
public class ApiExporter {
  private static ConfiguredRuleClassProvider createRuleClassProvider(String classProvider)
      throws ClassNotFoundException, NoSuchMethodException, InvocationTargetException,
          IllegalAccessException {
    Class<?> providerClass = Class.forName(classProvider);
    Method createMethod = providerClass.getMethod("create");
    return (ConfiguredRuleClassProvider) createMethod.invoke(null);
  }

  private static void appendBuiltins(
      ProtoFileBuildEncyclopediaProcessor processor, String filename) {
    try (BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(filename))) {
      Builtins.Builder builtins = Builtins.newBuilder();

      Map<String, SkylarkModuleDoc> allTypes = SkylarkDocumentationCollector.collectModules();

      // Add all global variables and functions in Builtins as Values.
      SkylarkModuleDoc topLevelModule =
          allTypes.remove(SkylarkDocumentationCollector.getTopLevelModule().name());
      for (SkylarkMethodDoc meth : topLevelModule.getMethods()) {
        builtins.addGlobal(collectFieldInfo(meth));
      }
      for (Map.Entry<String, SkylarkModuleDoc> modEntry : allTypes.entrySet()) {
        SkylarkModuleDoc mod = modEntry.getValue();

        // Include SkylarkModuleDoc in Builtins as a Type.
        Type.Builder type = Type.newBuilder();
        type.setName(mod.getName());
        type.setDoc(mod.getDocumentation());
        for (SkylarkMethodDoc meth : mod.getJavaMethods()) {
          // Constructors should be exported as globals.
          if (meth instanceof SkylarkConstructorMethodDoc) {
            builtins.addGlobal(collectFieldInfo(meth));
          } else {
            type.addField(collectFieldInfo(meth));
          }
        }
        // Add native rules to the native type.
        if (mod.getName().equals("native")) {
          for (Value.Builder rule : processor.getNativeRules()) {
            type.addField(rule);
          }
        }
        builtins.addType(type);

        // Include SkylarkModuleDoc in Builtins as a Value.
        Value.Builder value = Value.newBuilder();
        value.setName(mod.getName());
        value.setType(mod.getName());
        value.setDoc(mod.getDocumentation());
        builtins.addGlobal(value);
      }
      Builtins build = builtins.build();
      build.writeTo(out);
    } catch (IOException | ClassPathException e) {
      System.err.println(e);
    }
  }

  private static Value.Builder collectFieldInfo(SkylarkMethodDoc meth) {
    Value.Builder field = Value.newBuilder();
    field.setName(meth.getShortName());
    field.setDoc(meth.getDocumentation());
    if (meth.isCallable()) {
      Callable.Builder callable = Callable.newBuilder();
      for (SkylarkParamDoc par : meth.getParams()) {
        Param.Builder param = Param.newBuilder();
        param.setName(par.getName());
        param.setType(par.getType());
        param.setDoc(par.getDocumentation());
        param.setDefaultValue(par.getDefaultValue());
        callable.addParam(param);
      }
      callable.setReturnType(meth.getReturnType());
      field.setCallable(callable);
    } else {
      field.setType(meth.getReturnType());
    }
    return field;
  }

  private static void printUsage(OptionsParser parser) {
    System.err.println(
        "Usage: api_exporter_bin -n product_name -p rule_class_provider (-i input_dir)+\n"
            + "   -f outputFile [-b blacklist] [-h]\n\n"
            + "Exports all Starlark builtins to a file including the embedded native rules.\n"
            + "The product name (-n), rule class provider (-p), output file (-f) and at least \n"
            + " one input_dir (-i) must be specified.\n");
    System.err.println(
        parser.describeOptionsWithDeprecatedCategories(
            Collections.<String, String>emptyMap(), OptionsParser.HelpVerbosity.LONG));
  }

  public static void main(String[] args) {
    OptionsParser parser = OptionsParser.newOptionsParser(BuildEncyclopediaOptions.class);
    parser.parseAndExitUponError(args);
    BuildEncyclopediaOptions options = parser.getOptions(BuildEncyclopediaOptions.class);

    if (options.help) {
      printUsage(parser);
      Runtime.getRuntime().exit(0);
    }

    if (options.productName.isEmpty()
        || options.inputDirs.isEmpty()
        || options.provider.isEmpty()
        || options.outputFile.isEmpty()) {
      printUsage(parser);
      Runtime.getRuntime().exit(1);
    }

    try {
      ProtoFileBuildEncyclopediaProcessor processor =
          new ProtoFileBuildEncyclopediaProcessor(
              options.productName, createRuleClassProvider(options.provider));
      processor.generateDocumentation(options.inputDirs, options.outputFile, options.blacklist);

      appendBuiltins(processor, options.outputFile);
    } catch (Throwable e) {
      System.err.println("ERROR: " + e.getMessage());
    }
  }
}

