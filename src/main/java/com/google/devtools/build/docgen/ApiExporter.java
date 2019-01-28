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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.docgen.builtin.BuiltinProtos.ApiContext;
import com.google.devtools.build.docgen.builtin.BuiltinProtos.Builtins;
import com.google.devtools.build.docgen.builtin.BuiltinProtos.Callable;
import com.google.devtools.build.docgen.builtin.BuiltinProtos.Param;
import com.google.devtools.build.docgen.builtin.BuiltinProtos.Type;
import com.google.devtools.build.docgen.builtin.BuiltinProtos.Value;
import com.google.devtools.build.docgen.skylark.SkylarkConstructorMethodDoc;
import com.google.devtools.build.docgen.skylark.SkylarkMethodDoc;
import com.google.devtools.build.docgen.skylark.SkylarkModuleDoc;
import com.google.devtools.build.docgen.skylark.SkylarkParamDoc;
import com.google.devtools.build.lib.skylarkinterface.SkylarkInterfaceUtils;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.MethodDescriptor;
import com.google.devtools.build.lib.syntax.SkylarkSemantics;
import com.google.devtools.build.lib.syntax.SkylarkSignatureProcessor;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.common.options.OptionsParser;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/** The main class for the Skylark documentation generator. */
public class ApiExporter {

  private static void appendTypes(
      Builtins.Builder builtins,
      Map<String, SkylarkModuleDoc> types,
      List<RuleDocumentation> nativeRules)
      throws BuildEncyclopediaDocException {

    for (Entry<String, SkylarkModuleDoc> modEntry : types.entrySet()) {
      SkylarkModuleDoc mod = modEntry.getValue();

      Type.Builder type = Type.newBuilder();
      type.setName(mod.getName());
      type.setDoc(mod.getDocumentation());
      for (SkylarkMethodDoc meth : mod.getJavaMethods()) {
        // Constructors are exported as global symbols.
        if (!(meth instanceof SkylarkConstructorMethodDoc)) {
          Value.Builder value = collectMethodInfo(meth);
          // Methods from the native package are available as top level functions in BUILD files.
          if (mod.getName().equals("native")) {
            value.setApiContext(ApiContext.BUILD);
            builtins.addGlobal(value);

            value.setApiContext(ApiContext.BZL);
            type.addField(value);
          } else {
            value.setApiContext(ApiContext.ALL);
            type.addField(value);
          }
        }
      }
      // Native rules are available in BZL file as methods of the native package.
      if (mod.getName().equals("native")) {
        for (RuleDocumentation rule : nativeRules) {
          Value.Builder field = collectRuleInfo(rule);
          field.setApiContext(ApiContext.BZL);
          type.addField(field);
        }
      }
      builtins.addType(type);
    }
  }

  // Globals are available for both BUILD and BZL files.
  private static void appendGlobals(Builtins.Builder builtins, Map<String, Object> globalMethods) {
    for (Entry<String, Object> entry : globalMethods.entrySet()) {
      Object obj = entry.getValue();
      Value.Builder value = Value.newBuilder();
      if (obj instanceof BaseFunction) {
        value = collectFunctionInfo((BaseFunction) obj);
      } else {
        value.setName(entry.getKey());
      }
      value.setApiContext(ApiContext.ALL);
      builtins.addGlobal(value);
    }
  }

  private static void appendBzlGlobals(
      Builtins.Builder builtins, Map<String, Object> starlarkGlobals) {
    for (Entry<String, Object> entry : starlarkGlobals.entrySet()) {
      Object obj = entry.getValue();
      Value.Builder value = Value.newBuilder();

      if (obj instanceof BaseFunction) {
        value = collectFunctionInfo((BaseFunction) obj);
      } else {
        SkylarkModule typeModule = SkylarkInterfaceUtils.getSkylarkModule(obj.getClass());
        if (typeModule != null) {
          if (FuncallExpression.hasSelfCallMethod(
              SkylarkSemantics.DEFAULT_SEMANTICS, obj.getClass())) {
            MethodDescriptor descriptor =
                FuncallExpression.getSelfCallMethodDescriptor(
                    SkylarkSemantics.DEFAULT_SEMANTICS, obj);

            value = collectFunctionInfo(
                descriptor.getName(),
                SkylarkSignatureProcessor.getSignatureForCallable(
                    descriptor.getName(), descriptor, null, null));
          } else {
            value.setName(entry.getKey());
            value.setType(entry.getKey());
            value.setDoc(typeModule.doc());
          }
        }
      }
      value.setApiContext(ApiContext.BZL);
      builtins.addGlobal(value);
    }
  }

  // Native rules are available as top level functions in BUILD files.
  private static void appendNativeRules(
      Builtins.Builder builtins, List<RuleDocumentation> nativeRules)
      throws BuildEncyclopediaDocException {
    for (RuleDocumentation rule : nativeRules) {
      Value.Builder global = collectRuleInfo(rule);
      global.setApiContext(ApiContext.BUILD);
      builtins.addGlobal(global);
    }
  }

  private static Value.Builder collectFunctionInfo(BaseFunction func) {
    return collectFunctionInfo(func.getName(), func.getSignature());
  }

  private static Value.Builder collectFunctionInfo(
      String funcName, FunctionSignature.WithValues<Object, SkylarkType> funcSignature) {
    Value.Builder value = Value.newBuilder();
    value.setName(funcName);
    Callable.Builder callable = Callable.newBuilder();

    ImmutableList<String> paramNames = funcSignature.getSignature().getNames();
    List<Object> defaultValues = funcSignature.getDefaultValues();
    int positionals = funcSignature.getSignature().getShape().getMandatoryPositionals();
    int optionals = funcSignature.getSignature().getShape().getOptionals();
    int nameIndex = 0;

    for (int i = 0; i < positionals; i++) {
      Param.Builder param = Param.newBuilder();
      param.setName(paramNames.get(nameIndex));
      param.setIsMandatory(true);
      callable.addParam(param);
      nameIndex++;
    }

    for (int i = 0; i < optionals; i++) {
      Param.Builder param = Param.newBuilder();
      param.setName(paramNames.get(nameIndex));
      param.setIsMandatory(false);
      param.setDefaultValue(defaultValues.get(i).toString());
      callable.addParam(param);
      nameIndex++;
    }

    if (funcSignature.getSignature().getShape().hasStarArg()) {
      Param.Builder param = Param.newBuilder();
      param.setName("*" + paramNames.get(nameIndex));
      param.setIsMandatory(false);
      param.setIsStarArg(true);
      nameIndex++;
      callable.addParam(param);
    }
    if (funcSignature.getSignature().getShape().hasKwArg()) {
      Param.Builder param = Param.newBuilder();
      param.setIsMandatory(false);
      param.setIsStarStarArg(true);
      param.setName("**" + paramNames.get(nameIndex));
      callable.addParam(param);
    }
    value.setCallable(callable);
    return value;
  }

  private static Value.Builder collectMethodInfo(SkylarkMethodDoc meth) {
    Value.Builder field = Value.newBuilder();
    field.setName(meth.getShortName());
    field.setDoc(meth.getDocumentation());
    if (meth.isCallable()) {
      Callable.Builder callable = Callable.newBuilder();
      for (SkylarkParamDoc par : meth.getParams()) {
        Param.Builder param = newParam(par.getName(), par.getDefaultValue().isEmpty());
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

  private static Param.Builder newParam(String name, Boolean isMandatory) {
    Param.Builder param = Param.newBuilder();
    param.setName(name);
    param.setIsMandatory(isMandatory);
    return param;
  }

  private static Value.Builder collectRuleInfo(RuleDocumentation rule)
      throws BuildEncyclopediaDocException {
    Value.Builder value = Value.newBuilder();
    value.setName(rule.getRuleName());
    value.setDoc(rule.getHtmlDocumentation());
    Callable.Builder callable = Callable.newBuilder();
    // All native rules have attribute "name". It is not included in the attributes list and needs
    // to be added separately.
    callable.addParam(newParam("name", true));
    for (RuleDocumentationAttribute attr : rule.getAttributes()) {
      callable.addParam(newParam(attr.getAttributeName(), attr.isMandatory()));
    }
    value.setCallable(callable);
    return value;
  }

  private static void writeBuiltins(String filename, Builtins.Builder builtins) throws IOException {
    try (BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(filename))) {
      Builtins build = builtins.build();
      build.writeTo(out);
    }
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
      SymbolFamilies symbols =
          new SymbolFamilies(
              options.productName, options.provider, options.inputDirs, options.blacklist);
      Builtins.Builder builtins = Builtins.newBuilder();

      appendTypes(builtins, symbols.getTypes(), symbols.getNativeRules());
      appendGlobals(builtins, symbols.getGlobals());
      appendBzlGlobals(builtins, symbols.getBzlGlobals());
      appendNativeRules(builtins, symbols.getNativeRules());
      writeBuiltins(options.outputFile, builtins);

    } catch (Throwable e) {
      System.err.println("ERROR: " + e.getMessage());
      e.printStackTrace();
    }
  }
}
