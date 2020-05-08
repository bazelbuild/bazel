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
import com.google.devtools.build.docgen.starlark.StarlarkBuiltinDoc;
import com.google.devtools.build.docgen.starlark.StarlarkConstructorMethodDoc;
import com.google.devtools.build.docgen.starlark.StarlarkMethodDoc;
import com.google.devtools.build.docgen.starlark.StarlarkParamDoc;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.StarlarkBuiltin;
import com.google.devtools.build.lib.skylarkinterface.StarlarkInterfaceUtils;
import com.google.devtools.build.lib.syntax.BuiltinCallable;
import com.google.devtools.build.lib.syntax.CallUtils;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkCallable;
import com.google.devtools.build.lib.syntax.StarlarkFunction;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.common.options.OptionsParser;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.function.Function;

/** The main class for the Starlark documentation generator. */
public class ApiExporter {

  private static void appendTypes(
      Builtins.Builder builtins,
      Map<String, StarlarkBuiltinDoc> types,
      List<RuleDocumentation> nativeRules)
      throws BuildEncyclopediaDocException {

    for (Entry<String, StarlarkBuiltinDoc> modEntry : types.entrySet()) {
      StarlarkBuiltinDoc mod = modEntry.getValue();

      Type.Builder type = Type.newBuilder();
      type.setName(mod.getName());
      type.setDoc(mod.getDocumentation());
      for (StarlarkMethodDoc meth : mod.getJavaMethods()) {
        // Constructors are exported as global symbols.
        if (!(meth instanceof StarlarkConstructorMethodDoc)) {
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
      if (obj instanceof StarlarkCallable) {
        value = valueFromCallable((StarlarkCallable) obj);
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

      if (obj instanceof StarlarkCallable) {
        value = valueFromCallable((StarlarkCallable) obj);
      } else {
        StarlarkBuiltin typeModule = StarlarkInterfaceUtils.getStarlarkBuiltin(obj.getClass());
        if (typeModule != null) {
          Method selfCallMethod =
              CallUtils.getSelfCallMethod(StarlarkSemantics.DEFAULT, obj.getClass());
          if (selfCallMethod != null) {
            // selfCallMethod may be from a subclass of the annotated method.
            SkylarkCallable annotation = StarlarkInterfaceUtils.getSkylarkCallable(selfCallMethod);
            value = valueFromAnnotation(annotation);
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

  private static Value.Builder valueFromCallable(StarlarkCallable x) {
    // Starlark def statement?
    if (x instanceof StarlarkFunction) {
      StarlarkFunction fn = (StarlarkFunction) x;
      Signature sig = new Signature();
      sig.name = fn.getName();
      sig.parameterNames = fn.getParameterNames();
      sig.hasVarargs = fn.hasVarargs();
      sig.hasKwargs = fn.hasKwargs();
      sig.getDefaultValue =
          (i) -> {
            Object v = fn.getDefaultValue(i);
            return v == null ? null : Starlark.repr(v);
          };
      return signatureToValue(sig);
    }

    // annotated Java method?
    if (x instanceof BuiltinCallable) {
      return valueFromAnnotation(((BuiltinCallable) x).getAnnotation());
    }

    // application-defined callable?  Treat as def f(**kwargs).
    Signature sig = new Signature();
    sig.name = x.getName();
    sig.parameterNames = ImmutableList.of("kwargs");
    sig.hasKwargs = true;
    return signatureToValue(sig);
  }

  private static Value.Builder valueFromAnnotation(SkylarkCallable annot) {
    return signatureToValue(getSignature(annot));
  }

  private static class Signature {
    String name;
    List<String> parameterNames;
    boolean hasVarargs;
    boolean hasKwargs;

    // Returns the string form of the ith default value, using the
    // index, ordering, and null Conventions of StarlarkFunction.getDefaultValue.
    Function<Integer, String> getDefaultValue = (i) -> null;
  }

  private static Value.Builder signatureToValue(Signature sig) {
    Value.Builder value = Value.newBuilder();
    value.setName(sig.name);

    int nparams = sig.parameterNames.size();
    int kwargsIndex = sig.hasKwargs ? --nparams : -1;
    int varargsIndex = sig.hasVarargs ? --nparams : -1;
    // Inv: nparams is number of regular parameters.

    Callable.Builder callable = Callable.newBuilder();
    for (int i = 0; i < sig.parameterNames.size(); i++) {
      String name = sig.parameterNames.get(i);
      Param.Builder param = Param.newBuilder();
      if (i == varargsIndex) {
        // *args
        param.setName("*" + name); // * seems redundant
        param.setIsStarArg(true);
      } else if (i == kwargsIndex) {
        // **kwargs
        param.setName("**" + name); // ** seems redundant
        param.setIsStarStarArg(true);
      } else {
        // regular parameter
        param.setName(name);
        String v = sig.getDefaultValue.apply(i);
        if (v != null) {
          param.setDefaultValue(v);
        } else {
          param.setIsMandatory(true); // bool seems redundant
        }
      }
      callable.addParam(param);
    }
    value.setCallable(callable);
    return value;
  }

  private static Value.Builder collectMethodInfo(StarlarkMethodDoc meth) {
    Value.Builder field = Value.newBuilder();
    field.setName(meth.getShortName());
    field.setDoc(meth.getDocumentation());
    if (meth.isCallable()) {
      Callable.Builder callable = Callable.newBuilder();
      for (StarlarkParamDoc par : meth.getParams()) {
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
    OptionsParser parser =
        OptionsParser.builder().optionsClasses(BuildEncyclopediaOptions.class).build();
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

  // Extracts signature and parameter default value expressions from a SkylarkCallable annotation.
  private static Signature getSignature(SkylarkCallable annot) {
    // Build-time annotation processing ensures mandatory parameters do not follow optional ones.
    boolean hasStar = false;
    String star = null;
    String starStar = null;
    ArrayList<String> params = new ArrayList<>();
    ArrayList<String> defaults = new ArrayList<>();

    for (com.google.devtools.build.lib.skylarkinterface.Param param : annot.parameters()) {
      // Implicit * or *args parameter separates transition from positional to named.
      // f (..., *, ... )  or  f(..., *args, ...)
      // TODO(adonovan): this logic looks fishy. Clean it up.
      if (param.named() && !param.positional() && !hasStar) {
        hasStar = true;
        if (!annot.extraPositionals().name().isEmpty()) {
          star = annot.extraPositionals().name();
        }
      }
      params.add(param.name());
      defaults.add(param.defaultValue().isEmpty() ? null : param.defaultValue());
    }

    // f(..., *args, ...)
    if (!annot.extraPositionals().name().isEmpty() && !hasStar) {
      star = annot.extraPositionals().name();
    }
    if (star != null) {
      params.add(star);
    }

    // f(..., **kwargs)
    if (!annot.extraKeywords().name().isEmpty()) {
      starStar = annot.extraKeywords().name();
      params.add(starStar);
    }

    Signature sig = new Signature();
    sig.name = annot.name();
    sig.parameterNames = params;
    sig.hasVarargs = star != null;
    sig.hasKwargs = starStar != null;
    sig.getDefaultValue = defaults::get;
    return sig;
  }
}
