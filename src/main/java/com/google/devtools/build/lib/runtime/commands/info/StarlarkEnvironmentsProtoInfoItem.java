// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime.commands.info;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.docgen.builtin.BuiltinProtos.Builtins;
import com.google.devtools.build.docgen.builtin.BuiltinProtos;
import com.google.devtools.build.docgen.builtin.BuiltinProtos.Value;
import com.google.devtools.build.docgen.builtin.BuiltinProtos.ApiContext;
import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.bazel.rules.BazelRuleClassProvider;
import com.google.devtools.build.lib.packages.*;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.InfoItem;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.skyframe.StarlarkBuiltinsValue;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyValue;
import net.starlark.java.annot.StarlarkAnnotations;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.*;

import java.lang.reflect.Method;
import java.util.*;
import java.util.function.Function;

import static com.google.common.base.Preconditions.checkNotNull;

/**
 * TODO:
 *   - Types are not set for now
 *   - StarlarkFunction, like cc_proto_library (which is autoloaded), are missing doc and params or have strange values.
 *       Examples: java_binary, java_import, java_library, java_proto_library, proto_library, java_test, py_binary
 *       Documentation generated for py_binary, seems to be generated based on: https://github.com/bazelbuild/rules_python/blob/026b300d918ced0f4e9f99a22ab8407656ed20ac/python/py_binary.bzl#L28
 *           doc: "Creates an executable Python program.\n\nThis is the public macro wrapping the underlying rule. Args are forwarded\non as-is unless otherwise specified. See the underlying {rule}`py_binary`\nrule for detailed attribute documentation.\n\nThis macro affects the following args:\n* `python_version`: cannot be `PY2`\n* `srcs_version`: cannot be `PY2` or `PY2ONLY`\n* `tags`: May have special marker values added, if not already present.\n\nArgs:\n  **attrs: Rule attributes forwarded onto the underlying {rule}`py_binary`."
 *   - There is 7 symbols that report as members of com.google.devtools.build.lib.packages.AutoloadSymbols$1, possibly
 *      related to the issue that bazel 8.0.0 displays message:
 *      WARNING: Couldn't auto load rules or symbols, because no dependency on module/repository 'rules_android' found. This will result in a failure if there's a reference to those rules or symbols.
 *         AndroidIdeInfo, aar_import, android_*
 *   - Some common rule attributes are missing documentation, documentation seems to be in HTML pages for those:
 *     https://github.com/bazelbuild/bazel/blob/master/src/main/java/com/google/devtools/build/docgen/PredefinedAttributes.java
 *     E.g. visibility, compatible_with, flaky, env, etc.
 *
 */
public final class StarlarkEnvironmentsProtoInfoItem extends InfoItem {

  public StarlarkEnvironmentsProtoInfoItem() {
    super("starlark-environments-proto", "TODO", true);
  }

  @Override
  public boolean needsSyncPackageLoading() {
    // Requires CommandEnvironment.syncPackageLoading to be called in order to initialize the
    // skyframe executor.
    return true;
  }

  @Override
  public byte[] get(Supplier<BuildConfigurationValue> configurationSupplier, CommandEnvironment env)
      throws AbruptExitException {
    checkNotNull(env);
    StarlarkBuiltinsValue builtins = loadStarlarkBuiltins(env);
    return print(build(builtins));
  }

  private StarlarkBuiltinsValue loadStarlarkBuiltins(CommandEnvironment env)
          throws AbruptExitException {
    EvaluationResult<SkyValue> result =
            env.getSkyframeExecutor()
                    .evaluateSkyKeys(
                            env.getReporter(),
                            ImmutableList.of(StarlarkBuiltinsValue.key(true)),
                            /* keepGoing= */ false);
    if (result.hasError()) {
      throw new AbruptExitException(
              DetailedExitCode.of(
                      FailureDetails.FailureDetail.newBuilder()
                              .setMessage("Failed to load Starlark builtins")
                              .setInfoCommand(FailureDetails.InfoCommand.getDefaultInstance())
                              .build()));
    }
    return (StarlarkBuiltinsValue) result.get(StarlarkBuiltinsValue.key(true));
  }

  private byte[] build(StarlarkBuiltinsValue builtins) {
    Builtins.Builder builder = Builtins.newBuilder();

    ConfiguredRuleClassProvider provider = BazelRuleClassProvider.create();
    BazelStarlarkEnvironment env = provider.getBazelStarlarkEnvironment();

    buildFor(builder, ApiContext.ALL, Starlark.UNIVERSE);

    // TODO: Probably there should be BUILD_BZL and WORKSPACE_BZL
    ImmutableMap.Builder<String, Object> bzlBuilder = ImmutableMap.builder();
    bzlBuilder.putAll(builtins.predeclaredForWorkspaceBzl);
    bzlBuilder.putAll(builtins.predeclaredForBuildBzl);
    buildFor(builder, ApiContext.BZL, bzlBuilder.buildKeepingLast());

    buildFor(builder, ApiContext.BUILD, builtins.predeclaredForBuild);
    buildFor(builder, ApiContext.MODULE, env.getModuleBazelEnv());
    buildFor(builder, ApiContext.REPO, env.getRepoBazelEnv());
    buildFor(builder, ApiContext.VENDOR, env.getStarlarkGlobals().getVendorToplevels());

    // Construct WORKSPACE symbols manually, similarly as done in WorkspaceFactory.
    // As commented in WorkspaceFactory.getDefaultEnvironment, the WORKSPACE file is going away so method to get
    // its symbols have not been added to the BazelStarlarkEnvironment.
    ImmutableMap.Builder<String, Object> workspaceEnv = ImmutableMap.builder();
    for (Map.Entry<String, RuleClass> entry : provider.getRuleClassMap().entrySet()) {
      // Add workspace-only symbols, otherwise a lot of unsupported symbols would be added to the WORKSPACE environment.
      if (entry.getValue().getWorkspaceOnly()) {
        workspaceEnv.put(entry.getKey(), entry.getValue());
      }
    }
    WorkspaceGlobals workspaceGlobals = new WorkspaceGlobals(
            /* allowWorkspaceFunction */ true,
            provider.getRuleClassMap());
    Starlark.addMethods(workspaceEnv, workspaceGlobals, StarlarkSemantics.DEFAULT);
    buildFor(builder, ApiContext.WORKSPACE, workspaceEnv.buildKeepingLast());

    return builder.build().toByteArray();
  }

  private void buildFor(Builtins.Builder builtins, ApiContext env, Map<String, Object> symbols) {
    List<Map.Entry<String, Object>> entries = new ArrayList<>(symbols.entrySet());
    entries.sort(Map.Entry.comparingByKey());

    for (var entry : entries) {
      String name = entry.getKey();
      Object obj = entry.getValue();

      if (name.startsWith("_")) {
        continue;
      }

      if (obj instanceof GuardedValue guardedValue) {
        obj = guardedValue.getObject();
      }

      Value.Builder value = Value.newBuilder();
      value.setName(name);
      value.setApiContext(env);

      // TODO: Figure out if number of cases can be reduced.
      if (name.equals("True") || name.equals("False")) { // Special case for a few well known symbols.
        value.setType("bool");
      } else if (name.equals("None")) {
        value.setType("NoneType");
      } else if (obj instanceof BuiltinFunction builtinFunction) {
        // Samples: depset, int, package, rule, str
        fillForStarlarkMethod(value, builtinFunction.getAnnotation());
      } else if (obj instanceof RuleFunction fn) {
        // Samples: alias, cc_binary, filegroup, toolchain, xcode_config
        fillForRuleFunction(value, fn);
      } else if (obj instanceof StarlarkFunction fn) {
        // Samples: cc_proto_binary, java_binary, py_binary
        fillForStarlarkFunction(value, fn);
      } else if (obj instanceof StarlarkProvider p) {
        // Samples: CcSharedLibraryInfo, JavaInfo, ProtoInfo, PyInfo
        fillForStarlarkProvider(value, p);
      } else if (obj instanceof StarlarkInfoNoSchema info) {
        // Samples: cc_common, java_common, native
        // TODO: Export as Type.
        value.setType("CLASS: " + obj.getClass().getName());
      } else if (obj instanceof BuiltinProvider p) {
        // Samples: CcInfo, DefaultInfo, InstrumentedFilesInfo, struct
        fillForBuiltinProvider(value, p);
      } else if (obj instanceof StarlarkAspect aspect) {
        // Sample: cc_proto_aspect
        fillForStarlarkAspect(value, aspect);
      } else {
          StarlarkBuiltin starlarkBuiltin = StarlarkAnnotations.getStarlarkBuiltin(obj.getClass());
          if (starlarkBuiltin != null) {
            // Sample: attr, config, json, proto,
            // TODO: Export as Type, use Starlark.getMethodAnnotations to collect all available methods.
            value.setDoc(starlarkBuiltin.doc());
          }
        // TODO: Handle more gracefully.
        value.setType("CLASS: " + obj.getClass().getName());
      }

      builtins.addGlobal(value.build());
    }
  }

  private static void fillForStarlarkFunction(Value.Builder value, StarlarkFunction fn) {
    Signature sig = new Signature();
    sig.doc = fn.getDocumentation();
    sig.parameterNames = fn.getParameterNames();
    sig.hasVarargs = fn.hasVarargs();
    sig.hasKwargs = fn.hasKwargs();
    sig.getDefaultValue =
            (i) -> {
              Object v = fn.getDefaultValue(i);
              return v == null ? null : Starlark.repr(v);
            };
    signatureToValue(value, sig);
  }

  private static void fillForStarlarkAspect(Value.Builder value, StarlarkAspect aspect) {
    Signature sig = new Signature();

    sig.parameterNames = new ArrayList<>();
    sig.paramDocs = new ArrayList<>();
    for (String fieldName : aspect.getParamAttributes()) {
      sig.parameterNames.add(fieldName);
      sig.paramDocs.add("");
    }
    signatureToValue(value, sig);
  }

  private static void fillForBuiltinProvider(Value.Builder value, BuiltinProvider p) {
    Signature sig = new Signature();
    sig.parameterNames = new ArrayList<>();
    sig.paramDocs = new ArrayList<>();

    Class valueClass = p.getValueClass();

    Method selfCall = Starlark.getSelfCallMethod(StarlarkSemantics.DEFAULT, p.getClass());
    if (selfCall != null) {
      StarlarkMethod m = StarlarkAnnotations.getStarlarkMethod(selfCall);
      if (m != null) {
        sig = getSignature(m);
      } else {
        // TODO: Handle.
        value.setType("CLASS: " + valueClass);
        return;
      }
    } else {
      // TODO: Handle.
      value.setType("CLASS: " + valueClass);
      return;
    }

    // Override doc with documentation of type itself, otherwise it will have simple
    // "Constructor for type" documentation.
    StarlarkBuiltin starlarkBuiltin = StarlarkAnnotations.getStarlarkBuiltin(valueClass);
    sig.doc = starlarkBuiltin.doc();

    signatureToValue(value, sig);
  }

  // TODO: Order methods.
  private static void fillForStarlarkProvider(Value.Builder value, StarlarkProvider p) {
    Signature sig = new Signature();
    sig.doc = p.getDocumentation().orElseGet(() -> "NO DOC");

    sig.parameterNames = new ArrayList<>();
    sig.paramDocs = new ArrayList<>();
    if (p.getFields() != null) {
      Map<String, Optional<String>> schema = p.getSchema();
      for (String fieldName : p.getFields()) {
        sig.parameterNames.add(fieldName);

        Optional<String> fieldDoc = schema.get(fieldName);
        sig.paramDocs.add(fieldDoc.orElse(null));
      }
    }
    signatureToValue(value, sig);
  }

  private static void fillForRuleFunction(Value.Builder value, RuleFunction fn) {
    RuleClass clz = fn.getRuleClass();
    Signature sig = new Signature();
    // TODO: Documentation for some native rules is in comments, like here:
    //     https://github.com/bazelbuild/bazel/blob/b8073bbcaa63c9405824f94184014b19a2255a52/src/main/java/com/google/devtools/build/lib/rules/Alias.java#L110
    //     and its parsed out by BuildDocCollector from source files...
    sig.doc = clz.getStarlarkDocumentation();
    sig.parameterNames = new ArrayList<>();
    sig.paramDocs = new ArrayList<>();
    List<Object> defaultValues = new ArrayList<>();
    for (Attribute attr : clz.getAttributes()) {
      // Remove all undocumented attributes, including e.g. generator_{name,function,location}, or attributes
      // with names beggining with $ and :.
      // TODO: Provide better way of marking attributes that are not writable in Stalark, undocumented does not imply
      //   that attribute cannot be used in Stalark files.
      if (!attr.isDocumented()) {
        continue;
      }
      sig.parameterNames.add(attr.getName());
      // TODO: Common attributes do not have documentation.
      sig.paramDocs.add(attr.getDoc());
      if (!attr.isMandatory()) {
        // TODO: Support default values.
        defaultValues.add(attr.getType().getDefaultValue());
      } else {
        defaultValues.add(null);
      }
    }
    sig.getDefaultValue = i -> defaultValues.get(i) != null ? String.valueOf(defaultValues.get(i)) : null;
    signatureToValue(value, sig);
  }

  // ------------------------------------------------
  // ----- TODO: Code chopped from ApiExporter
  private static void fillForStarlarkMethod(Value.Builder value, StarlarkMethod annot) {
    signatureToValue(value, getSignature(annot));
  }

  private static Value.Builder signatureToValue(Value.Builder value, Signature sig) {
    // TODO: E.g. cc_proto_library that is autoloaded does not have doc.
    if (sig.doc != null) {
      value.setDoc(sig.doc);
    }

    int nparams = sig.parameterNames.size();
    int kwargsIndex = sig.hasKwargs ? --nparams : -1;
    int varargsIndex = sig.hasVarargs ? --nparams : -1;
    // Inv: nparams is number of regular parameters.

    BuiltinProtos.Callable.Builder callable = BuiltinProtos.Callable.newBuilder();
    for (int i = 0; i < sig.parameterNames.size(); i++) {
      String name = sig.parameterNames.get(i);
      BuiltinProtos.Param.Builder param = BuiltinProtos.Param.newBuilder();
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
      // TODO: Validate when it can be null.
      if (sig.paramDocs != null && sig.paramDocs.get(i) != null) {
        param.setDoc(sig.paramDocs.get(i));
      }
      callable.addParam(param);
    }
    value.setCallable(callable);
    return value;
  }


  // Extracts signature and parameter default value expressions from a StarlarkMethod annotation.
  private static Signature getSignature(StarlarkMethod annot) {
    // Build-time annotation processing ensures mandatory parameters do not follow optional ones.
    boolean hasStar = false;
    String star = null;
    String starStar = null;
    ArrayList<String> params = new ArrayList<>();
    ArrayList<String> defaults = new ArrayList<>();
    ArrayList<String> docs = new ArrayList<>();

    for (net.starlark.java.annot.Param param : annot.parameters()) {
      // Ignore undocumented parameters
      if (!param.documented()) {
        docs.add("");
        continue;
      }
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
      docs.add(param.doc());
    }

    // f(..., *args, ...)
    if (!annot.extraPositionals().name().isEmpty() && !hasStar) {
      star = annot.extraPositionals().name();
    }
    if (star != null) {
      params.add(star);
      docs.add(annot.extraPositionals().doc());
    }

    // f(..., **kwargs)
    if (!annot.extraKeywords().name().isEmpty()) {
      starStar = annot.extraKeywords().name();
      params.add(starStar);
      docs.add(annot.extraKeywords().doc());
    }

    Signature sig = new Signature();
    sig.doc = annot.doc();
    sig.parameterNames = params;
    sig.hasVarargs = star != null;
    sig.hasKwargs = starStar != null;
    sig.getDefaultValue = defaults::get;
    sig.paramDocs = docs;
    return sig;
  }

  private static class Signature {
    List<String> parameterNames;
    List<String> paramDocs;
    boolean hasVarargs;
    boolean hasKwargs;
    String doc;

    // Returns the string form of the ith default value, using the
    // index, ordering, and null Conventions of StarlarkFunction.getDefaultValue.
    Function<Integer, String> getDefaultValue = (i) -> null;
  }
}
