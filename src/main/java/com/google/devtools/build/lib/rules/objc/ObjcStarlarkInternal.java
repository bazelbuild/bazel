// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TemplateVariableInfo;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesInfo;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.cpp.CppSemantics;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.shell.ShellUtils.TokenizationException;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkValue;

/** Utility methods for Objc rules in Starlark Builtins */
@StarlarkBuiltin(name = "objc_internal", category = DocCategory.BUILTIN, documented = false)
public class ObjcStarlarkInternal implements StarlarkValue {

  public static final String NAME = "objc_internal";

  /**
   * Converts a possibly NoneType object to the real object if it is not NoneType or returns the
   * default value if it is.
   */
  @SuppressWarnings("unchecked")
  public static <T> T convertFromNoneable(Object obj, @Nullable T defaultValue) {
    if (Starlark.UNBOUND == obj || Starlark.isNullOrNone(obj)) {
      return defaultValue;
    }
    return (T) obj;
  }

  @StarlarkMethod(
      name = "create_compilation_attributes",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
      })
  public CompilationAttributes createCompilationAttributes(StarlarkRuleContext starlarkRuleContext)
      throws EvalException {
    CompilationAttributes.Builder builder = new CompilationAttributes.Builder();

    CompilationAttributes.Builder.addHeadersFromRuleContext(
        builder, starlarkRuleContext.getRuleContext());
    CompilationAttributes.Builder.addIncludesFromRuleContext(
        builder, starlarkRuleContext.getRuleContext());
    CompilationAttributes.Builder.addSdkAttributesFromRuleContext(
        builder, starlarkRuleContext.getRuleContext());
    ImmutableMap<String, String> toolchainMap =
        starlarkRuleContext
            .getRuleContext()
            .getPrerequisite("$cc_toolchain", TemplateVariableInfo.PROVIDER)
            .getVariables();
    ImmutableMap<String, String> starlarkRuleContextMap =
        ImmutableMap.<String, String>builder().putAll(starlarkRuleContext.var()).build();
    List<String> copts = new ArrayList<>();
    for (String copt :
        starlarkRuleContext.getRuleContext().attributes().get("copts", Type.STRING_LIST)) {
      String expandedCopt = expandCopt(copt, toolchainMap, starlarkRuleContextMap);
      try {
        ShellUtils.tokenize(copts, expandedCopt);
      } catch (TokenizationException e) {
        throw new EvalException(e);
      }
    }
    CompilationAttributes.Builder.addCompileOptionsFromRuleContext(
        builder, starlarkRuleContext.getRuleContext(), copts);
    CompilationAttributes.Builder.addModuleOptionsFromRuleContext(
        builder, starlarkRuleContext.getRuleContext());

    return builder.build();
  }

  private String expandCopt(
      String copt,
      ImmutableMap<String, String> toolchainMap,
      ImmutableMap<String, String> contextMap) {
    if (!copt.contains("$(")) {
      return copt;
    }
    int beginning = copt.indexOf("$(");
    int end = copt.indexOf(')', beginning);
    String variable = copt.substring(beginning + 2, end);
    String expandedVariable;
    if (toolchainMap.containsKey(variable)) {
      expandedVariable = toolchainMap.get(variable);
    } else {
      expandedVariable = contextMap.get(variable);
    }
    String expandedCopt = copt.replace("$(" + variable + ")", expandedVariable);
    return expandCopt(expandedCopt, toolchainMap, contextMap);
  }

  @StarlarkMethod(
      name = "create_intermediate_artifacts",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
      })
  public IntermediateArtifacts createIntermediateArtifacts(
      StarlarkRuleContext starlarkRuleContext) {
    return ObjcRuleClasses.intermediateArtifacts(starlarkRuleContext.getRuleContext());
  }

  @StarlarkMethod(
      name = "create_compilation_artifacts",
      documented = false,
      parameters = {
        @Param(
            name = "ctx",
            positional = false,
            named = true,
            defaultValue = "None",
            allowedTypes = {
              @ParamType(type = StarlarkRuleContext.class),
              @ParamType(type = NoneType.class)
            }),
      })
  public CompilationArtifacts createCompilationArtifacts(Object starlarkRuleContextObject) {
    StarlarkRuleContext starlarkRuleContext =
        convertFromNoneable(starlarkRuleContextObject, /* defaultValue= */ null);
    if (starlarkRuleContext != null) {
      return CompilationSupport.compilationArtifacts(starlarkRuleContext.getRuleContext());
    } else {
      return new CompilationArtifacts.Builder().build();
    }
  }

  @StarlarkMethod(
      name = "j2objc_providers_from_deps",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
      })
  public Sequence<NativeInfo> createj2objcProvidersFromDeps(StarlarkRuleContext starlarkRuleContext)
      throws EvalException {
    J2ObjcMappingFileProvider j2ObjcMappingFileProvider =
        J2ObjcMappingFileProvider.union(
            starlarkRuleContext
                .getRuleContext()
                .getPrerequisites("deps", J2ObjcMappingFileProvider.PROVIDER));
    J2ObjcEntryClassProvider j2ObjcEntryClassProvider =
        new J2ObjcEntryClassProvider.Builder()
            .addTransitive(
                starlarkRuleContext
                    .getRuleContext()
                    .getPrerequisites("deps", J2ObjcEntryClassProvider.PROVIDER))
            .build();

    return StarlarkList.immutableCopyOf(
        ImmutableList.of(j2ObjcEntryClassProvider, j2ObjcMappingFileProvider));
  }

  @StarlarkMethod(
      name = "create_common",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
        @Param(name = "purpose", positional = false, named = true),
        @Param(name = "compilation_attributes", positional = false, named = true),
        @Param(name = "deps", positional = false, named = true),
        @Param(name = "runtime_deps", positional = false, named = true, defaultValue = "[]"),
        @Param(name = "intermediate_artifacts", positional = false, named = true),
        @Param(name = "alwayslink", positional = false, named = true),
        @Param(name = "has_module_map", positional = false, named = true),
        @Param(
            name = "extra_import_libraries",
            positional = false,
            defaultValue = "[]",
            named = true),
        @Param(name = "compilation_artifacts", positional = false, named = true)
      })
  public ObjcCommon createObjcCommon(
      StarlarkRuleContext starlarkRuleContext,
      String purpose,
      CompilationAttributes compilationAttributes,
      Sequence<?> deps,
      Sequence<?> runtimeDeps,
      IntermediateArtifacts intermediateArtifacts,
      boolean alwayslink,
      boolean hasModuleMap,
      Sequence<?> extraImportLibraries,
      CompilationArtifacts compilationArtifacts)
      throws InterruptedException, EvalException {
    ObjcCommon.Builder builder =
        new ObjcCommon.Builder(
                ObjcCommon.Purpose.valueOf(purpose), starlarkRuleContext.getRuleContext())
            .setCompilationAttributes(compilationAttributes)
            .addDeps(Sequence.cast(deps, TransitiveInfoCollection.class, "deps"))
            .addRuntimeDeps(
                Sequence.cast(runtimeDeps, TransitiveInfoCollection.class, "runtime_deps"))
            .setIntermediateArtifacts(intermediateArtifacts)
            .setAlwayslink(alwayslink)
            .addExtraImportLibraries(
                Sequence.cast(extraImportLibraries, Artifact.class, "archives"))
            .setCompilationArtifacts(compilationArtifacts);

    if (hasModuleMap) {
      builder.setHasModuleMap();
    }

    return builder.build();
  }

  @StarlarkMethod(
      name = "create_compilation_support",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
        @Param(name = "semantics", positional = false, named = true),
        @Param(name = "compilation_attributes", positional = false, named = true),
      })
  public CompilationSupport createCompilationSupport(
      StarlarkRuleContext starlarkRuleContext,
      CppSemantics cppSemantics,
      CompilationAttributes compilationAttributes)
      throws InterruptedException, EvalException {
    try {
      return new CompilationSupport.Builder(starlarkRuleContext.getRuleContext(), cppSemantics)
          .setCompilationAttributes(compilationAttributes)
          .build();
    } catch (RuleErrorException e) {
      throw new EvalException(e);
    }
  }

  @StarlarkMethod(
      name = "instrumented_files_info",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
        @Param(name = "object_files", positional = false, defaultValue = "[]", named = true),
      })
  public InstrumentedFilesInfo createInstrumentedFilesInfo(
      StarlarkRuleContext starlarkRuleContext, Sequence<?> objectFiles) throws EvalException {
    return CompilationSupport.getInstrumentedFilesProvider(
        starlarkRuleContext.getRuleContext(),
        Sequence.cast(objectFiles, Artifact.class, "object_files").getImmutableList());
  }
}
