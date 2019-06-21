// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylarkbuildapi.cpp;

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.FilesToRunProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkRuleContextApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.ParamType;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Runtime.NoneType;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;

/**
 * Helper class for the C++ aspects of {py,java,go}_wrap_cc. Provides methods to create the swig and
 * C++ actions. The swig action translates the swig declaration of an interface into Python or Java
 * wrapper code that can then be used to access the wrapped native APIs. The swig action also
 * creates C++ wrapper code that gets compiled and linked into a library that is interfacing between
 * the Java and Python wrappers and the actual wrapped APIs.
 */
@SkylarkModule(name = "WrapCcHelperDoNotUse", doc = "", documented = false)
public interface WrapCcHelperApi<
    FeatureConfigurationT extends FeatureConfigurationApi,
    SkylarkRuleContextT extends SkylarkRuleContextApi,
    CcToolchainProviderT extends CcToolchainProviderApi<FeatureConfigurationT>,
    CompilationInfoT extends CompilationInfoApi,
    FileT extends FileApi,
    CcCompilationContextT extends CcCompilationContextApi,
    WrapCcIncludeProviderT extends WrapCcIncludeProviderApi> {

  @SkylarkCallable(
      name = "feature_configuration",
      documented = false,
      doc = "",
      parameters = {
        @Param(name = "ctx", positional = false, named = true, type = SkylarkRuleContextApi.class),
        @Param(
            name = "cc_toolchain",
            positional = false,
            named = true,
            type = CcToolchainProviderApi.class),
      })
  public FeatureConfigurationT skylarkGetFeatureConfiguration(
      SkylarkRuleContextT skylarkRuleContext, CcToolchainProviderT ccToolchain)
      throws EvalException, InterruptedException;

  @SkylarkCallable(
      name = "collect_transitive_swig_includes",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true, type = SkylarkRuleContextApi.class),
      })
  public SkylarkNestedSet skylarkCollectTransitiveSwigIncludes(
      SkylarkRuleContextT skylarkRuleContext);

  @SkylarkCallable(
      name = "create_compile_actions",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true, type = SkylarkRuleContextApi.class),
        @Param(
            name = "feature_configuration",
            positional = false,
            named = true,
            type = FeatureConfigurationApi.class),
        @Param(
            name = "cc_toolchain",
            positional = false,
            named = true,
            type = CcToolchainProviderApi.class),
        @Param(name = "cc_file", positional = false, named = true, type = FileApi.class),
        @Param(name = "header_file", positional = false, named = true, type = FileApi.class),
        @Param(
            name = "dep_compilation_contexts",
            positional = false,
            named = true,
            type = SkylarkList.class),
        @Param(name = "target_copts", positional = false, named = true, type = SkylarkList.class),
      })
  public CompilationInfoT skylarkCreateCompileActions(
      SkylarkRuleContextT skylarkRuleContext,
      FeatureConfigurationT featureConfiguration,
      CcToolchainProviderT ccToolchain,
      FileT ccFile,
      FileT headerFile,
      SkylarkList<CcCompilationContextT> depCcCompilationContexts,
      SkylarkList<String> targetCopts)
      throws EvalException, InterruptedException;

  @SkylarkCallable(
      name = "mangled_target_name",
      documented = false,
      doc = "",
      parameters = {
        @Param(name = "ctx", positional = false, named = true, type = SkylarkRuleContextApi.class),
      })
  public String skylarkGetMangledTargetName(SkylarkRuleContextT skylarkRuleContext)
      throws EvalException, InterruptedException;

  @SkylarkCallable(
      name = "wrap_cc_include_provider",
      doc = "",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true, type = SkylarkRuleContextApi.class),
        @Param(
            name = "swig_includes",
            positional = false,
            named = true,
            type = SkylarkNestedSet.class),
      })
  // TODO(plf): Not written in Skylark because of PythonRunfilesProvider.
  public WrapCcIncludeProviderT getWrapCcIncludeProvider(
      SkylarkRuleContextT skylarkRuleContext, SkylarkNestedSet swigIncludes)
      throws EvalException, InterruptedException;

  @SkylarkCallable(
      name = "register_swig_action",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true, type = SkylarkRuleContextApi.class),
        @Param(
            name = "cc_toolchain",
            positional = false,
            named = true,
            type = CcToolchainProviderApi.class),
        @Param(
            name = "feature_configuration",
            positional = false,
            named = true,
            noneable = false,
            type = FeatureConfigurationApi.class),
        @Param(
            name = "wrapper_compilation_context",
            positional = false,
            named = true,
            type = CcCompilationContextApi.class),
        @Param(
            name = "swig_includes",
            positional = false,
            named = true,
            type = SkylarkNestedSet.class),
        @Param(name = "swig_source", positional = false, named = true, type = FileApi.class),
        @Param(name = "sub_parameters", positional = false, named = true, type = SkylarkList.class),
        @Param(name = "cc_file", positional = false, named = true, type = FileApi.class),
        @Param(name = "header_file", positional = false, named = true, type = FileApi.class),
        @Param(name = "output_files", positional = false, named = true, type = SkylarkList.class),
        @Param(
            name = "out_dir",
            positional = false,
            named = true,
            noneable = true,
            defaultValue = "None",
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)}),
        @Param(
            name = "java_dir",
            positional = false,
            named = true,
            noneable = true,
            defaultValue = "None",
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)}),
        @Param(
            name = "auxiliary_inputs",
            positional = false,
            named = true,
            type = SkylarkNestedSet.class),
        @Param(name = "swig_attribute_name", positional = false, named = true, type = String.class),
        @Param(
            name = "zip_tool",
            positional = false,
            named = true,
            noneable = true,
            defaultValue = "None",
            allowedTypes = {
              @ParamType(type = FilesToRunProviderApi.class),
              @ParamType(type = NoneType.class)
            })
      })
  // TODO(plf): Write in Skylark when all 3 SWIG rules are in Skylark.
  public void registerSwigAction(
      SkylarkRuleContextT skylarkRuleContext,
      CcToolchainProviderT ccToolchain,
      FeatureConfigurationT featureConfiguration,
      CcCompilationContextT wrapperCcCompilationContext,
      SkylarkNestedSet swigIncludes,
      FileT swigSource,
      SkylarkList<String> subParameters,
      FileT ccFile,
      FileT headerFile,
      SkylarkList<FileT> outputFiles,
      Object outDir,
      Object javaDir,
      SkylarkNestedSet auxiliaryInputs,
      String swigAttributeName,
      Object zipTool)
      throws EvalException, InterruptedException;
}
