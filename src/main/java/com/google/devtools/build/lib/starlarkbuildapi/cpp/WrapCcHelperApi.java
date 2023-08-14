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

package com.google.devtools.build.lib.starlarkbuildapi.cpp;

import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.FilesToRunProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkRuleContextApi;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ConstraintValueInfoApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkValue;

/**
 * Helper class for the C++ aspects of {py,java,go}_wrap_cc. Provides methods to create the swig and
 * C++ actions. The swig action translates the swig declaration of an interface into Python or Java
 * wrapper code that can then be used to access the wrapped native APIs. The swig action also
 * creates C++ wrapper code that gets compiled and linked into a library that is interfacing between
 * the Java and Python wrappers and the actual wrapped APIs.
 */
@StarlarkBuiltin(name = "WrapCcHelperDoNotUse", doc = "", documented = false)
public interface WrapCcHelperApi<
        FeatureConfigurationT extends FeatureConfigurationApi,
        ConstraintValueT extends ConstraintValueInfoApi,
        starlarkRuleContextT extends StarlarkRuleContextApi<ConstraintValueT>,
        CcToolchainProviderT extends
            CcToolchainProviderApi<
                    FeatureConfigurationT,
                    ?,
                    ?,
                    ConstraintValueT,
                    starlarkRuleContextT,
                    ?,
                    ? extends CppConfigurationApi<?>,
                    ?>,
        CompilationInfoT extends CompilationInfoApi<FileT>,
        FileT extends FileApi,
        CcCompilationContextT extends CcCompilationContextApi<FileT>,
        WrapCcIncludeProviderT extends WrapCcIncludeProviderApi>
    extends StarlarkValue {

  @StarlarkMethod(
      name = "collect_transitive_swig_includes",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
      })
  public Depset starlarkCollectTransitiveSwigIncludes(starlarkRuleContextT starlarkRuleContext);

  @StarlarkMethod(
      name = "mangled_target_name",
      documented = false,
      doc = "",
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
      })
  public String starlarkGetMangledTargetName(starlarkRuleContextT starlarkRuleContext)
      throws EvalException, InterruptedException;

  @StarlarkMethod(
      name = "wrap_cc_include_provider",
      doc = "",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
        @Param(name = "swig_includes", positional = false, named = true),
      })
  // TODO(plf): Not written in Starlark because of PythonRunfilesProvider.
  public WrapCcIncludeProviderT getWrapCcIncludeProvider(
      starlarkRuleContextT starlarkRuleContext, Depset swigIncludes)
      throws EvalException, InterruptedException;

  @StarlarkMethod(
      name = "register_swig_action",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
        @Param(name = "cc_toolchain", positional = false, named = true),
        @Param(name = "feature_configuration", positional = false, named = true),
        @Param(name = "wrapper_compilation_context", positional = false, named = true),
        @Param(name = "swig_includes", positional = false, named = true),
        @Param(name = "swig_source", positional = false, named = true),
        @Param(name = "sub_parameters", positional = false, named = true),
        @Param(name = "cc_file", positional = false, named = true),
        @Param(name = "header_file", positional = false, named = true),
        @Param(name = "output_files", positional = false, named = true),
        @Param(
            name = "out_dir",
            positional = false,
            named = true,
            defaultValue = "None",
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)}),
        @Param(
            name = "java_dir",
            positional = false,
            named = true,
            defaultValue = "None",
            allowedTypes = {@ParamType(type = String.class), @ParamType(type = NoneType.class)}),
        @Param(name = "auxiliary_inputs", positional = false, named = true),
        @Param(name = "swig_attribute_name", positional = false, named = true),
        @Param(
            name = "zip_tool",
            positional = false,
            named = true,
            defaultValue = "None",
            allowedTypes = {
              @ParamType(type = FilesToRunProviderApi.class),
              @ParamType(type = NoneType.class)
            })
      })
  // TODO(plf): Write in Starlark when all 3 SWIG rules are in Starlark.
  public void registerSwigAction(
      starlarkRuleContextT starlarkRuleContext,
      CcToolchainProviderT ccToolchain,
      FeatureConfigurationT featureConfiguration,
      CcCompilationContextT wrapperCcCompilationContext,
      Depset swigIncludes,
      FileT swigSource,
      Sequence<?> subParameters, // <String> expected
      FileT ccFile,
      FileT headerFile,
      Sequence<?> outputFiles, // <FileT> expected
      Object outDir,
      Object javaDir,
      Depset auxiliaryInputs,
      String swigAttributeName,
      Object zipTool)
      throws EvalException, InterruptedException;
}
