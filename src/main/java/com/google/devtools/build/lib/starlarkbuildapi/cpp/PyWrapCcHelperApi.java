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

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.RunfilesApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkRuleContextApi;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ConstraintValueInfoApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;

/**
 * Helper class for the C++ aspects of {py,java,go}_wrap_cc. Provides methods to create the swig and
 * C++ actions. The swig action translates the swig declaration of an interface into Python or Java
 * wrapper code that can then be used to access the wrapped native APIs. The swig action also
 * creates C++ wrapper code that gets compiled and linked into a library that is interfacing between
 * the Java and Python wrappers and the actual wrapped APIs.
 */
@StarlarkBuiltin(
    name = "py_wrap_cc_helper_do_not_use",
    documented = false,
    doc = "",
    category = DocCategory.TOP_LEVEL_TYPE)
public interface PyWrapCcHelperApi<
        FileT extends FileApi,
        ConstraintValueT extends ConstraintValueInfoApi,
        StarlarkRuleContextT extends StarlarkRuleContextApi<ConstraintValueT>,
        CcInfoT extends CcInfoApi<FileT>,
        FeatureConfigurationT extends FeatureConfigurationApi,
        CcToolchainProviderT extends CcToolchainProviderApi<FeatureConfigurationT, ?, ?>,
        CompilationInfoT extends CompilationInfoApi<FileT>,
        CcCompilationContextT extends CcCompilationContextApi<FileT>,
        WrapCcIncludeProviderT extends WrapCcIncludeProviderApi>
    extends WrapCcHelperApi<
        FeatureConfigurationT,
        ConstraintValueT,
        StarlarkRuleContextT,
        CcToolchainProviderT,
        CompilationInfoT,
        FileT,
        CcCompilationContextT,
        WrapCcIncludeProviderT> {

  @StarlarkMethod(
      name = "py_extension_linkopts",
      doc = "",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
      })
  // TODO(plf): PyExtension is not in Starlark.
  public Sequence<String> getPyExtensionLinkopts(StarlarkRuleContextT starlarkRuleContext)
      throws EvalException, InterruptedException;

  @StarlarkMethod(
      name = "get_transitive_python_sources",
      doc = "",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
        @Param(name = "py_file", positional = false, named = true),
      })
  // TODO(plf): Not written in Starlark because of PyCommon.
  public Depset getTransitivePythonSources(StarlarkRuleContextT starlarkRuleContext, FileT pyFile)
      throws EvalException, InterruptedException;

  @StarlarkMethod(
      name = "get_python_runfiles",
      doc = "",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
        @Param(name = "files_to_build", positional = false, named = true),
      })
  // TODO(plf): Not written in Starlark because of PythonRunfilesProvider.
  public RunfilesApi getPythonRunfiles(
      StarlarkRuleContextT starlarkRuleContext, Depset filesToBuild)
      throws EvalException, InterruptedException;

  @StarlarkMethod(
      name = "py_wrap_cc_info",
      doc = "",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true),
        @Param(name = "cc_info", positional = false, named = true),
      })
  // TODO(plf): PyWrapCcInfo is not written in Starlark because several native rules use it.
  public PyWrapCcInfoApi<FileT> getPyWrapCcInfo(
      StarlarkRuleContextT starlarkRuleContext, CcInfoT ccInfo)
      throws EvalException, InterruptedException;
}
