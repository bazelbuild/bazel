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
import com.google.devtools.build.lib.skylarkbuildapi.RunfilesApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkRuleContextApi;
import com.google.devtools.build.lib.skylarkbuildapi.platform.ConstraintValueInfoApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Sequence;

/**
 * Helper class for the C++ aspects of {py,java,go}_wrap_cc. Provides methods to create the swig and
 * C++ actions. The swig action translates the swig declaration of an interface into Python or Java
 * wrapper code that can then be used to access the wrapped native APIs. The swig action also
 * creates C++ wrapper code that gets compiled and linked into a library that is interfacing between
 * the Java and Python wrappers and the actual wrapped APIs.
 */
@SkylarkModule(
    name = "py_wrap_cc_helper_do_not_use",
    documented = false,
    doc = "",
    category = SkylarkModuleCategory.TOP_LEVEL_TYPE)
public interface PyWrapCcHelperApi<
        FileT extends FileApi,
        ConstraintValueT extends ConstraintValueInfoApi,
        SkylarkRuleContextT extends SkylarkRuleContextApi<ConstraintValueT>,
        CcInfoT extends CcInfoApi<FileT>,
        FeatureConfigurationT extends FeatureConfigurationApi,
        CcToolchainProviderT extends CcToolchainProviderApi<FeatureConfigurationT>,
        CompilationInfoT extends CompilationInfoApi<FileT>,
        CcCompilationContextT extends CcCompilationContextApi<FileT>,
        WrapCcIncludeProviderT extends WrapCcIncludeProviderApi>
    extends WrapCcHelperApi<
        FeatureConfigurationT,
        ConstraintValueT,
        SkylarkRuleContextT,
        CcToolchainProviderT,
        CompilationInfoT,
        FileT,
        CcCompilationContextT,
        WrapCcIncludeProviderT> {

  @SkylarkCallable(
      name = "py_extension_linkopts",
      doc = "",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true, type = SkylarkRuleContextApi.class),
      })
  // TODO(plf): PyExtension is not in Skylark.
  public Sequence<String> getPyExtensionLinkopts(SkylarkRuleContextT skylarkRuleContext)
      throws EvalException, InterruptedException;

  @SkylarkCallable(
      name = "get_transitive_python_sources",
      doc = "",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true, type = SkylarkRuleContextApi.class),
        @Param(name = "py_file", positional = false, named = true, type = FileApi.class),
      })
  // TODO(plf): Not written in Skylark because of PyCommon.
  public Depset getTransitivePythonSources(SkylarkRuleContextT skylarkRuleContext, FileT pyFile)
      throws EvalException, InterruptedException;

  @SkylarkCallable(
      name = "get_python_runfiles",
      doc = "",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true, type = SkylarkRuleContextApi.class),
        @Param(name = "files_to_build", positional = false, named = true, type = Depset.class),
      })
  // TODO(plf): Not written in Skylark because of PythonRunfilesProvider.
  public RunfilesApi getPythonRunfiles(SkylarkRuleContextT skylarkRuleContext, Depset filesToBuild)
      throws EvalException, InterruptedException;

  @SkylarkCallable(
      name = "py_wrap_cc_info",
      doc = "",
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = false, named = true, type = SkylarkRuleContextApi.class),
        @Param(name = "cc_info", positional = false, named = true, type = CcInfoApi.class),
      })
  // TODO(plf): PyWrapCcInfo is not written in Skylark because several native rules use it.
  public PyWrapCcInfoApi<FileT> getPyWrapCcInfo(
      SkylarkRuleContextT skylarkRuleContext, CcInfoT ccInfo)
      throws EvalException, InterruptedException;
}
