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

package com.google.devtools.build.lib.starlarkbuildapi.cpp;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ToolchainInfoApi;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkThread;

/** Information about the C++ toolchain. */
@StarlarkBuiltin(
    name = "CcToolchainInfo",
    category = DocCategory.PROVIDER,
    doc = "Information about the C++ compiler being used.")
public interface CcToolchainProviderApi<
        FeatureConfigurationT extends FeatureConfigurationApi,
        BranchFdoProfileT extends BranchFdoProfileApi,
        FdoContextT extends FdoContextApi<BranchFdoProfileT>>
    extends ToolchainInfoApi {

  @StarlarkMethod(
      name = "needs_pic_for_dynamic_libraries",
      doc =
          "Returns true if this rule's compilations should apply -fPIC, false otherwise. "
              + "Determines if we should apply -fPIC for this rule's C++ compilations depending "
              + "on the C++ toolchain and presence of `--force_pic` Bazel option.",
      parameters = {
        @Param(
            name = "feature_configuration",
            doc = "Feature configuration to be queried.",
            positional = false,
            named = true)
      })
  boolean usePicForDynamicLibrariesFromStarlark(FeatureConfigurationT featureConfigurationApi);

  @StarlarkMethod(
      name = "built_in_include_directories",
      doc = "Returns the list of built-in directories of the compiler.",
      structField = true)
  public ImmutableList<String> getBuiltInIncludeDirectoriesAsStrings();

  @StarlarkMethod(
      name = "all_files",
      doc =
          "Returns all toolchain files (so they can be passed to actions using this "
              + "toolchain as inputs).",
      structField = true)
  public Depset getAllFilesForStarlark();

  @StarlarkMethod(
      name = "static_runtime_lib",
      doc =
          "Returns the files from `static_runtime_lib` attribute (so they can be passed to actions "
              + "using this toolchain as inputs). The caller should check whether the "
              + "feature_configuration enables `static_link_cpp_runtimes` feature (if not, "
              + "neither `static_runtime_lib` nor `dynamic_runtime_lib` should be used), and "
              + "use `dynamic_runtime_lib` if dynamic linking mode is active.",
      parameters = {
        @Param(
            name = "feature_configuration",
            doc = "Feature configuration to be queried.",
            positional = false,
            named = true)
      })
  public Depset getStaticRuntimeLibForStarlark(FeatureConfigurationT featureConfiguration)
      throws EvalException;

  @StarlarkMethod(
      name = "dynamic_runtime_lib",
      doc =
          "Returns the files from `dynamic_runtime_lib` attribute (so they can be passed to"
              + " actions using this toolchain as inputs). The caller can check whether the "
              + "feature_configuration enables `static_link_cpp_runtimes` feature (if not, neither"
              + " `static_runtime_lib` nor `dynamic_runtime_lib` have to be used), and use"
              + " `static_runtime_lib` if static linking mode is active.",
      parameters = {
        @Param(
            name = "feature_configuration",
            doc = "Feature configuration to be queried.",
            positional = false,
            named = true)
      })
  public Depset getDynamicRuntimeLibForStarlark(FeatureConfigurationT featureConfiguration)
      throws EvalException;

  @StarlarkMethod(
      name = "sysroot",
      structField = true,
      allowReturnNones = true,
      doc =
          "Returns the sysroot to be used. If the toolchain compiler does not support "
              + "different sysroots, or the sysroot is the same as the default sysroot, then "
              + "this method returns <code>None</code>.")
  @Nullable
  public String getSysroot();

  @StarlarkMethod(
      name = "compiler",
      structField = true,
      doc = "C++ compiler.",
      allowReturnNones = true)
  @Nullable
  public String getCompiler();

  @StarlarkMethod(
      name = "libc",
      structField = true,
      doc = "libc version string.",
      allowReturnNones = true)
  @Nullable
  public String getTargetLibc();

  @StarlarkMethod(
      name = "cpu",
      structField = true,
      doc = "Target CPU of the C++ toolchain.",
      allowReturnNones = true)
  @Nullable
  public String getTargetCpu();

  @StarlarkMethod(
      name = "target_gnu_system_name",
      structField = true,
      doc = "The GNU System Name.",
      allowReturnNones = true)
  @Nullable
  public String getTargetGnuSystemName();

  @StarlarkMethod(name = "as_files", documented = false, useStarlarkThread = true)
  Depset getAsFilesForStarlark(StarlarkThread thread) throws EvalException;

  @StarlarkMethod(name = "ar_files", documented = false, useStarlarkThread = true)
  Depset getArFilesForStarlark(StarlarkThread thread) throws EvalException;

  @StarlarkMethod(name = "strip_files", documented = false, useStarlarkThread = true)
  Depset getStripFilesForStarlark(StarlarkThread thread) throws EvalException;

  @StarlarkMethod(name = "objcopy_files", documented = false, useStarlarkThread = true)
  Depset getObjcopyFilesForStarlark(StarlarkThread thread) throws EvalException;

  @StarlarkMethod(
      name = "tool_path",
      documented = false,
      useStarlarkThread = true,
      allowReturnNones = true,
      parameters = {@Param(name = "tool", positional = false, named = true)})
  @Nullable
  String getToolPathStringOrNoneForStarlark(String tool, StarlarkThread thread)
      throws EvalException;

  @StarlarkMethod(name = "solib_dir", documented = false, useStarlarkThread = true)
  String getSolibDirectoryForStarlark(StarlarkThread thread) throws EvalException;

  @StarlarkMethod(name = "dynamic_runtime_solib_dir", documented = false, useStarlarkThread = true)
  String getDynamicRuntimeSolibDirForStarlark(StarlarkThread thread) throws EvalException;

  @StarlarkMethod(name = "linker_files", documented = false, useStarlarkThread = true)
  Depset getLinkerFilesForStarlark(StarlarkThread thread) throws EvalException;

  @StarlarkMethod(name = "coverage_files", documented = false, useStarlarkThread = true)
  Depset getCoverageFilesForStarlark(StarlarkThread thread) throws EvalException;

  @StarlarkMethod(name = "fdo_context", documented = false, useStarlarkThread = true)
  FdoContextT getFdoContextForStarlark(StarlarkThread thread) throws EvalException;
}
