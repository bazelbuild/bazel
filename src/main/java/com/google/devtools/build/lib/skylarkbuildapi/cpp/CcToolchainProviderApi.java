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

package com.google.devtools.build.lib.skylarkbuildapi.cpp;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skylarkbuildapi.platform.ToolchainInfoApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import javax.annotation.Nullable;

/** Information about the C++ toolchain. */
@SkylarkModule(name = "CcToolchainInfo", doc = "Information about the C++ compiler being used.")
public interface CcToolchainProviderApi<FeatureConfigurationT extends FeatureConfigurationApi>
    extends ToolchainInfoApi {

  @SkylarkCallable(
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
            named = true,
            type = FeatureConfigurationApi.class)
      })
  boolean usePicForDynamicLibrariesFromStarlark(FeatureConfigurationT featureConfigurationApi);

  @SkylarkCallable(
      name = "built_in_include_directories",
      doc = "Returns the list of built-in directories of the compiler.",
      structField = true)
  public ImmutableList<String> getBuiltInIncludeDirectoriesAsStrings();

  @SkylarkCallable(
      name = "all_files",
      doc =
          "Returns all toolchain files (so they can be passed to actions using this "
              + "toolchain as inputs).",
      structField = true)
  public SkylarkNestedSet getAllFilesForStarlark();

  @SkylarkCallable(
      name = "sysroot",
      structField = true,
      allowReturnNones = true,
      doc =
          "Returns the sysroot to be used. If the toolchain compiler does not support "
              + "different sysroots, or the sysroot is the same as the default sysroot, then "
              + "this method returns <code>None</code>.")
  @Nullable
  public String getSysroot();

  @SkylarkCallable(name = "compiler", structField = true, doc = "C++ compiler.",
      allowReturnNones = true)
  public String getCompiler();

  @SkylarkCallable(name = "libc", structField = true, doc = "libc version string.",
      allowReturnNones = true)
  public String getTargetLibc();

  @SkylarkCallable(name = "cpu", structField = true, doc = "Target CPU of the C++ toolchain.",
      allowReturnNones = true)
  public String getTargetCpu();

  @SkylarkCallable(
    name = "target_gnu_system_name",
    structField = true,
    doc = "The GNU System Name.",
    allowReturnNones = true
  )
  public String getTargetGnuSystemName();

  @SkylarkCallable(
      name = "ld_executable",
      doc =
          "Path to the linker binary. \n WARNING: This method is only added to allow incremental"
              + "migration of existing users. Please do not use in new code. Will be removed soon"
              + "as part of the new Starlark API to the C++ toolchain.")
  public String getLdExecutableForSkylark();

  @SkylarkCallable(
      name = "objcopy_executable",
      doc =
          "Path to GNU binutils 'objcopy' binary. \n WARNING: This method is only added to allow"
              + "incremental migration of existing users. Please do not use in new code. Will be"
              + "removed soon as part of the new Starlark API to the C++ toolchain.")
  public String getObjCopyExecutableForSkylark();

  @SkylarkCallable(
      name = "compiler_executable",
      doc =
          "Path to C/C++ compiler binary. \n WARNING: This method is only added to allow "
              + "incremental migration of existing users. Please do not use in new code. Will be"
              + "removed soon as part of the new Starlark API to the C++ toolchain.")
  public String getCppExecutableForSkylark();

  @SkylarkCallable(
      name = "preprocessor_executable",
      doc =
          "Path to C/C++ preprocessor binary. \n WARNING: This method is only added to allow"
              + "incremental migration of existing users. Please do not use in new code. Will be "
              + "removed soon as part of the new Starlark API to the C++ toolchain.")
  public String getCpreprocessorExecutableForSkylark();

  @SkylarkCallable(
      name = "nm_executable",
      doc =
          "Path to GNU binutils 'nm' binary. \n WARNING: This method is only added to allow"
              + "incremental migration of existing users. Please do not use in new code. Will be "
              + "removed soon as part of the new Starlark API to the C++ toolchain.")
  public String getNmExecutableForSkylark();

  @SkylarkCallable(
      name = "objdump_executable",
      doc =
          "Path to GNU binutils 'objdump' binary. \n WARNING: This method is only added to allow"
              + "incremental migration of existing users. Please do not use in new code. Will be "
              + "removed soon as part of the new Starlark API to the C++ toolchain.")
  public String getObjdumpExecutableForSkylark();

  @SkylarkCallable(
      name = "ar_executable",
      doc =
          "Path to GNU binutils 'ar' binary. \n WARNING: This method is only added to allow"
              + "incremental migration of existing users. Please do not use in new code. Will be "
              + "removed soon as part of the new Starlark API to the C++ toolchain.")
  public String getArExecutableForSkylark();

  @SkylarkCallable(
      name = "strip_executable",
      doc =
          "Path to GNU binutils 'strip' binary. \n WARNING: This method is only added to allow"
              + "incremental migration of existing users. Please do not use in new code. Will be "
              + "removed soon as part of the new Starlark API to the C++ toolchain.")
  public String getStripExecutableForSkylark();
}
