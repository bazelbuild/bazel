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
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;

/** Information about the C++ toolchain. */
@SkylarkModule(name = "CcToolchainInfo", doc = "Information about the C++ compiler being used.")
public interface CcToolchainProviderApi extends ToolchainInfoApi {

  @SkylarkCallable(
      name = "use_pic_for_dynamic_libraries",
      doc =
          "Returns true if this rule's compilations should apply -fPIC, false otherwise. "
              + "Determines if we should apply -fPIC for this rule's C++ compilations. This "
              + "determination is generally made by the global C++ configuration settings "
              + "<code>needsPic</code> and <code>usePicForBinaries</code>. However, an individual "
              + "rule may override these settings by applying <code>-fPIC</code> to its "
              + "<code>nocopts</code> attribute. This allows incompatible rules to opt out of "
              + "global PIC settings.",
      structField = true)
  boolean usePicForDynamicLibraries();

  @SkylarkCallable(
      name = "built_in_include_directories",
      doc = "Returns the list of built-in directories of the compiler.",
      structField = true)
  public ImmutableList<String> getBuiltInIncludeDirectoriesAsStrings();

  @SkylarkCallable(
      name = "sysroot",
      structField = true,
      doc =
          "Returns the sysroot to be used. If the toolchain compiler does not support "
              + "different sysroots, or the sysroot is the same as the default sysroot, then "
              + "this method returns <code>None</code>.")
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
}
