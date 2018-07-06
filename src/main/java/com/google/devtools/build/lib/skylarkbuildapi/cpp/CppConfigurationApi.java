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
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkList;

/**
 * The C++ configuration fragment.
 */
@SkylarkModule(
  name = "cpp",
  doc = "A configuration fragment for C++.",
  category = SkylarkModuleCategory.CONFIGURATION_FRAGMENT
)
public interface CppConfigurationApi <InvalidConfigurationExceptionT extends Exception> {

  @SkylarkCallable(name = "compiler", structField = true, doc = "C++ compiler.")
  @Deprecated
  String getCompiler() throws EvalException;

  @SkylarkCallable(name = "libc", structField = true, doc = "libc version string.")
  @Deprecated
  String getTargetLibc() throws EvalException;

  @SkylarkCallable(name = "cpu", structField = true, doc = "Target CPU of the C++ toolchain.")
  @Deprecated
  String getTargetCpu() throws EvalException;

  @SkylarkCallable(
    name = "built_in_include_directories",
    structField = true,
    doc =
        "Built-in system include paths for the toolchain compiler. All paths in this list"
            + " should be relative to the exec directory. They may be absolute if they are also"
            + " installed on the remote build nodes or for local compilation."
  )
  ImmutableList<String> getBuiltInIncludeDirectoriesForSkylark()
      throws InvalidConfigurationExceptionT, EvalException;

  @SkylarkCallable(name = "sysroot", structField = true,
      doc = "Returns the sysroot to be used. If the toolchain compiler does not support "
      + "different sysroots, or the sysroot is the same as the default sysroot, then "
      + "this method returns <code>None</code>.")
  String getSysroot() throws EvalException;

  @SkylarkCallable(
    name = "compiler_options",
    doc =
        "<b>Deprecated.</b> Returns the default options to use for compiling C, C++, and "
            + "assembler. This is just the options that should be used for all three languages. "
            + "There may be additional C-specific or C++-specific options that should be used, "
            + "in addition to the ones returned by this method",
    parameters = {
      @Param(
          name = "features",
          doc = "Unused.",
          positional = true,
          named = false,
          type = SkylarkList.class
      )
    }
  )
  @Deprecated
  ImmutableList<String> getCompilerOptions(Iterable<String> featuresNotUsedAnymore)
      throws EvalException;

  @SkylarkCallable(
      name = "c_options",
      structField = true,
      doc =
          "Returns the list of additional C-specific options to use for compiling C. "
              + "These should be go on the command line after the common options returned by "
              + "<code>compiler_options</code>")
  ImmutableList<String> getCOptionsForSkylark() throws EvalException;

  @SkylarkCallable(
    name = "cxx_options",
    doc =
        "Returns the list of additional C++-specific options to use for compiling C++. "
            + "These should be go on the command line after the common options returned by "
            + "<code>compiler_options</code>",
    parameters = {
      @Param(
          name = "features",
          doc = "Unused.",
          positional = true,
          named = false,
          type = SkylarkList.class
      )
    }
  )
  @Deprecated
  ImmutableList<String> getCxxOptions(Iterable<String> featuresNotUsedAnymore)
      throws EvalException;

  @SkylarkCallable(
    name = "unfiltered_compiler_options",
    doc =
        "Returns the default list of options which cannot be filtered by BUILD "
            + "rules. These should be appended to the command line after filtering.",
    parameters = {
      @Param(
          name = "features",
          doc = "Unused.",
          positional = true,
          named = false,
          type = SkylarkList.class
      )
    }
  )
  ImmutableList<String> getUnfilteredCompilerOptionsWithLegacySysroot(
      Iterable<String> featuresNotUsedAnymore) throws EvalException;

  @SkylarkCallable(
    name = "link_options",
    structField = true,
    doc =
        "Returns the set of command-line linker options, including any flags "
            + "inferred from the command-line options."
  )
  ImmutableList<String> getLinkOptionsWithLegacySysroot() throws EvalException;

  @SkylarkCallable(
    name = "fully_static_link_options",
    doc =
        "Returns the immutable list of linker options for fully statically linked "
            + "outputs. Does not include command-line options passed via --linkopt or "
            + "--linkopts.",
    parameters = {
      @Param(
          name = "features",
          doc = "Unused.",
          positional = true,
          named = false,
          type = SkylarkList.class
      ),
      @Param(
          name = "shared_lib",
          doc = "If true, returns the link options for a shared library.",
          positional = true,
          named = false,
          type = Boolean.class
      )
    }
  )
  @Deprecated
  ImmutableList<String> getFullyStaticLinkOptions(
      Iterable<String> featuresNotUsedAnymore, Boolean sharedLib) throws EvalException;

  @SkylarkCallable(
    name = "mostly_static_link_options",
    doc =
        "Returns the immutable list of linker options for mostly statically linked "
            + "outputs. Does not include command-line options passed via --linkopt or "
            + "--linkopts.",
    parameters = {
      @Param(
          name = "features",
          doc = "Unused.",
          positional = true,
          named = false,
          type = SkylarkList.class
      ),
      @Param(
          name = "shared_lib",
          doc = "If true, returns the link options for a shared library.",
          positional = true,
          named = false,
          type = Boolean.class
      )
    }
  )
  @Deprecated
  ImmutableList<String> getMostlyStaticLinkOptions(
      Iterable<String> featuresNotUsedAnymore, Boolean sharedLib) throws EvalException;

  @SkylarkCallable(
    name = "dynamic_link_options",
    doc =
        "Returns the immutable list of linker options for artifacts that are not "
            + "fully or mostly statically linked. Does not include command-line options "
            + "passed via --linkopt or --linkopts.",
    parameters = {
      @Param(
          name = "features",
          doc = "Unused.",
          positional = true,
          named = false,
          type = SkylarkList.class
      ),
      @Param(
          name = "shared_lib",
          doc = "If true, returns the link options for a shared library.",
          positional = true,
          named = false,
          type = Boolean.class
      )
    }
  )
  @Deprecated
  ImmutableList<String> getDynamicLinkOptions(
      Iterable<String> featuresNotUsedAnymore, Boolean sharedLib) throws EvalException;

  @SkylarkCallable(name = "ld_executable", structField = true, doc = "Path to the linker binary.")
  String getLdExecutableForSkylark() throws EvalException;

  @SkylarkCallable(
    name = "objcopy_executable",
    structField = true,
    doc = "Path to GNU binutils 'objcopy' binary."
  )
  String getObjCopyExecutableForSkylark() throws EvalException;

  @SkylarkCallable(
    name = "compiler_executable",
    structField = true,
    doc = "Path to C/C++ compiler binary."
  )
  String getCppExecutableForSkylark() throws EvalException;

  @SkylarkCallable(
    name = "preprocessor_executable",
    structField = true,
    doc = "Path to C/C++ preprocessor binary."
  )
  String getCpreprocessorExecutableForSkylark() throws EvalException;

  @SkylarkCallable(
    name = "nm_executable",
    structField = true,
    doc = "Path to GNU binutils 'nm' binary."
  )
  String getNmExecutableForSkylark() throws EvalException;

  @SkylarkCallable(
    name = "objdump_executable",
    structField = true,
    doc = "Path to GNU binutils 'objdump' binary."
  )
  String getObjdumpExecutableForSkylark() throws EvalException;

  @SkylarkCallable(
    name = "ar_executable",
    structField = true,
    doc = "Path to GNU binutils 'ar' binary."
  )
  String getArExecutableForSkylark() throws EvalException;

  @SkylarkCallable(
    name = "strip_executable",
    structField = true,
    doc = "Path to GNU binutils 'strip' binary."
  )
  String getStripExecutableForSkylark() throws EvalException;

  @SkylarkCallable(name = "target_gnu_system_name", structField = true,
      doc = "The GNU System Name.")
  @Deprecated
  String getTargetGnuSystemName() throws EvalException;
}
