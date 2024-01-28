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

import com.google.devtools.build.docgen.annot.DocCategory;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.StarlarkValue;

/**
 * This is a dummy interface which is not deleted because stardoc does not yet work for providers.
 * Delete this and point to Starlark implementation once it does.
 */
@StarlarkBuiltin(
    name = "CcToolchainInfo",
    category = DocCategory.PROVIDER,
    doc = "Information about the C++ compiler being used.")
public interface CcToolchainProviderApi extends StarlarkValue {

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
  default void usePicForDynamicLibrariesFromStarlark(Object featureConfigurationApi) {
    throw new UnsupportedOperationException(
        "Native CcToolchainInfo API no longer exists, use Starlark provider instead.");
  }

  @StarlarkMethod(
      name = "built_in_include_directories",
      doc = "Returns the list of built-in directories of the compiler.",
      structField = true)
  default void getBuiltInIncludeDirectoriesAsStrings() {
    throw new UnsupportedOperationException(
        "Native CcToolchainInfo API no longer exists, use Starlark provider instead.");
  }

  @StarlarkMethod(
      name = "all_files",
      doc =
          "Returns all toolchain files (so they can be passed to actions using this "
              + "toolchain as inputs).",
      structField = true)
  default void getAllFilesForStarlark() {
    throw new UnsupportedOperationException(
        "Native CcToolchainInfo API no longer exists, use Starlark provider instead.");
  }

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
  default void getStaticRuntimeLibForStarlark(Object featureConfiguration) {
    throw new UnsupportedOperationException(
        "Native CcToolchainInfo API no longer exists, use Starlark provider instead.");
  }

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
  default void getDynamicRuntimeLibForStarlark(Object featureConfiguration) {
    throw new UnsupportedOperationException(
        "Native CcToolchainInfo API no longer exists, use Starlark provider instead.");
  }

  @StarlarkMethod(
      name = "sysroot",
      structField = true,
      doc =
          "Returns the sysroot to be used. If the toolchain compiler does not support "
              + "different sysroots, or the sysroot is the same as the default sysroot, then "
              + "this method returns <code>None</code>.")
  default void getSysroot() {
    throw new UnsupportedOperationException(
        "Native CcToolchainInfo API no longer exists, use Starlark provider instead.");
  }

  @StarlarkMethod(name = "compiler", structField = true, doc = "C++ compiler.")
  default void getCompiler() {
    throw new UnsupportedOperationException(
        "Native CcToolchainInfo API no longer exists, use Starlark provider instead.");
  }

  @StarlarkMethod(name = "libc", structField = true, doc = "libc version string.")
  default void getTargetLibc() {
    throw new UnsupportedOperationException(
        "Native CcToolchainInfo API no longer exists, use Starlark provider instead.");
  }

  @StarlarkMethod(name = "cpu", structField = true, doc = "Target CPU of the C++ toolchain.")
  default void getTargetCpu() {
    throw new UnsupportedOperationException(
        "Native CcToolchainInfo API no longer exists, use Starlark provider instead.");
  }

  @StarlarkMethod(name = "target_gnu_system_name", structField = true, doc = "The GNU System Name.")
  default void getTargetGnuSystemName() {
    throw new UnsupportedOperationException(
        "Native CcToolchainInfo API no longer exists, use Starlark provider instead.");
  }

  @StarlarkMethod(
      name = "objcopy_executable",
      structField = true,
      doc = "The path to the objcopy binary.")
  default void objcopyExecutable() {
    throw new UnsupportedOperationException(
        "Native CcToolchainInfo API no longer exists, use Starlark provider instead.");
  }

  @StarlarkMethod(
      name = "compiler_executable",
      structField = true,
      doc = "The path to the compiler binary.")
  default void compilerExecutable() {
    throw new UnsupportedOperationException(
        "Native CcToolchainInfo API no longer exists, use Starlark provider instead.");
  }

  @StarlarkMethod(
      name = "preprocessor_executable",
      structField = true,
      doc = "The path to the preprocessor binary.")
  default void preprocessorExecutable() {
    throw new UnsupportedOperationException(
        "Native CcToolchainInfo API no longer exists, use Starlark provider instead.");
  }

  @StarlarkMethod(name = "nm_executable", structField = true, doc = "The path to the nm binary.")
  default void nmExecutable() {
    throw new UnsupportedOperationException(
        "Native CcToolchainInfo API no longer exists, use Starlark provider instead.");
  }

  @StarlarkMethod(
      name = "objdump_executable",
      structField = true,
      doc = "The path to the objdump binary.")
  default void objdumpExecutable() {
    throw new UnsupportedOperationException(
        "Native CcToolchainInfo API no longer exists, use Starlark provider instead.");
  }

  @StarlarkMethod(name = "ar_executable", structField = true, doc = "The path to the ar binary.")
  default void arExecutable() {
    throw new UnsupportedOperationException(
        "Native CcToolchainInfo API no longer exists, use Starlark provider instead.");
  }

  @StarlarkMethod(
      name = "strip_executable",
      structField = true,
      doc = "The path to the strip binary.")
  default void stripExecutable() {
    throw new UnsupportedOperationException(
        "Native CcToolchainInfo API no longer exists, use Starlark provider instead.");
  }

  @StarlarkMethod(name = "ld_executable", structField = true, doc = "The path to the ld binary.")
  default void ldExecutable() {
    throw new UnsupportedOperationException(
        "Native CcToolchainInfo API no longer exists, use Starlark provider instead.");
  }

  @StarlarkMethod(
      name = "gcov_executable",
      structField = true,
      doc = "The path to the gcov binary.")
  default void gcovExecutable() {
    throw new UnsupportedOperationException(
        "Native CcToolchainInfo API no longer exists, use Starlark provider instead.");
  }
}
