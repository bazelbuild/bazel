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

package com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.r19;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.NdkPaths;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.StlImpl;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CompilationMode;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CompilationModeFlags;

/**
 * Crosstool definitions for x86. These values are based on the setup.mk files in the Android NDK
 * toolchain directories.
 */
final class X86Crosstools {
  private final NdkPaths ndkPaths;
  private final StlImpl stlImpl;
  private final String clangVersion;

  X86Crosstools(NdkPaths ndkPaths, StlImpl stlImpl, String clangVersion) {
    this.ndkPaths = ndkPaths;
    this.stlImpl = stlImpl;
    this.clangVersion = clangVersion;
  }

  ImmutableList<CToolchain.Builder> createCrosstools() {
    /** x86 */
    // clang
    CToolchain.Builder x86Clang =
        createBaseX86ClangToolchain("x86", "i686", "i686-linux-android")
            // Workaround for https://code.google.com/p/android/issues/detail?id=220159.
            .addCompilerFlag("-mstackrealign")
            .setToolchainIdentifier("x86-clang" + clangVersion)
            .setTargetCpu("x86")
            .addAllToolPath(ndkPaths.createClangToolpaths("x86-4.9", "i686-linux-android", null))
            .setBuiltinSysroot(ndkPaths.createBuiltinSysroot("x86"));

    stlImpl.addStlImpl(x86Clang, null);

    /** x86_64 */
    CToolchain.Builder x8664Clang =
        createBaseX86ClangToolchain("x86_64", "x86_64", "x86_64-linux-android")
            .setToolchainIdentifier("x86_64-clang" + clangVersion)
            .setTargetCpu("x86_64")
            .addAllToolPath(
                ndkPaths.createClangToolpaths("x86_64-4.9", "x86_64-linux-android", null))
            .setBuiltinSysroot(ndkPaths.createBuiltinSysroot("x86_64"));

    stlImpl.addStlImpl(x8664Clang, null);

    return ImmutableList.of(x86Clang, x8664Clang);
  }

  private CToolchain.Builder createBaseX86ClangToolchain(
      String x86Arch, String llvmArch, String triple) {
    String gccToolchain = ndkPaths.createGccToolchainPath(x86Arch + "-4.9");
    String llvmTriple = llvmArch + "-none-linux-android";

    CToolchain.Builder cToolchainBuilder = CToolchain.newBuilder();

    cToolchainBuilder
        .setCompiler("clang" + clangVersion)
        .addCxxBuiltinIncludeDirectory(
            ndkPaths.createClangToolchainBuiltinIncludeDirectory(clangVersion))

        // Compiler flags
        .addCompilerFlag("-gcc-toolchain")
        .addCompilerFlag(gccToolchain)
        .addCompilerFlag("-target")
        .addCompilerFlag(llvmTriple)
        .addCompilerFlag("-fPIC")
        .addCompilerFlag(
            "-isystem%ndk%/usr/include/%triple%"
                .replace("%ndk%", ndkPaths.createBuiltinSysroot())
                .replace("%triple%", triple))
        .addCompilerFlag("-D__ANDROID_API__=" + ndkPaths.getCorrectedApiLevel(x86Arch))

        // Linker flags
        .addLinkerFlag("-gcc-toolchain")
        .addLinkerFlag(gccToolchain)
        .addLinkerFlag("-target")
        .addLinkerFlag(llvmTriple)

        // Additional release flags
        .addCompilationModeFlags(
            CompilationModeFlags.newBuilder()
                .setMode(CompilationMode.OPT)
                .addCompilerFlag("-O2")
                .addCompilerFlag("-g")
                .addCompilerFlag("-DNDEBUG"))

        // Additional debug flags
        .addCompilationModeFlags(
            CompilationModeFlags.newBuilder()
                .setMode(CompilationMode.DBG)
                .addCompilerFlag("-O0")
                .addCompilerFlag("-g"))
        .setTargetSystemName("x86-linux-android");

    if (Integer.parseInt(ndkPaths.getCorrectedApiLevel(x86Arch)) < 24) {
      // "For x86 targets prior to Android Nougat (API 24), -mstackrealign is needed to properly
      // align stacks for global constructors. See Issue 635."
      // https://android.googlesource.com/platform/ndk/+/ndk-release-r19/docs/BuildSystemMaintainers.md#additional-required-arguments
      cToolchainBuilder.addCompilerFlag("-mstackrealign");
    }

    return cToolchainBuilder;
  }
}
