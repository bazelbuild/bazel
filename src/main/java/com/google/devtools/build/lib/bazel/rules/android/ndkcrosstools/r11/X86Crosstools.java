// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.r11;

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
class X86Crosstools {

  private final NdkPaths ndkPaths;
  private final StlImpl stlImpl;

  X86Crosstools(NdkPaths ndkPaths, StlImpl stlImpl) {
    this.ndkPaths = ndkPaths;
    this.stlImpl = stlImpl;
  }

  ImmutableList<CToolchain.Builder> createCrosstools() {

    ImmutableList.Builder<CToolchain.Builder> toolchains = ImmutableList.builder();

    /*
     * x86
     */

    // gcc 4.9
    toolchains.add(createX86Toolchain());

    // clang
    CToolchain.Builder x86Clang = createBaseX86ClangToolchain("x86", "i686")
        // Workaround for https://code.google.com/p/android/issues/detail?id=220159.
        .addCompilerFlag("-mstackrealign")
        .setToolchainIdentifier("x86-clang3.8")
        .setTargetCpu("x86")

        .addAllToolPath(ndkPaths.createClangToolpaths(
            "x86-4.9", "i686-linux-android", null))

        .setBuiltinSysroot(ndkPaths.createBuiltinSysroot("x86"));

    stlImpl.addStlImpl(x86Clang, "4.9");
    toolchains.add(x86Clang);

    /*
     * x86_64
     */

    CToolchain.Builder x8664 =
        createBaseX86Toolchain()
            .setToolchainIdentifier("x86_64-4.9")
            .setTargetCpu("x86_64")
            .setCompiler("gcc-4.9")
            .addAllToolPath(ndkPaths.createToolpaths("x86_64-4.9", "x86_64-linux-android"))
            .addAllCxxBuiltinIncludeDirectory(
                ndkPaths.createGccToolchainBuiltinIncludeDirectories(
                    "x86_64-4.9", "x86_64-linux-android", "4.9"))
            .setBuiltinSysroot(ndkPaths.createBuiltinSysroot("x86_64"))
            .addCompilerFlag("-fstack-protector-strong");

    stlImpl.addStlImpl(x8664, "4.9");
    toolchains.add(x8664);

    CToolchain.Builder x8664Clang =
        createBaseX86ClangToolchain("x86_64", "x86_64")
            .setToolchainIdentifier("x86_64-clang3.8")
            .setTargetCpu("x86_64")

            .addAllToolPath(ndkPaths.createClangToolpaths(
                "x86_64-4.9", "x86_64-linux-android", null))

            .setBuiltinSysroot(ndkPaths.createBuiltinSysroot("x86_64"));

    stlImpl.addStlImpl(x8664Clang, "4.9");
    toolchains.add(x8664Clang);

    ImmutableList<CToolchain.Builder> toolchainBuilders = toolchains.build();

    // x86_64 also sets "x86-linux-android"
    for (CToolchain.Builder toolchainBuilder : toolchainBuilders) {
      toolchainBuilder.setTargetSystemName("x86-linux-android");
    }

    return toolchainBuilders;
  }

  private CToolchain.Builder createX86Toolchain() {

    CToolchain.Builder toolchain = createBaseX86Toolchain()
        .setToolchainIdentifier("x86-4.9")
        .setTargetCpu("x86")
        .setCompiler("gcc-4.9")

        .addAllToolPath(ndkPaths.createToolpaths("x86-4.9", "i686-linux-android"))
        .addAllCxxBuiltinIncludeDirectory(
            ndkPaths.createGccToolchainBuiltinIncludeDirectories(
                "x86-4.9", "i686-linux-android", "4.9"))
        .setBuiltinSysroot(ndkPaths.createBuiltinSysroot("x86"))

        .addCompilerFlag("-fstack-protector-strong");

    stlImpl.addStlImpl(toolchain, "4.9");
    return toolchain;
  }

  private CToolchain.Builder createBaseX86Toolchain() {
    return CToolchain.newBuilder()
        // Compiler flags
        .addCompilerFlag("-ffunction-sections")
        .addCompilerFlag("-funwind-tables")
        .addCompilerFlag("-no-canonical-prefixes")
        .addCompilerFlag("-fno-canonical-system-headers")

        // Linker flags
        .addLinkerFlag("-no-canonical-prefixes")

        // Additional release flags
        .addCompilationModeFlags(
            CompilationModeFlags.newBuilder()
                .setMode(CompilationMode.OPT)
                .addCompilerFlag("-O2")
                .addCompilerFlag("-g")
                .addCompilerFlag("-DNDEBUG")
                .addCompilerFlag("-fomit-frame-pointer")
                .addCompilerFlag("-fstrict-aliasing")
                .addCompilerFlag("-funswitch-loops")
                .addCompilerFlag("-finline-limit=300"))

        // Additional debug flags
        .addCompilationModeFlags(
            CompilationModeFlags.newBuilder()
                .setMode(CompilationMode.DBG)
                .addCompilerFlag("-O0")
                .addCompilerFlag("-g")
                .addCompilerFlag("-fno-omit-frame-pointer")
                .addCompilerFlag("-fno-strict-aliasing"));
  }

  private CToolchain.Builder createBaseX86ClangToolchain(String x86Arch, String llvmArch) {

    String gccToolchain = ndkPaths.createGccToolchainPath(x86Arch + "-4.9");

    String llvmTriple = llvmArch + "-none-linux-android";

    return CToolchain.newBuilder()
        .setCompiler("clang3.8")

        .addCxxBuiltinIncludeDirectory(
            ndkPaths.createClangToolchainBuiltinIncludeDirectory(
                AndroidNdkCrosstoolsR11.CLANG_VERSION))

        // Compiler flags
        .addCompilerFlag("-gcc-toolchain")
        .addCompilerFlag(gccToolchain)
        .addCompilerFlag("-target")
        .addCompilerFlag(llvmTriple)
        .addCompilerFlag("-ffunction-sections")
        .addCompilerFlag("-funwind-tables")
        .addCompilerFlag("-fstack-protector-strong")
        .addCompilerFlag("-fPIC")
        .addCompilerFlag("-Wno-invalid-command-line-argument")
        .addCompilerFlag("-Wno-unused-command-line-argument")
        .addCompilerFlag("-no-canonical-prefixes")

        // Linker flags
        .addLinkerFlag("-gcc-toolchain")
        .addLinkerFlag(gccToolchain)
        .addLinkerFlag("-target")
        .addLinkerFlag(llvmTriple)
        .addLinkerFlag("-no-canonical-prefixes")

        // Additional release flags
        .addCompilationModeFlags(
            CompilationModeFlags.newBuilder()
                .setMode(CompilationMode.OPT)
                .addCompilerFlag("-O2")
                .addCompilerFlag("-g")
                .addCompilerFlag("-DNDEBUG")
                .addCompilerFlag("-fomit-frame-pointer")
                .addCompilerFlag("-fstrict-aliasing"))

        // Additional debug flags
        .addCompilationModeFlags(
            CompilationModeFlags.newBuilder()
                .setMode(CompilationMode.DBG)
                .addCompilerFlag("-O0")
                .addCompilerFlag("-g")
                .addCompilerFlag("-fno-omit-frame-pointer")
                .addCompilerFlag("-fno-strict-aliasing"));
  }
}
