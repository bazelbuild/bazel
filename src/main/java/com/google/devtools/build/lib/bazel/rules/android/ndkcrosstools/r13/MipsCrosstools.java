// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.r13;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.NdkPaths;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.StlImpl;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CompilationMode;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CompilationModeFlags;

/**
 * Crosstool definitions for MIPS. These values are based on the setup.mk files in the Android NDK
 * toolchain directories.
 */
final class MipsCrosstools {
  private final NdkPaths ndkPaths;
  private final StlImpl stlImpl;

  MipsCrosstools(NdkPaths ndkPaths, StlImpl stlImpl) {
    this.ndkPaths = ndkPaths;
    this.stlImpl = stlImpl;
  }

  ImmutableList<CToolchain.Builder> createCrosstools() {
    return ImmutableList.of(createMips64Toolchain(), createMipsToolchain());
  }

  private CToolchain.Builder createMips64Toolchain() {
    CToolchain.Builder mips64Clang =
        createBaseMipsClangToolchain("mips64el")
            .setToolchainIdentifier("mips64el-linux-android-clang3.8")
            .setTargetSystemName("mips64el-linux-android")
            .setTargetCpu("mips64")
            .addAllToolPath(
                ndkPaths.createClangToolpaths(
                    "mips64el-linux-android-4.9",
                    "mips64el-linux-android",
                    null,
                    CppConfiguration.Tool.DWP))
            .setBuiltinSysroot(ndkPaths.createBuiltinSysroot("mips64"));

    //List<CToolchain.Builder> toolchains = toolchainsListBuilder.build();
    ndkPaths.addToolchainIncludePaths(
        mips64Clang, "mips64el-linux-android-4.9", "mips64el-linux-android", "4.9.x");
    stlImpl.addStlImpl(mips64Clang, "4.9");
    return mips64Clang;
  }

  private CToolchain.Builder createMipsToolchain() {
    CToolchain.Builder mipsClang =
        createBaseMipsClangToolchain("mipsel")
            // Purposefully no hyphen between "clang" and clang version.
            .setToolchainIdentifier("mipsel-linux-android-clang3.8")
            .setTargetSystemName("mipsel-linux-android")
            .setTargetCpu("mips")
            .addAllToolPath(
                ndkPaths.createClangToolpaths(
                    "mipsel-linux-android-4.9",
                    "mipsel-linux-android",
                    null,
                    CppConfiguration.Tool.DWP,
                    CppConfiguration.Tool.GCOVTOOL))
            .setBuiltinSysroot(ndkPaths.createBuiltinSysroot("mips"));

    ndkPaths.addToolchainIncludePaths(
        mipsClang, "mipsel-linux-android-4.9", "mipsel-linux-android", "4.9.x");
    stlImpl.addStlImpl(mipsClang, "4.9");

    return mipsClang;
  }

  private CToolchain.Builder createBaseMipsClangToolchain(String mipsArch) {
    String gccToolchain =
        ndkPaths.createGccToolchainPath(String.format("%s-linux-android-4.9", mipsArch));

    String llvmTriple = mipsArch + "-none-linux-android";

    return CToolchain.newBuilder()
        .setCompiler("clang3.8")

        // Compiler flags
        .addCompilerFlag("-gcc-toolchain")
        .addCompilerFlag(gccToolchain)
        .addCompilerFlag("-target")
        .addCompilerFlag(llvmTriple)
        .addCompilerFlag("-fpic")
        .addCompilerFlag("-fno-strict-aliasing")
        .addCompilerFlag("-finline-functions")
        .addCompilerFlag("-ffunction-sections")
        .addCompilerFlag("-funwind-tables")
        .addCompilerFlag("-fmessage-length=0")
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
                .addCompilerFlag("-DNDEBUG"))

        // Additional debug flags
        .addCompilationModeFlags(
            CompilationModeFlags.newBuilder()
                .setMode(CompilationMode.DBG)
                .addCompilerFlag("-O0")
                .addCompilerFlag("-g"));
  }
}
