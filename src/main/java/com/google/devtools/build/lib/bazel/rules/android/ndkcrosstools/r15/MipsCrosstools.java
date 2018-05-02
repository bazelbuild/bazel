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

package com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.r15;

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
  private final String clangVersion;

  MipsCrosstools(NdkPaths ndkPaths, StlImpl stlImpl, String clangVersion) {
    this.ndkPaths = ndkPaths;
    this.stlImpl = stlImpl;
    this.clangVersion = clangVersion;
  }

  ImmutableList<CToolchain.Builder> createCrosstools() {
    return ImmutableList.of(createMips64Toolchain(), createMipsToolchain());
  }

  private CToolchain.Builder createMips64Toolchain() {
    String targetPlatform = "mips64el-linux-android";
    String targetCpu = "mips64";
    CToolchain.Builder mips64Clang =
        createBaseMipsClangToolchain("mips64el")
            .setToolchainIdentifier("mips64el-linux-android-clang" + clangVersion)
            .setTargetSystemName(targetPlatform)
            .setTargetCpu(targetCpu)
            .addAllToolPath(
                ndkPaths.createClangToolpaths(
                    "mips64el-linux-android-4.9", targetPlatform, null, CppConfiguration.Tool.DWP))
            .addCompilerFlag(
                "-isystem%ndk%/usr/include/%triple%"
                    .replace("%ndk%", ndkPaths.createBuiltinSysroot())
                    .replace("%triple%", targetPlatform))
            .addCompilerFlag("-D__ANDROID_API__=" + ndkPaths.getCorrectedApiLevel(targetCpu))
            .setBuiltinSysroot(ndkPaths.createBuiltinSysroot("mips64"));

    stlImpl.addStlImpl(mips64Clang, "4.9");
    return mips64Clang;
  }

  private CToolchain.Builder createMipsToolchain() {
    String targetPlatform = "mipsel-linux-android";
    String targetCpu = "mips";
    CToolchain.Builder mipsClang =
        createBaseMipsClangToolchain("mipsel")
            // Purposefully no hyphen between "clang" and clang version.
            .setToolchainIdentifier("mipsel-linux-android-clang" + clangVersion)
            .setTargetSystemName("mipsel-linux-android")
            .setTargetCpu(targetCpu)
            .addCompilerFlag(
                "-isystem%ndk%/usr/include/%triple%"
                    .replace("%ndk%", ndkPaths.createBuiltinSysroot())
                    .replace("%triple%", targetPlatform))
            .addCompilerFlag("-D__ANDROID_API__=" + ndkPaths.getCorrectedApiLevel("mips"))
            .addAllToolPath(
                ndkPaths.createClangToolpaths(
                    "mipsel-linux-android-4.9",
                    targetPlatform,
                    null,
                    CppConfiguration.Tool.DWP,
                    CppConfiguration.Tool.GCOVTOOL))
            .setBuiltinSysroot(ndkPaths.createBuiltinSysroot(targetCpu));

    stlImpl.addStlImpl(mipsClang, "4.9");

    return mipsClang;
  }

  private CToolchain.Builder createBaseMipsClangToolchain(String mipsArch) {
    String gccToolchain =
        ndkPaths.createGccToolchainPath(String.format("%s-linux-android-4.9", mipsArch));

    String llvmTriple = mipsArch + "-none-linux-android";

    return CToolchain.newBuilder()
        .setCompiler("clang" + clangVersion)
        .addCxxBuiltinIncludeDirectory(
            ndkPaths.createClangToolchainBuiltinIncludeDirectory(clangVersion))

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
