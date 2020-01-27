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
 * Crosstool definitions for ARM. These values are based on the setup.mk files in the Android NDK
 * toolchain directories.
 */
final class ArmCrosstools {
  private final NdkPaths ndkPaths;
  private final StlImpl stlImpl;
  private final String clangVersion;

  ArmCrosstools(NdkPaths ndkPaths, StlImpl stlImpl, String clangVersion) {
    this.ndkPaths = ndkPaths;
    this.stlImpl = stlImpl;
    this.clangVersion = clangVersion;
  }

  ImmutableList<CToolchain.Builder> createCrosstools() {
    CToolchain.Builder aarch64Toolchain = createAarch64ClangToolchain();
    CToolchain.Builder armeabiToolchain = createArmeabiClangToolchain();

    stlImpl.addStlImpl(aarch64Toolchain, "4.9");
    stlImpl.addStlImpl(armeabiToolchain, "4.9");

    return ImmutableList.<CToolchain.Builder>builder()
        .add(aarch64Toolchain)
        .add(armeabiToolchain)
        .build();
  }

  private CToolchain.Builder createAarch64ClangToolchain() {
    String toolchainName = "aarch64-linux-android-4.9";
    String targetPlatform = "aarch64-linux-android";
    String gccToolchain = ndkPaths.createGccToolchainPath(toolchainName);
    String llvmTriple = "aarch64-none-linux-android";

    return CToolchain.newBuilder()
        .setToolchainIdentifier("aarch64-linux-android-clang" + clangVersion)
        .setTargetSystemName(targetPlatform)
        .setTargetCpu("arm64-v8a")
        .setCompiler("clang" + clangVersion)
        .addAllToolPath(ndkPaths.createClangToolpaths(toolchainName, targetPlatform, null))
        .addCxxBuiltinIncludeDirectory(
            ndkPaths.createClangToolchainBuiltinIncludeDirectory(clangVersion))
        .setBuiltinSysroot(ndkPaths.createBuiltinSysroot("arm64"))

        // Compiler flags
        .addCompilerFlag("-gcc-toolchain")
        .addCompilerFlag(gccToolchain)
        .addCompilerFlag("-target")
        .addCompilerFlag(llvmTriple)
        .addCompilerFlag("-fpic")
        .addCompilerFlag(
            "-isystem%ndk%/usr/include/%triple%"
                .replace("%ndk%", ndkPaths.createBuiltinSysroot())
                .replace("%triple%", targetPlatform))
        .addCompilerFlag("-D__ANDROID_API__=" + ndkPaths.getCorrectedApiLevel("arm"))

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
                .addCompilerFlag("-g")
                .addCompilerFlag("-UNDEBUG"));
  }

  private CToolchain.Builder createArmeabiClangToolchain() {
    String toolchainName = "arm-linux-androideabi-4.9";
    String targetPlatform = "arm-linux-androideabi";
    String gccToolchain = ndkPaths.createGccToolchainPath("arm-linux-androideabi-4.9");

    return CToolchain.newBuilder()
        .setToolchainIdentifier("arm-linux-androideabi-clang" + clangVersion + "-v7a")
        .setTargetCpu("armeabi-v7a")
        .setTargetSystemName("arm-linux-androideabi")
        .setCompiler("clang" + clangVersion)
        .addAllToolPath(ndkPaths.createClangToolpaths(toolchainName, targetPlatform, null))
        .addCxxBuiltinIncludeDirectory(
            ndkPaths.createClangToolchainBuiltinIncludeDirectory(clangVersion))
        .setBuiltinSysroot(ndkPaths.createBuiltinSysroot("arm"))
        .addCompilerFlag("-D__ANDROID_API__=" + ndkPaths.getCorrectedApiLevel("arm"))
        .addCompilerFlag(
            "-isystem%ndk%/usr/include/%triple%"
                .replace("%ndk%", ndkPaths.createBuiltinSysroot())
                .replace("%triple%", targetPlatform))

        // Compiler flags
        .addCompilerFlag("-target")
        .addCompilerFlag("armv7-none-linux-androideabi") // LLVM_TRIPLE
        .addCompilerFlag("-march=armv7-a")
        .addCompilerFlag("-mfloat-abi=softfp")
        // "32-bit ARM targets should use -mfpu=vfpv3-d16 when compiling unless using NEON. This
        // allows the compiler to make use of the FPU."
        // https://android.googlesource.com/platform/ndk/+/ndk-release-r19/docs/BuildSystemMaintainers.md#additional-required-arguments
        .addCompilerFlag("-mfpu=vfpv3-d16")
        .addCompilerFlag("-gcc-toolchain")
        .addCompilerFlag(gccToolchain)
        .addCompilerFlag("-fpic")

        // Linker flags
        .addLinkerFlag("-target")
        .addLinkerFlag("armv7-none-linux-androideabi") // LLVM_TRIPLE
        .addLinkerFlag("-gcc-toolchain")
        .addLinkerFlag(gccToolchain)

        // Additional release flags
        .addCompilationModeFlags(
            CompilationModeFlags.newBuilder()
                .setMode(CompilationMode.OPT)
                .addCompilerFlag("-mthumb")
                .addCompilerFlag("-Os")
                .addCompilerFlag("-g")
                .addCompilerFlag("-DNDEBUG"))

        // Additional debug flags
        .addCompilationModeFlags(
            CompilationModeFlags.newBuilder()
                .setMode(CompilationMode.DBG)
                .addCompilerFlag("-g")
                .addCompilerFlag("-fno-strict-aliasing")
                .addCompilerFlag("-O0")
                .addCompilerFlag("-UNDEBUG"));
  }
}
