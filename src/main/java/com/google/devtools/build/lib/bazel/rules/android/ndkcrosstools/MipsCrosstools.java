// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CompilationMode;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CompilationModeFlags;

/**
 * Crosstool definitions for MIPS. These values are based on the setup.mk files in the Android NDK
 * toolchain directories.
 */
class MipsCrosstools {

  private final NdkPaths ndkPaths;

  MipsCrosstools(NdkPaths ndkPaths) {
    this.ndkPaths = ndkPaths;
  }

  ImmutableList<CToolchain.Builder> createCrosstools() {

    ImmutableList.Builder<CToolchain.Builder> builder = ImmutableList.builder();

    /**
     * mips64
     */

    builder.add(
        createMipsToolchain()
            .setToolchainIdentifier("mips64el-linux-android-4.9")
            .setTargetSystemName("mips64el-linux-android")
            .setTargetCpu("mips64")
            .setCompiler("gcc-4.9")

            .addAllToolPath(
                ndkPaths.createToolpaths(
                    "mips64el-linux-android-4.9",
                    "mips64el-linux-android",
                    // mips64 toolchain doesn't have the dwp tool.
                    CppConfiguration.Tool.DWP))

            .addAllCxxBuiltinIncludeDirectory(
                ndkPaths.createToolchainIncludePaths(
                    "mips64el-linux-android-4.9", "mips64el-linux-android", "4.9"))

            .setBuiltinSysroot(ndkPaths.createBuiltinSysroot("mips64"))

            .setSupportsEmbeddedRuntimes(true)
            .setStaticRuntimesFilegroup("static-runtime-libs-" + "mips64el-linux-android-4.9")
            .setDynamicRuntimesFilegroup("dynamic-runtime-libs-" + "mips64el-linux-android-4.9"));

    // The flags for mips64 clang 3.5 and 3.6 are the same, they differ only in the LLVM version
    // given in their tool paths.
    for (String clangVersion : new String[] { "3.5", "3.6" }) {
      String toolchainIdentifier = "mips64el-linux-android-clang" + clangVersion;
      builder.add(
          createMipsClangToolchain("mips64", "4.9")
              .setToolchainIdentifier(toolchainIdentifier)
              .setTargetSystemName("mips64el-linux-android")
              .setTargetCpu("mips64")
              .setCompiler("gcc-4.9")

              .addAllToolPath(
                  ndkPaths.createClangToolpaths(
                      "mips64el-linux-android-4.9",
                      "mips64el-linux-android",
                      clangVersion,
                      CppConfiguration.Tool.DWP))

              .addAllCxxBuiltinIncludeDirectory(
                  ndkPaths.createToolchainIncludePaths(
                      "mips64el-linux-android-4.9", "mips64el-linux-android", "4.9"))

              .setBuiltinSysroot(ndkPaths.createBuiltinSysroot("mips64"))

              .setSupportsEmbeddedRuntimes(true)
              .setStaticRuntimesFilegroup("static-runtime-libs-" + toolchainIdentifier)
              .setDynamicRuntimesFilegroup("static-runtime-libs-" + toolchainIdentifier));
    }

    /**
     * mips
     */

    builder.add(
        createMipsToolchain()
            .setToolchainIdentifier("mipsel-linux-android-4.8")
            .setTargetSystemName("mipsel-linux-android")
            .setTargetCpu("mips")
            .setCompiler("gcc-4.8")

            .addAllToolPath(
                ndkPaths.createToolpaths(
                    "mipsel-linux-android-4.8",
                    "mipsel-linux-android",
                    // gcc-4.8 mips toolchain doesn't have dwp or gcov-tool.
                    CppConfiguration.Tool.DWP,
                    CppConfiguration.Tool.GCOVTOOL))

            .addAllCxxBuiltinIncludeDirectory(
                ndkPaths.createToolchainIncludePaths(
                    "mipsel-linux-android-4.8", "mipsel-linux-android", "4.8"))

            .setBuiltinSysroot(ndkPaths.createBuiltinSysroot("mips"))

            .setSupportsEmbeddedRuntimes(true)
            .setStaticRuntimesFilegroup("static-runtime-libs-" + "mipsel-linux-android-4.8")
            .setDynamicRuntimesFilegroup("static-runtime-libs-" + "mipsel-linux-android-4.8"));

    builder.add(
        createMipsToolchain()
            .setToolchainIdentifier("mipsel-linux-android-4.9")
            .setTargetSystemName("mipsel-linux-android")
            .setTargetCpu("mips")
            .setCompiler("gcc-4.9")

            .addAllToolPath(
                ndkPaths.createToolpaths(
                    "mipsel-linux-android-4.9",
                    "mipsel-linux-android",
                    // gcc-4.9 mips toolchain doesn't have the dwp tool.
                    CppConfiguration.Tool.DWP))

            .addAllCxxBuiltinIncludeDirectory(
                ndkPaths.createToolchainIncludePaths(
                    "mipsel-linux-android-4.9", "mipsel-linux-android", "4.9"))

            .setBuiltinSysroot(ndkPaths.createBuiltinSysroot("mips"))

            .setSupportsEmbeddedRuntimes(true)
            .setStaticRuntimesFilegroup("static-runtime-libs-" + "mipsel-linux-android-4.9")
            .setDynamicRuntimesFilegroup("dynamic-runtime-libs-" + "mipsel-linux-android-4.9"));

    // The flags for mips clang 3.5 and 3.6 are the same, they differ only in the LLVM version
    // given in their tool paths.
    for (String clangVersion : new String[] { "3.5", "3.6" }) {
      String toolchainIdentifier = "mipsel-linux-android-clang" + clangVersion;
      builder.add(
          createMipsClangToolchain("mips", "4.8")
              .setToolchainIdentifier(toolchainIdentifier)
              .setTargetSystemName("mipsel-linux-android")
              .setTargetCpu("mips")
              .setCompiler("gcc-4.8")

              .addAllToolPath(
                  ndkPaths.createClangToolpaths(
                      "mipsel-linux-android-4.8",
                      "mipsel-linux-android",
                      clangVersion,
                      CppConfiguration.Tool.DWP,
                      CppConfiguration.Tool.GCOVTOOL))

              .addAllCxxBuiltinIncludeDirectory(
                  ndkPaths.createToolchainIncludePaths(
                      "mipsel-linux-android-4.8", "mipsel-linux-android", "4.8"))

              .setBuiltinSysroot(ndkPaths.createBuiltinSysroot("mips"))

              .setSupportsEmbeddedRuntimes(true)
              .setStaticRuntimesFilegroup("static-runtime-libs-" + toolchainIdentifier)
              .setDynamicRuntimesFilegroup("dynamic-runtime-libs-" + toolchainIdentifier));
    }

    return builder.build();
  }

  private CToolchain.Builder createMipsToolchain() {
    return CToolchain.newBuilder()
        // Compiler flags
        .addCompilerFlag("-fpic")
        .addCompilerFlag("-fno-strict-aliasing")
        .addCompilerFlag("-finline-functions")
        .addCompilerFlag("-ffunction-sections")
        .addCompilerFlag("-funwind-tables")
        .addCompilerFlag("-fmessage-length=0")
        .addCompilerFlag("-fno-inline-functions-called-once")
        .addCompilerFlag("-fgcse-after-reload")
        .addCompilerFlag("-frerun-cse-after-loop")
        .addCompilerFlag("-frename-registers")
        .addCompilerFlag("-no-canonical-prefixes")
  
        // Linker flags
        .addLinkerFlag("-no-canonical-prefixes")
  
        // Additional release flags
        .addCompilationModeFlags(CompilationModeFlags.newBuilder()
            .setMode(CompilationMode.OPT)
            .addCompilerFlag("-O2")
            .addCompilerFlag("-g")
            .addCompilerFlag("-DNDEBUG")
            .addCompilerFlag("-fomit-frame-pointer")
            .addCompilerFlag("-funswitch-loops")
            .addCompilerFlag("-finline-limit=300"))
  
        // Additional debug flags
        .addCompilationModeFlags(CompilationModeFlags.newBuilder()
            .setMode(CompilationMode.DBG)
            .addCompilerFlag("-O0")
            .addCompilerFlag("-g")
            .addCompilerFlag("-fno-omit-frame-pointer"));
  }

  private CToolchain.Builder createMipsClangToolchain(String mipsArch, String gccVersion) {

    String gccToolchain = ndkPaths.createGccToolchainPath(
        String.format("%s-linux-android-%s", mipsArch, gccVersion));
  
    String llvmTriple = mipsArch + "-none-linux-android";
    
    return CToolchain.newBuilder()
        .setCompiler("gcc-" + gccVersion)

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
        .addCompilationModeFlags(CompilationModeFlags.newBuilder()
            .setMode(CompilationMode.OPT)
            .addCompilerFlag("-O2")
            .addCompilerFlag("-g")
            .addCompilerFlag("-DNDEBUG")
            .addCompilerFlag("-fomit-frame-pointer"))
  
        // Additional debug flags
        .addCompilationModeFlags(CompilationModeFlags.newBuilder()
            .setMode(CompilationMode.DBG)
            .addCompilerFlag("-O0")
            .addCompilerFlag("-g")
            .addCompilerFlag("-fno-omit-frame-pointer"));
  }
}

