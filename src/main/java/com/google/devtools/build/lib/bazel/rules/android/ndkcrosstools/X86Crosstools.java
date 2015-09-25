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
 * Crosstool definitions for x86. These values are based on the setup.mk files in the Android NDK
 * toolchain directories.
 */
class X86Crosstools {

  private final NdkPaths ndkPaths;

  X86Crosstools(NdkPaths ndkPaths) {
    this.ndkPaths = ndkPaths;
  }

  ImmutableList<CToolchain.Builder> createCrosstools() {

    ImmutableList.Builder<CToolchain.Builder> builder = ImmutableList.builder();

    /**
     * x86
     */

    builder.add(
        createBaseX86Toolchain()
            .setToolchainIdentifier("x86-4.8")
            .setTargetCpu("x86")
            .setCompiler("gcc-4.8")

            .addAllToolPath(
                ndkPaths.createToolpaths(
                    "x86-4.8",
                    "i686-linux-android",
                    // gcc-4.8 x86 toolchain doesn't have gcov-tool.
                    CppConfiguration.Tool.GCOVTOOL))

            .addAllCxxBuiltinIncludeDirectory(
                ndkPaths.createToolchainIncludePaths("x86-4.8", "i686-linux-android", "4.8"))

            .setBuiltinSysroot(ndkPaths.createBuiltinSysroot("x86"))

            .setSupportsEmbeddedRuntimes(true)
            .setStaticRuntimesFilegroup("static-runtime-libs-" + "x86-4.8")
            .setDynamicRuntimesFilegroup("dynamic-runtime-libs-" + "x86-4.8")

            .addCompilerFlag("-fstack-protector"));

    builder.add(
        createBaseX86Toolchain()
            .setToolchainIdentifier("x86-4.9")
            .setTargetCpu("x86")
            .setCompiler("gcc-4.9")

            .addAllToolPath(ndkPaths.createToolpaths("x86-4.9", "i686-linux-android"))

            .addAllCxxBuiltinIncludeDirectory(
                ndkPaths.createToolchainIncludePaths("x86-4.9", "i686-linux-android", "4.9"))

            .setBuiltinSysroot(ndkPaths.createBuiltinSysroot("x86"))

            .setSupportsEmbeddedRuntimes(true)
            .setStaticRuntimesFilegroup("static-runtime-libs-" + "x86-4.9")
            .setDynamicRuntimesFilegroup("dynamic-runtime-libs-" + "x86-4.9")

            .addCompilerFlag("-fstack-protector-strong"));

    // Add Clang toolchains. x86 uses gcc-4.8.
    for (String clangVersion : new String[] { "3.5", "3.6" }) {      
      builder.add(
          createX86ClangToolchain("x86", "i686", "4.8")
              .setToolchainIdentifier("x86-clang" + clangVersion)
              .setTargetCpu("x86")

              .addAllToolPath(
                  ndkPaths.createClangToolpaths(
                      "x86-4.8",
                      "i686-linux-android",
                      clangVersion,
                      // gcc-4.8 x86 toolchain doesn't have gcov-tool.
                      CppConfiguration.Tool.GCOVTOOL))

              .addAllCxxBuiltinIncludeDirectory(
                  ndkPaths.createToolchainIncludePaths("x86-4.8", "i686-linux-android", "4.8"))

              .setBuiltinSysroot(ndkPaths.createBuiltinSysroot("x86"))

              .setSupportsEmbeddedRuntimes(true)
              .setStaticRuntimesFilegroup("static-runtime-libs-" + "x86-clang" + clangVersion)
              .setDynamicRuntimesFilegroup("dynamic-runtime-libs-" + "x86-clang" + clangVersion));
    }

    /**
     * x86_64
     */

    builder.add(
        createBaseX86Toolchain()
            .setToolchainIdentifier("x86_64-4.9")
            .setTargetCpu("x86_64")
            .setCompiler("gcc-4.9")

            .addAllToolPath(ndkPaths.createToolpaths("x86_64-4.9", "x86_64-linux-android"))

            .addAllCxxBuiltinIncludeDirectory(
                ndkPaths.createToolchainIncludePaths("x86_64-4.9", "x86_64-linux-android", "4.9"))

            .setBuiltinSysroot(ndkPaths.createBuiltinSysroot("x86_64"))

            .setSupportsEmbeddedRuntimes(true)
            .setStaticRuntimesFilegroup("static-runtime-libs-" + "x86_64-4.9")
            .setDynamicRuntimesFilegroup("static-runtime-libs-" + "x86_64-4.9")

            .addCompilerFlag("-fstack-protector-strong"));

    // Add Clang toolchains. x86_64 uses gcc-4.9.
    for (String clangVersion : new String[] { "3.5", "3.6" }) {
      builder.add(
          createX86ClangToolchain("x86_64", "x86_64", "4.9")
              .setToolchainIdentifier("x86_64-clang" + clangVersion)
              .setTargetCpu("x86_64")

              .addAllToolPath(
                  ndkPaths.createClangToolpaths("x86_64-4.9", "x86_64-linux-android", clangVersion))

              .addAllCxxBuiltinIncludeDirectory(
                  ndkPaths.createToolchainIncludePaths("x86_64-4.9", "x86_64-linux-android", "4.9"))

              .setBuiltinSysroot(ndkPaths.createBuiltinSysroot("x86_64"))

              .setSupportsEmbeddedRuntimes(true)
              .setStaticRuntimesFilegroup("static-runtime-libs-" + "x86_64-clang" + clangVersion)
              .setDynamicRuntimesFilegroup(
                  "dynamic-runtime-libs-" + "x86_64-clang" + clangVersion));
    }

    ImmutableList<CToolchain.Builder> toolchainBuilders = builder.build();
    
    for (CToolchain.Builder toolchainBuilder : toolchainBuilders) {
      toolchainBuilder.setTargetSystemName("x86-linux-android");
    }

    return toolchainBuilders;
  }
  
  private CToolchain.Builder createBaseX86Toolchain() {
    return CToolchain.newBuilder()
        // Compiler flags
        .addCompilerFlag("-ffunction-sections")
        .addCompilerFlag("-funwind-tables")
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
            .addCompilerFlag("-fstrict-aliasing")
            .addCompilerFlag("-funswitch-loops")
            .addCompilerFlag("finline-limit=300"))
  
        // Additional debug flags
        .addCompilationModeFlags(CompilationModeFlags.newBuilder()
            .setMode(CompilationMode.DBG)
            .addCompilerFlag("-O0")
            .addCompilerFlag("-g")
            .addCompilerFlag("-fno-omit-frame-pointer")
            .addCompilerFlag("-fno-strict-aliasing"));
  }

  private CToolchain.Builder createX86ClangToolchain(
      String x86Arch, String llvmArch, String gccVersion) {

    String gccToolchain = ndkPaths.createGccToolchainPath(
        String.format("%s-linux-android-%s", x86Arch, gccVersion));

    String llvmTriple = llvmArch + "-none-linux-android";

    return CToolchain.newBuilder()
        .setCompiler("gcc-" + gccVersion)

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
        .addCompilationModeFlags(CompilationModeFlags.newBuilder()
            .setMode(CompilationMode.OPT)
            .addCompilerFlag("-O2")
            .addCompilerFlag("-g")
            .addCompilerFlag("-DNDEBUG")
            .addCompilerFlag("-fomit-frame-pointer")
            .addCompilerFlag("-fstrict-aliasing"))
  
        // Additional debug flags
        .addCompilationModeFlags(CompilationModeFlags.newBuilder()
            .setMode(CompilationMode.DBG)
            .addCompilerFlag("-O0")
            .addCompilerFlag("-g")
            .addCompilerFlag("-fno-omit-frame-pointer")
            .addCompilerFlag("-fnostrict-aliasing"));
  }
}
