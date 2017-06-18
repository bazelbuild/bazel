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
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CompilationMode;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CompilationModeFlags;
import java.util.List;

/**
 * Crosstool definitions for ARM. These values are based on the setup.mk files in the Android NDK
 * toolchain directories.
 */
class ArmCrosstools {

  private final NdkPaths ndkPaths;
  private final StlImpl stlImpl;

  ArmCrosstools(NdkPaths ndkPaths, StlImpl stlImpl) {
    this.ndkPaths = ndkPaths;
    this.stlImpl = stlImpl;
  }

  ImmutableList<CToolchain.Builder> createCrosstools() {

    ImmutableList.Builder<CToolchain.Builder> toolchains = ImmutableList.builder();

    toolchains.add(createAarch64Toolchain());
    toolchains.add(createAarch64ClangToolchain());

    // The Android NDK Make files create several sets of flags base on
    // arm vs armeabi-v7a vs armeabi-v7a-hard, and arm vs thumb mode,
    // resulting in:
    //    arm-linux-androideabi-4.9
    //    arm-linux-androideabi-4.9-v7a
    //    arm-linux-androideabi-4.9-v7a-hard
    //    arm-linux-androideabi-4.9-thumb
    //    arm-linux-androideabi-4.9-v7a-thumb
    //    arm-linux-androideabi-4.9-v7a-hard-thumb
    //
    // and similar for the Clang toolchains.

    toolchains.addAll(createArmeabiToolchains(false));
    toolchains.addAll(createArmeabiToolchains(true));

    toolchains.addAll(createArmeabiClangToolchain(false));
    toolchains.addAll(createArmeabiClangToolchain(true));

    return toolchains.build();
  }

  private CToolchain.Builder createAarch64Toolchain() {

    String toolchainName = "aarch64-linux-android-4.9";
    String targetPlatform = "aarch64-linux-android";

    CToolchain.Builder toolchain = CToolchain.newBuilder()
        .setToolchainIdentifier("aarch64-linux-android-4.9")
        .setTargetSystemName("aarch64-linux-android")
        .setTargetCpu("arm64-v8a")
        .setCompiler("gcc-4.9")

        .addAllToolPath(ndkPaths.createToolpaths(toolchainName, targetPlatform))
        .addAllCxxBuiltinIncludeDirectory(
            ndkPaths.createGccToolchainBuiltinIncludeDirectories(
                toolchainName, targetPlatform, "4.9"))
        .setBuiltinSysroot(ndkPaths.createBuiltinSysroot("arm64"))

        // Compiler flags
        .addCompilerFlag("-fpic")
        .addCompilerFlag("-ffunction-sections")
        .addCompilerFlag("-funwind-tables")
        .addCompilerFlag("-fstack-protector-strong")
        .addCompilerFlag("-no-canonical-prefixes")
        .addCompilerFlag("-fno-canonical-system-headers")

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
            .addCompilerFlag("-finline-limit=300"))

        // Additional debug flags
        .addCompilationModeFlags(CompilationModeFlags.newBuilder()
            .setMode(CompilationMode.DBG)
            .addCompilerFlag("-O0")
            .addCompilerFlag("-UNDEBUG")
            .addCompilerFlag("-fno-omit-frame-pointer")
            .addCompilerFlag("-fno-strict-aliasing"));

    stlImpl.addStlImpl(toolchain, "4.9");
    return toolchain;
  }

  private CToolchain.Builder createAarch64ClangToolchain() {

    String toolchainName = "aarch64-linux-android-4.9";
    String targetPlatform = "aarch64-linux-android";
    String gccToolchain = ndkPaths.createGccToolchainPath(toolchainName);
    String llvmTriple = "aarch64-none-linux-android";

    CToolchain.Builder toolchain =  CToolchain.newBuilder()
        .setToolchainIdentifier("aarch64-linux-android-clang3.8")
        .setTargetSystemName("aarch64-linux-android")
        .setTargetCpu("arm64-v8a")
        .setCompiler("clang3.8")

        .addAllToolPath(ndkPaths.createClangToolpaths(toolchainName, targetPlatform, null))

        .setBuiltinSysroot(ndkPaths.createBuiltinSysroot("arm64"))

        // Compiler flags
        .addCompilerFlag("-gcc-toolchain")
        .addCompilerFlag(gccToolchain)
        .addCompilerFlag("-target")
        .addCompilerFlag(llvmTriple)
        .addCompilerFlag("-ffunction-sections")
        .addCompilerFlag("-funwind-tables")
        .addCompilerFlag("-fstack-protector-strong")
        .addCompilerFlag("-fpic")
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
            .addCompilerFlag("-UNDEBUG")
            .addCompilerFlag("-fno-omit-frame-pointer")
            .addCompilerFlag("-fno-strict-aliasing"));

    stlImpl.addStlImpl(toolchain, "4.9");
    return toolchain;
  }

  private List<CToolchain.Builder> createArmeabiToolchains(boolean thumb,
      CppConfiguration.Tool... excludedTools) {

    ImmutableList<CToolchain.Builder> toolchains = ImmutableList.of(
        createBaseArmeabiToolchain(thumb, excludedTools)
              .setToolchainIdentifier(
                  createArmeabiName("arm-linux-androideabi-4.9", thumb))
              .setTargetCpu(createArmeabiCpuName("armeabi", thumb))

              .addCompilerFlag("-march=armv5te")
              .addCompilerFlag("-mtune=xscale")
              .addCompilerFlag("-msoft-float"),

        createBaseArmeabiToolchain(thumb, excludedTools)
            .setToolchainIdentifier(
                createArmeabiName("arm-linux-androideabi-4.9-v7a", thumb))
            .setTargetCpu(createArmeabiCpuName("armeabi-v7a", thumb))

            .addCompilerFlag("-march=armv7-a")
            .addCompilerFlag("-mfpu=vfpv3-d16")
            .addCompilerFlag("-mfloat-abi=softfp")

            .addLinkerFlag("-march=armv7-a")
            .addLinkerFlag("-Wl,--fix-cortex-a8"),

        createBaseArmeabiToolchain(thumb, excludedTools)
            .setToolchainIdentifier(
                createArmeabiName("arm-linux-androideabi-4.9-v7a-hard", thumb))
            .setTargetCpu(createArmeabiCpuName("armeabi-v7a-hard", thumb))

            .addCompilerFlag("-march=armv7-a")
            .addCompilerFlag("-mfpu=vfpv3-d16")
            .addCompilerFlag("-mhard-float")
            .addCompilerFlag("-D_NDK_MATH_NO_SOFTFP=1")

            .addLinkerFlag("-march=armv7-a")
            .addLinkerFlag("-Wl,--fix-cortex-a8")
            .addLinkerFlag("-Wl,--no-warn-mismatch")
            .addLinkerFlag("-lm_hard"));

    stlImpl.addStlImpl(toolchains, "4.9");
    return toolchains;
  }

  /**
   * Flags common to arm-linux-androideabi*
   */
  private CToolchain.Builder createBaseArmeabiToolchain(
      boolean thumb, CppConfiguration.Tool... excludedTools) {

    String toolchainName = "arm-linux-androideabi-4.9";
    String targetPlatform = "arm-linux-androideabi";

    CToolchain.Builder toolchain = CToolchain.newBuilder()
        .setTargetSystemName(targetPlatform)
        .setCompiler("gcc-4.9")

        .addAllToolPath(ndkPaths.createToolpaths(toolchainName, targetPlatform, excludedTools))
        .addAllCxxBuiltinIncludeDirectory(
            ndkPaths.createGccToolchainBuiltinIncludeDirectories(
                toolchainName, targetPlatform, "4.9"))
        .setBuiltinSysroot(ndkPaths.createBuiltinSysroot("arm"))

        // Compiler flags
        .addCompilerFlag("-fstack-protector-strong")
        .addCompilerFlag("-fpic")
        .addCompilerFlag("-ffunction-sections")
        .addCompilerFlag("-funwind-tables")
        .addCompilerFlag("-no-canonical-prefixes")
        .addCompilerFlag("-fno-canonical-system-headers")

        // Linker flags
        .addLinkerFlag("-no-canonical-prefixes");

    if (thumb) {
      toolchain.addCompilationModeFlags(CompilationModeFlags.newBuilder()
          .setMode(CompilationMode.OPT)
          .addCompilerFlag("-mthumb")
          .addCompilerFlag("-Os")
          .addCompilerFlag("-g")
          .addCompilerFlag("-DNDEBUG")
          .addCompilerFlag("-fomit-frame-pointer")
          .addCompilerFlag("-fno-strict-aliasing")
          .addCompilerFlag("-finline-limit=64"));

      toolchain.addCompilationModeFlags(CompilationModeFlags.newBuilder()
          .setMode(CompilationMode.DBG)
          .addCompilerFlag("-g")
          .addCompilerFlag("-fno-strict-aliasing")
          .addCompilerFlag("-finline-limit=64")
          .addCompilerFlag("-O0")
          .addCompilerFlag("-UNDEBUG")
          .addCompilerFlag("-marm")
          .addCompilerFlag("-fno-omit-frame-pointer"));
    } else {
      toolchain.addCompilationModeFlags(CompilationModeFlags.newBuilder()
          .setMode(CompilationMode.OPT)
          .addCompilerFlag("-O2")
          .addCompilerFlag("-g")
          .addCompilerFlag("-DNDEBUG")
          .addCompilerFlag("-fomit-frame-pointer")
          .addCompilerFlag("-fstrict-aliasing")
          .addCompilerFlag("-funswitch-loops")
          .addCompilerFlag("-finline-limit=300"));

      toolchain.addCompilationModeFlags(CompilationModeFlags.newBuilder()
          .setMode(CompilationMode.DBG)
          .addCompilerFlag("-g")
          .addCompilerFlag("-funswitch-loops")
          .addCompilerFlag("-finline-limit=300")
          .addCompilerFlag("-O0")
          .addCompilerFlag("-UNDEBUG")
          .addCompilerFlag("-fno-omit-frame-pointer")
          .addCompilerFlag("-fno-strict-aliasing"));
    }

    return toolchain;
  }

  private List<CToolchain.Builder> createArmeabiClangToolchain(boolean thumb) {

    ImmutableList<CToolchain.Builder> toolchains = ImmutableList.of(
        createBaseArmeabiClangToolchain(thumb)
            .setToolchainIdentifier(createArmeabiName("arm-linux-androideabi-clang3.8", thumb))
            .setTargetCpu(createArmeabiCpuName("armeabi", thumb))

            .addCompilerFlag("-target")
            .addCompilerFlag("armv5te-none-linux-androideabi") // LLVM_TRIPLE
            .addCompilerFlag("-march=armv5te")
            .addCompilerFlag("-mtune=xscale")
            .addCompilerFlag("-msoft-float")

            .addLinkerFlag("-target")
            // LLVM_TRIPLE
            .addLinkerFlag("armv5te-none-linux-androideabi"),

      createBaseArmeabiClangToolchain(thumb)
          .setToolchainIdentifier(createArmeabiName("arm-linux-androideabi-clang3.8-v7a", thumb))
          .setTargetCpu(createArmeabiCpuName("armeabi-v7a", thumb))

          .addCompilerFlag("-target")
          .addCompilerFlag("armv7-none-linux-androideabi") // LLVM_TRIPLE
          .addCompilerFlag("-march=armv7-a")
          .addCompilerFlag("-mfloat-abi=softfp")
          .addCompilerFlag("-mfpu=vfpv3-d16")

          .addLinkerFlag("-target")
          .addLinkerFlag("armv7-none-linux-androideabi") // LLVM_TRIPLE
          .addLinkerFlag("-Wl,--fix-cortex-a8"),

      createBaseArmeabiClangToolchain(thumb)
          .setToolchainIdentifier(
              createArmeabiName("arm-linux-androideabi-clang3.8-v7a-hard", thumb))
          .setTargetCpu(createArmeabiCpuName("armeabi-v7a-hard", thumb))

          .addCompilerFlag("-target")
          .addCompilerFlag("armv7-none-linux-androideabi") // LLVM_TRIPLE
          .addCompilerFlag("-march=armv7-a")
          .addCompilerFlag("-mfpu=vfpv3-d16")
          .addCompilerFlag("-mhard-float")
          .addCompilerFlag("-D_NDK_MATH_NO_SOFTFP=1")

          .addLinkerFlag("-target")
          .addLinkerFlag("armv7-none-linux-androideabi") // LLVM_TRIPLE
          .addLinkerFlag("-Wl,--fix-cortex-a8")
          .addLinkerFlag("-Wl,--no-warn-mismatch")
          .addLinkerFlag("-lm_hard"));

    stlImpl.addStlImpl(toolchains, "4.9");
    return toolchains;
  }

  private CToolchain.Builder createBaseArmeabiClangToolchain(boolean thumb) {

    String toolchainName = "arm-linux-androideabi-4.9";
    String targetPlatform = "arm-linux-androideabi";
    String gccToolchain = ndkPaths.createGccToolchainPath("arm-linux-androideabi-4.9");

    CToolchain.Builder toolchain = CToolchain.newBuilder()
        .setTargetSystemName("arm-linux-androideabi")
        .setCompiler("clang3.8")

        .addAllToolPath(ndkPaths.createClangToolpaths(toolchainName, targetPlatform, null))
        .addCxxBuiltinIncludeDirectory(
            ndkPaths.createClangToolchainBuiltinIncludeDirectory(
                AndroidNdkCrosstoolsR11.CLANG_VERSION))
        .setBuiltinSysroot(ndkPaths.createBuiltinSysroot("arm"))

        // Compiler flags
        .addCompilerFlag("-gcc-toolchain")
        .addCompilerFlag(gccToolchain)
        .addCompilerFlag("-fpic")
        .addCompilerFlag("-ffunction-sections")
        .addCompilerFlag("-funwind-tables")
        .addCompilerFlag("-fstack-protector-strong")
        .addCompilerFlag("-Wno-invalid-command-line-argument")
        .addCompilerFlag("-Wno-unused-command-line-argument")
        .addCompilerFlag("-no-canonical-prefixes")
        .addCompilerFlag("-fno-integrated-as")

        // Linker flags
        .addLinkerFlag("-gcc-toolchain")
        .addLinkerFlag(gccToolchain)
        .addLinkerFlag("-no-canonical-prefixes");

    if (thumb) {
      toolchain.addCompilationModeFlags(CompilationModeFlags.newBuilder()
          .setMode(CompilationMode.OPT)
          .addCompilerFlag("-mthumb")
          .addCompilerFlag("-Os")
          .addCompilerFlag("-g")
          .addCompilerFlag("-DNDEBUG")
          .addCompilerFlag("-fomit-frame-pointer")
          .addCompilerFlag("-fno-strict-aliasing"));

      toolchain.addCompilationModeFlags(CompilationModeFlags.newBuilder()
          .setMode(CompilationMode.DBG)
          .addCompilerFlag("-g")
          .addCompilerFlag("-fno-strict-aliasing")
          .addCompilerFlag("-O0")
          .addCompilerFlag("-UNDEBUG")
          .addCompilerFlag("-marm")
          .addCompilerFlag("-fno-omit-frame-pointer"));
    } else {
      toolchain.addCompilationModeFlags(CompilationModeFlags.newBuilder()
          .setMode(CompilationMode.OPT)
          .addCompilerFlag("-O2")
          .addCompilerFlag("-g")
          .addCompilerFlag("-DNDEBUG")
          .addCompilerFlag("-fomit-frame-pointer")
          .addCompilerFlag("-fstrict-aliasing"));

      toolchain.addCompilationModeFlags(CompilationModeFlags.newBuilder()
          .setMode(CompilationMode.DBG)
          .addCompilerFlag("-g")
          .addCompilerFlag("-O0")
          .addCompilerFlag("-UNDEBUG")
          .addCompilerFlag("-fno-omit-frame-pointer")
          .addCompilerFlag("-fno-strict-aliasing"));
    }

    return toolchain;
  }

  private static String createArmeabiName(String base, boolean thumb) {
    return base + (thumb ? "-thumb" : "");
  }

  private static String createArmeabiCpuName(String base, boolean thumb) {
    return base + (thumb ? "-thumb" : "");
  }
}
