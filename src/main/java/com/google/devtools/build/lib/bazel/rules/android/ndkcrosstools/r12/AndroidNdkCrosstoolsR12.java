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

package com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.r12;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.NdkPaths;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.StlImpl;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CrosstoolRelease;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.DefaultCpuToolchain;
import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;

/** Generates a CrosstoolRelease proto for the Android NDK. */
final class AndroidNdkCrosstoolsR12 {
  private AndroidNdkCrosstoolsR12() {}
  /**
   * Creates a CrosstoolRelease proto for the Android NDK, given the API level to use and the
   * release revision. The crosstools are generated through code rather than checked in as a flat
   * file to reduce the amount of templating needed (for parameters like the release name and
   * certain paths), to reduce duplication, and to make it easier to support future versions of the
   * NDK. TODO(bazel-team): Eventually we should move this into Skylark so the crosstools can be
   * updated independently of Bazel itself.
   *
   * @return A CrosstoolRelease for the Android NDK.
   */
  static CrosstoolRelease create(NdkPaths ndkPaths, StlImpl stlImpl, String hostPlatform) {
    return CrosstoolRelease.newBuilder()
        .setMajorVersion("android")
        .setMinorVersion("")
        .setDefaultTargetCpu("armeabi")
        .addAllDefaultToolchain(getDefaultCpuToolchains(stlImpl))
        .addAllToolchain(createToolchains(ndkPaths, stlImpl, hostPlatform))
        .build();
  }

  private static ImmutableList<CToolchain> createToolchains(
      NdkPaths ndkPaths, StlImpl stlImpl, String hostPlatform) {

    List<CToolchain.Builder> toolchainBuilders = new ArrayList<>();
    toolchainBuilders.addAll(new ArmCrosstools(ndkPaths, stlImpl).createCrosstools());
    toolchainBuilders.addAll(new MipsCrosstools(ndkPaths, stlImpl).createCrosstools());
    toolchainBuilders.addAll(new X86Crosstools(ndkPaths, stlImpl).createCrosstools());

    ImmutableList.Builder<CToolchain> toolchains = new ImmutableList.Builder<>();

    // Set attributes common to all toolchains.
    for (CToolchain.Builder toolchainBuilder : toolchainBuilders) {
      toolchainBuilder
          .setHostSystemName(hostPlatform)
          .setTargetLibc("local")
          .setAbiVersion(toolchainBuilder.getTargetCpu())
          .setAbiLibcVersion("local");

      // builtin_sysroot is set individually on each toolchain.
      toolchainBuilder.addCxxBuiltinIncludeDirectory("%sysroot%/usr/include");

      toolchains.add(toolchainBuilder.build());
    }

    return toolchains.build();
  }

  private static ImmutableList<DefaultCpuToolchain> getDefaultCpuToolchains(StlImpl stlImpl) {
    // TODO(bazel-team): It would be better to auto-generate this somehow.

    ImmutableMap<String, String> defaultCpus =
        ImmutableMap.<String, String>builder()
            // arm
            .put("armeabi", "arm-linux-androideabi-4.9")
            .put("armeabi-v7a", "arm-linux-androideabi-4.9-v7a")
            .put("arm64-v8a", "aarch64-linux-android-4.9")

            // mips
            .put("mips", "mipsel-linux-android-4.9")
            .put("mips64", "mips64el-linux-android-4.9")

            // x86
            .put("x86", "x86-4.9")
            .put("x86_64", "x86_64-4.9")
            .build();

    ImmutableList.Builder<DefaultCpuToolchain> defaultCpuToolchains = ImmutableList.builder();
    for (Entry<String, String> defaultCpu : defaultCpus.entrySet()) {
      defaultCpuToolchains.add(
          DefaultCpuToolchain.newBuilder()
              .setCpu(defaultCpu.getKey())
              .setToolchainIdentifier(defaultCpu.getValue() + "-" + stlImpl.getName())
              .build());
    }
    return defaultCpuToolchains.build();
  }
}
