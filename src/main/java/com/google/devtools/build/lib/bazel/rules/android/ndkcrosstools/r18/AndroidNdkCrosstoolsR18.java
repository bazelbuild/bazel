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

package com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.r18;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.NdkPaths;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.StlImpl;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CrosstoolRelease;
import java.util.ArrayList;
import java.util.List;

/** Generates a CrosstoolRelease proto for the Android NDK. */
final class AndroidNdkCrosstoolsR18 {

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
  static CrosstoolRelease create(
      NdkPaths ndkPaths, StlImpl stlImpl, String hostPlatform, String clangVersion) {
    return CrosstoolRelease.newBuilder()
        .setMajorVersion("android")
        .setMinorVersion("")
        .setDefaultTargetCpu("armeabi")
        .addAllToolchain(createToolchains(ndkPaths, stlImpl, hostPlatform, clangVersion))
        .build();
  }

  private static ImmutableList<CToolchain> createToolchains(
      NdkPaths ndkPaths, StlImpl stlImpl, String hostPlatform, String clangVersion) {

    List<CToolchain.Builder> toolchainBuilders = new ArrayList<>();
    toolchainBuilders.addAll(new ArmCrosstools(ndkPaths, stlImpl, clangVersion).createCrosstools());
    toolchainBuilders.addAll(new X86Crosstools(ndkPaths, stlImpl, clangVersion).createCrosstools());

    ImmutableList.Builder<CToolchain> toolchains = new ImmutableList.Builder<>();

    // Set attributes common to all toolchains.
    for (CToolchain.Builder toolchainBuilder : toolchainBuilders) {
      toolchainBuilder
          .setHostSystemName(hostPlatform)
          .setTargetLibc("local")
          .setAbiVersion(toolchainBuilder.getTargetCpu())
          .setAbiLibcVersion("local");

      // builtin_sysroot is set individually on each toolchain.
      // platforms/arch sysroot
      toolchainBuilder.addCxxBuiltinIncludeDirectory("%sysroot%/usr/include");
      toolchainBuilder.addCxxBuiltinIncludeDirectory(
          ndkPaths.createBuiltinSysroot() + "/usr/include");
      toolchainBuilder.addUnfilteredCxxFlag(
          "-isystem%ndk%/usr/include".replace("%ndk%", ndkPaths.createBuiltinSysroot()));

      toolchains.add(toolchainBuilder.build());
    }

    return toolchains.build();
  }
}
