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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CrosstoolRelease;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.DefaultCpuToolchain;

import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;

/**
 * Generates a CrosstoolRelease proto for the Android NDK.
 */
public class AndroidNdkCrosstools {

  // TODO(bazel-team): Support future versions of the NDK.
  private static final String KNOWN_NDK_REVISION = "r10e";
  
  /**
   * Exception thrown when there is an error creating the crosstools file.
   */
  public static class NdkCrosstoolsException extends Exception {
    private NdkCrosstoolsException(String msg) {
      super(msg);
    }
  }

  /**
   * Creates a CrosstoolRelease proto for the Android NDK, given the API level to use and the
   * release revision. The crosstools are generated through code rather than checked in as a flat
   * file to reduce the amount of templating needed (for parameters like the release name and
   * certain paths), to reduce duplication, and to make it easier to support future versions of the
   * NDK. TODO(bazel-team): Eventually we should move this into Skylark so the crosstools can be
   * updated independently of Bazel itself.
   *
   * @param eventHandler The event handler for sending warning messages.
   * @param repositoryName The name of the repository, which should correspond to the name of the
   *        android_ndk_repository rule. 
   * @param apiLevel The API level used for the NDK.
   * @param ndkRelease The NDK release
   * @return A CrosstoolRelease for the Android NDK.
   * @throws NdkCrosstoolsException If the crosstool could not be created.
   */
  public static CrosstoolRelease createCrosstoolRelease(
      EventHandler eventHandler, String repositoryName, String apiLevel, NdkRelease ndkRelease)
          throws NdkCrosstoolsException {

    // Check that the Android NDK revision is both valid and one we know about. 
    if (!ndkRelease.isValid) {

      // Try using the NDK revision we know about.
      ndkRelease = NdkRelease.guessBitness(KNOWN_NDK_REVISION);

      eventHandler.handle(Event.warn(String.format(
          "The revision of the Andorid NDK given in android_ndk_repository rule '%s' could not be "
          + "determined (the revision string found is '%s'). "
          + "Defaulting to Android NDK revision %s", repositoryName, ndkRelease.rawRelease,
          ndkRelease)));

    } else if (!KNOWN_NDK_REVISION.equals(ndkRelease.release)) {
      eventHandler.handle(Event.warn(String.format(
          "Bazel Android NDK crosstools are based on Android NDK revision %s. "
          + "The revision of the Android NDK given in android_ndk_repository rule '%s' is '%s'",
          KNOWN_NDK_REVISION, repositoryName, ndkRelease.release)));
    }

    return CrosstoolRelease.newBuilder()
        .setMajorVersion("android")
        .setMinorVersion("")
        .setDefaultTargetCpu("armeabi")
        .addAllDefaultToolchain(getDefaultCpuToolchains())
        .addAllToolchain(createToolchains(eventHandler, repositoryName, apiLevel, ndkRelease))
        .build();
  }

  private static ImmutableList<CToolchain> createToolchains(
      EventHandler eventHandler, String repositoryName, String apiLevel, NdkRelease ndkRelease)
          throws NdkCrosstoolsException {

    String hostPlatform = getHostPlatform(ndkRelease);
    NdkPaths ndkPaths = new NdkPaths(eventHandler, repositoryName, hostPlatform, apiLevel);

    List<CToolchain.Builder> toolchainBuilders = new ArrayList<>();
    toolchainBuilders.addAll(new ArmCrosstools(ndkPaths).createCrosstools());
    toolchainBuilders.addAll(new MipsCrosstools(ndkPaths).createCrosstools());
    toolchainBuilders.addAll(new X86Crosstools(ndkPaths).createCrosstools());

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

      // Add a toolchain for each available STL implementation.
      for (CToolchain.Builder builder : createStlImpls(toolchainBuilder, ndkPaths)) {
        toolchains.add(builder.build());
      }
    }

    return toolchains.build();
  }

  /**
   * Creates a list of CToolchain builders for each STL implementation available in the NDK based
   * on the given CToolchain builder. 
   */
  private static ImmutableList<CToolchain.Builder> createStlImpls(
      CToolchain.Builder builder, NdkPaths ndkPaths) {
    ImmutableList.Builder<CToolchain.Builder> toolchains = ImmutableList.builder();
    
    CToolchain baseToolchain = builder.build();

    // Extract the gcc version from the compiler, which should look like "gcc-4.8" or "gcc-4.9".
    String gccVersion = baseToolchain.getCompiler().split("-")[1];

    toolchains.add(
        CToolchain.newBuilder(baseToolchain)
            .setToolchainIdentifier(baseToolchain.getToolchainIdentifier() + "-gnu-libstdcpp")
            .addAllUnfilteredCxxFlag(
                createIncludeFlags(
                    ndkPaths.createGnuLibstdcIncludePaths(
                        gccVersion, baseToolchain.getTargetCpu())))
            .setStaticRuntimesFilegroup(
                baseToolchain.getStaticRuntimesFilegroup() + "-gnu-libstdcpp")
            .setDynamicRuntimesFilegroup(
                baseToolchain.getDynamicRuntimesFilegroup() + "-gnu-libstdcpp"));

    toolchains.add(
        CToolchain.newBuilder(baseToolchain)
            .setToolchainIdentifier(baseToolchain.getToolchainIdentifier() + "-libcpp")
            .addAllUnfilteredCxxFlag(createIncludeFlags(ndkPaths.createLibcxxIncludePaths()))
            .setStaticRuntimesFilegroup(baseToolchain.getStaticRuntimesFilegroup() + "-libcpp")
            .setDynamicRuntimesFilegroup(baseToolchain.getDynamicRuntimesFilegroup() + "-libcpp"));

    toolchains.add(
        CToolchain.newBuilder(baseToolchain)
            .setToolchainIdentifier(baseToolchain.getToolchainIdentifier() + "-stlport")
            .addAllUnfilteredCxxFlag(createIncludeFlags(ndkPaths.createStlportIncludePaths()))
            .setStaticRuntimesFilegroup(baseToolchain.getStaticRuntimesFilegroup() + "-stlport")
            .setDynamicRuntimesFilegroup(baseToolchain.getDynamicRuntimesFilegroup() + "-stlport"));

    return toolchains.build();
  }

  private static Iterable<String> createIncludeFlags(Iterable<String> includePaths) {
    ImmutableList.Builder<String> includeFlags = ImmutableList.builder();
    for (String includePath : includePaths) {
      includeFlags.add("-isystem");
      includeFlags.add(includePath);
    }
    return includeFlags.build();
  }

  private static ImmutableList<DefaultCpuToolchain> getDefaultCpuToolchains() {
    // TODO(bazel-team): It would be better to generate this somehow.
    ImmutableMap<String, String> defaultCpus = ImmutableMap.<String, String>builder()
        .put("armeabi", "arm-linux-androideabi-4.9-gnu-libstdcpp")
        .put("armeabi-v7a", "arm-linux-androideabi-4.9-v7a-gnu-libstdcpp")
        .put("armeabi-v7a-hard", "arm-linux-androideabi-4.9-v7a-hard-gnu-libstdcpp")
        .put("armeabi-thumb", "arm-linux-androideabi-4.9-thumb-gnu-libstdcpp")
        .put("armeabi-v7a-thumb", "arm-linux-androideabi-4.9-v7a-thumb-gnu-libstdcpp")
        .put("armeabi-v7a-hard-thumb", "arm-linux-androideabi-4.9-v7a-hard-thumb-gnu-libstdcpp")
        .put("arm64-v8a", "aarch64-linux-android-4.9-gnu-libstdcpp")
        .put("mips", "mipsel-linux-android-4.9-gnu-libstdcpp")
        .put("mips64", "mips64el-linux-android-4.9-gnu-libstdcpp")
        .put("x86", "x86-4.9-gnu-libstdcpp")
        .put("x86_64", "x86_64-4.9-gnu-libstdcpp")
        .build();

    ImmutableList.Builder<DefaultCpuToolchain> defaultCpuToolchains = ImmutableList.builder();
    for (Entry<String, String> entry : defaultCpus.entrySet()) {
      defaultCpuToolchains.add(DefaultCpuToolchain.newBuilder()
          .setCpu(entry.getKey())
          .setToolchainIdentifier(entry.getValue())
          .build());
    }
    return defaultCpuToolchains.build();
  }

  private static String getHostPlatform(NdkRelease ndkRelease) throws NdkCrosstoolsException {
    String hostOs;
    switch (OS.getCurrent()) {
      case DARWIN:
        hostOs = "darwin";
        break;
      case LINUX:
        hostOs = "linux";
        break;
      case WINDOWS:
        hostOs = "windows";
        if (!ndkRelease.is64Bit) {
          // 32-bit windows paths don't have the "-x86" suffix in the NDK (added below), but
          // 64-bit windows does have the "-x86_64" suffix.
          return hostOs;
        }
        break;
      case UNKNOWN:
      default:
        throw new NdkCrosstoolsException(
            String.format("NDK does not support the host platform \"%s\"", OS.getCurrent()));
    }

    // Use the arch from the NDK rather than detecting the actual platform, since it's possible
    // to use the 32-bit NDK on a 64-bit machine.
    String hostArch = ndkRelease.is64Bit ? "x86_64" : "x86";

    return hostOs + "-" + hostArch;
  }
}
