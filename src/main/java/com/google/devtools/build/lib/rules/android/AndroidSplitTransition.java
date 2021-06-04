// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.transitions.SplitTransition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.cpp.CppOptions;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidSplitTransititionApi;
import java.util.List;
import net.starlark.java.eval.Printer;

/** Android Split configuration transition for properly handling native dependencies */
final class AndroidSplitTransition implements SplitTransition, AndroidSplitTransititionApi {

  @Override
  public ImmutableSet<Class<? extends FragmentOptions>> requiresOptionFragments() {
    return ImmutableSet.of(
        AndroidConfiguration.Options.class,
        CoreOptions.class,
        CppOptions.class,
        PlatformOptions.class);
  }

  @Override
  public ImmutableMap<String, BuildOptions> split(
      BuildOptionsView buildOptions, EventHandler eventHandler) {

    AndroidConfiguration.Options androidOptions =
        buildOptions.get(AndroidConfiguration.Options.class);

    CppOptions cppOptions = buildOptions.get(CppOptions.class);
    /*
     * The intended order of checks is:
     *  - When --incompatible_enable_android_toolchain_resolution is set:
     *    - --android_platforms
     *      - Split using the values of this flag as the target platform
     *      - If this is unset, use the first value from --platforms.
     *        - If this isn't a valid Android platform, an error will be thrown during the build.
     *  - Fall back to legacy flag logic:
     *    - --fat_apk_cpus
     *      - Split using the values of this flag as --cpu
     *      - If this is unset, fall though.
     *    - --android_cpu
     *      - Don't split, just use the value of this flag as --cpu
     *      - This will not update the output path to include "-android".
     *      - If this is unset, fall though.
     *    - Default
     *      - This will not update the output path to include "-android".
     *      - Don't split, using the same previously set --cpu value.
     */
    if (androidOptions.incompatibleUseToolchainResolution) {
      // Always use --android_platforms when toolchain resolution is enabled.
      List<Label> platformsToSplit = androidOptions.androidPlatforms;
      if (platformsToSplit.isEmpty()) {
        // If --android_platforms is unset, instead use only the first value from --platforms.
        Label targetPlatform =
            Iterables.getFirst(buildOptions.get(PlatformOptions.class).platforms, null);
        platformsToSplit = ImmutableList.of(targetPlatform);
      }
      return handleAndroidPlatforms(buildOptions, androidOptions, platformsToSplit);
    }

    // Fall back to the legacy flags.
    if (!androidOptions.fatApkCpus.isEmpty()) {
      return handleFatApkCpus(buildOptions, androidOptions);
    } else if (!androidOptions.cpu.isEmpty()
        && androidOptions.androidCrosstoolTop != null
        && !androidOptions.androidCrosstoolTop.equals(cppOptions.crosstoolTop)) {
      return handleAndroidCpu(buildOptions, androidOptions);
    } else {
      return handleDefaultSplit(buildOptions, buildOptions.get(CoreOptions.class).cpu);
    }
  }

  private void addNonCpuSplits(
      ImmutableMap.Builder<String, BuildOptions> result,
      String name,
      BuildOptionsView buildOptions) {

    AndroidConfiguration.Options androidOptions =
        buildOptions.get(AndroidConfiguration.Options.class);
    if (!androidOptions.fatApkHwasan) {
      return;
    }

    if (name.contains("arm64-v8a")) {
      BuildOptionsView hwasanSplitOptions = buildOptions.clone();

      // A HWASAN build is different from a regular one in these ways:
      // - The native library install directory gets a "-hwasan" suffix
      // - Some compiler/linker command line options are different (defined in the Android C++
      //   toolchain)
      // - The name of the output directory is changed so that HWASAN and non-HWASAN artifacts
      //   do not conflict
      hwasanSplitOptions.get(CppOptions.class).outputDirectoryTag = "hwasan";
      hwasanSplitOptions.get(AndroidConfiguration.Options.class).hwasan = true;

      result.put(name + "-hwasan", hwasanSplitOptions.underlying());
    }
  }

  /**
   * Splits the configuration based on the values of --android_platforms. Each split will set the
   * --platforms flag to one value from --android_platforms, as well as clean up a few other flags
   * around native CC builds.
   */
  private ImmutableMap<String, BuildOptions> handleAndroidPlatforms(
      BuildOptionsView buildOptions,
      AndroidConfiguration.Options androidOptions,
      List<Label> androidPlatforms) {
    ImmutableMap.Builder<String, BuildOptions> result = ImmutableMap.builder();
    for (Label platform : ImmutableSortedSet.copyOf(androidPlatforms)) {
      BuildOptionsView splitOptions = buildOptions.clone();

      // Disable fat APKs for the child configurations.
      splitOptions.get(AndroidConfiguration.Options.class).fatApkCpus = ImmutableList.of();
      splitOptions.get(AndroidConfiguration.Options.class).androidPlatforms = ImmutableList.of();

      // The cpu flag will be set by platform mapping if a mapping exists.
      splitOptions.get(PlatformOptions.class).platforms = ImmutableList.of(platform);
      setCcFlagsFromAndroid(androidOptions, splitOptions);
      result.put(platform.getName(), splitOptions.underlying());

      addNonCpuSplits(result, platform.getName(), splitOptions);
    }
    return result.build();
  }

  /** Returns a single-split transition that uses the "--cpu" and does not change any flags. */
  private ImmutableMap<String, BuildOptions> handleDefaultSplit(
      BuildOptionsView buildOptions, String cpu) {
    // Avoid a clone when nothing changes.
    ImmutableMap.Builder<String, BuildOptions> result = ImmutableMap.builder();
    result.put(cpu, buildOptions.underlying());
    addNonCpuSplits(result, cpu, buildOptions);
    return result.build();
  }

  /**
   * Returns a transition that sets "--cpu" to the value of "--android_cpu" and sets other C++ flags
   * based on the corresponding Android flags.
   */
  private ImmutableMap<String, BuildOptions> handleAndroidCpu(
      BuildOptionsView buildOptions, AndroidConfiguration.Options androidOptions) {
    BuildOptionsView splitOptions = buildOptions.clone();
    splitOptions.get(CoreOptions.class).cpu = androidOptions.cpu;
    setCcFlagsFromAndroid(androidOptions, splitOptions);
    // Ensure platforms aren't set so that platform mapping can take place.
    splitOptions.get(PlatformOptions.class).platforms = ImmutableList.of();
    return handleDefaultSplit(splitOptions, androidOptions.cpu);
  }

  /**
   * Returns a multi-split transition that sets "--cpu" with the values of "--fat_apk_cpu" and sets
   * other C++ flags based on the corresponding Android flags.
   */
  private ImmutableMap<String, BuildOptions> handleFatApkCpus(
      BuildOptionsView buildOptions, AndroidConfiguration.Options androidOptions) {
    ImmutableMap.Builder<String, BuildOptions> result = ImmutableMap.builder();
    for (String cpu : ImmutableSortedSet.copyOf(androidOptions.fatApkCpus)) {
      BuildOptionsView splitOptions = buildOptions.clone();
      // Disable fat APKs for the child configurations.
      splitOptions.get(AndroidConfiguration.Options.class).fatApkCpus = ImmutableList.of();
      splitOptions.get(AndroidConfiguration.Options.class).androidPlatforms = ImmutableList.of();

      // Set the cpu & android_cpu.
      // TODO(bazel-team): --android_cpu doesn't follow --cpu right now; it should.
      splitOptions.get(AndroidConfiguration.Options.class).cpu = cpu;
      splitOptions.get(CoreOptions.class).cpu = cpu;
      setCcFlagsFromAndroid(androidOptions, splitOptions);
      // Ensure platforms aren't set so that platform mapping can take place.
      splitOptions.get(PlatformOptions.class).platforms = ImmutableList.of();
      result.put(cpu, splitOptions.underlying());
      addNonCpuSplits(result, cpu, splitOptions);
    }
    return result.build();
  }

  private void setCcFlagsFromAndroid(
      AndroidConfiguration.Options androidOptions, BuildOptionsView newOptions) {

    // Set the CC options needed to build native code.
    CppOptions newCppOptions = newOptions.get(CppOptions.class);
    newCppOptions.cppCompiler = androidOptions.cppCompiler;
    newCppOptions.libcTopLabel = androidOptions.androidLibcTopLabel;
    newCppOptions.dynamicMode = androidOptions.dynamicMode;

    if (androidOptions.androidCrosstoolTop != null) {
      newCppOptions.crosstoolTop = androidOptions.androidCrosstoolTop;
    }

    newOptions.get(AndroidConfiguration.Options.class).configurationDistinguisher =
        ConfigurationDistinguisher.ANDROID;
  }

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public void repr(Printer printer) {
    printer.append("android_common.multi_cpu_configuration");
  }
}
