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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.cpp.CppOptions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;

/**
 * Transition that produces a configuration that causes c++ toolchain selection to use the CROSSTOOL
 * given in apple_crosstool_top.
 */
// TODO(bazel-team): replace this with objc_library.bzl#apple_crosstool_transition when all
// references move to Starlark.
public final class AppleCrosstoolTransition implements PatchTransition {

  /** A singleton instance of AppleCrosstoolTransition. */
  @SerializationConstant
  public static final PatchTransition APPLE_CROSSTOOL_TRANSITION = new AppleCrosstoolTransition();

  private AppleCrosstoolTransition() {}

  @Override
  public ImmutableSet<Class<? extends FragmentOptions>> requiresOptionFragments() {
    return ImmutableSet.of(
        AppleCommandLineOptions.class, CoreOptions.class, CppOptions.class, PlatformOptions.class);
  }

  @Override
  public BuildOptions patch(BuildOptionsView buildOptions, EventHandler eventHandler) {
    AppleCommandLineOptions appleOptions = buildOptions.get(AppleCommandLineOptions.class);

    if (appleOptions.configurationDistinguisher != ConfigurationDistinguisher.UNKNOWN) {
      // The configuration distinguisher is only set by AppleCrosstoolTransition and
      // AppleBinaryTransition, both of which also set the Crosstool and the CPU to Apple ones.
      // So we are fine not doing anything.
      return buildOptions.underlying();
    }

    CoreOptions configOptions = buildOptions.get(CoreOptions.class);
    String cpu =
        ApplePlatform.cpuStringForTarget(
            appleOptions.applePlatformType,
            determineSingleArchitecture(appleOptions, configOptions));

    // Avoid a clone if nothing would change.
    if (configOptions.cpu.equals(cpu)
        && buildOptions.get(CppOptions.class).crosstoolTop.equals(appleOptions.appleCrosstoolTop)) {
      return buildOptions.underlying();
    }

    BuildOptionsView result = buildOptions.clone();
    setAppleCrosstoolTransitionConfiguration(buildOptions, result, cpu);
    return result.underlying();
  }

  /**
   * Sets configuration fields required for a transition that uses apple_crosstool_top in place of
   * the default CROSSTOOL.
   *
   * @param from options from the originating configuration
   * @param to options for the destination configuration. This instance will be modified to so the
   *     destination configuration uses the apple crosstool
   * @param cpu {@code --cpu} value for toolchain selection in the destination configuration
   */
  public static void setAppleCrosstoolTransitionConfiguration(
      BuildOptionsView from, BuildOptionsView to, String cpu) {
    AppleCommandLineOptions appleOptions = from.get(AppleCommandLineOptions.class);

    CoreOptions toOptions = to.get(CoreOptions.class);
    CppOptions toCppOptions = to.get(CppOptions.class);

    if (toOptions.cpu.equals(cpu)
        && toCppOptions.crosstoolTop.equals(appleOptions.appleCrosstoolTop)) {
      // If neither the CPU nor the Crosstool changes, do nothing. This is so that C++ to
      // Objective-C dependencies work if the top-level configuration is already an Apple one.
      // Removing the configuration distinguisher (which can't be set from the command line) and
      // putting the platform type in the output directory name, which would obviate the need for
      // this hack.
      // TODO(b/112834725): Remove this branch by unifying the distinguisher and the platform type.
      return;
    }

    toOptions.cpu = cpu;
    toCppOptions.crosstoolTop = appleOptions.appleCrosstoolTop;
    to.get(AppleCommandLineOptions.class).configurationDistinguisher =
        ConfigurationDistinguisher.APPLE_CROSSTOOL;

    toCppOptions.cppCompiler = appleOptions.cppCompiler;
    toCppOptions.libcTopLabel = appleOptions.appleLibcTop;

    // OSX toolchains do not support fission.
    toCppOptions.fissionModes = ImmutableList.of();

    // Ensure platforms aren't set so that platform mapping can take place.
    to.get(PlatformOptions.class).platforms = ImmutableList.of();
  }

  /** Returns the Apple architecture implied by AppleCommandLineOptions and CoreOptions */
  private String determineSingleArchitecture(
      AppleCommandLineOptions appleOptions, CoreOptions configOptions) {
    if (!Strings.isNullOrEmpty(appleOptions.appleSplitCpu)) {
      return appleOptions.appleSplitCpu;
    }
    switch (appleOptions.applePlatformType) {
      case IOS:
        if (!appleOptions.iosMultiCpus.isEmpty()) {
          return appleOptions.iosMultiCpus.get(0);
        } else {
          return AppleConfiguration.iosCpuFromCpu(configOptions.cpu);
        }
      case WATCHOS:
        if (appleOptions.watchosCpus.isEmpty()) {
          return AppleCommandLineOptions.DEFAULT_WATCHOS_CPU;
        }
        return appleOptions.watchosCpus.get(0);
      case TVOS:
        if (appleOptions.tvosCpus.isEmpty()) {
          return AppleCommandLineOptions.DEFAULT_TVOS_CPU;
        }
        return appleOptions.tvosCpus.get(0);
      case MACOS:
        if (appleOptions.macosCpus.isEmpty()) {
          return AppleCommandLineOptions.DEFAULT_MACOS_CPU;
        }
        return appleOptions.macosCpus.get(0);
      case CATALYST:
        if (appleOptions.catalystCpus.isEmpty()) {
          return AppleCommandLineOptions.DEFAULT_CATALYST_CPU;
        }
        return appleOptions.catalystCpus.get(0);
      default:
        throw new IllegalArgumentException(
            "Unhandled platform type " + appleOptions.applePlatformType);
    }
  }
}
