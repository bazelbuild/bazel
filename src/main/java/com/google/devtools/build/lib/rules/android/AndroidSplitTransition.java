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
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.SplitTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.AttributeTransitionData;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.cpp.CppOptions;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidSplitTransitionApi;
import java.util.List;
import net.starlark.java.eval.Printer;

/** Android Split configuration transition for properly handling native dependencies */
public final class AndroidSplitTransition implements SplitTransition, AndroidSplitTransitionApi {

  public static final AndroidSplitTransition INSTANCE = new AndroidSplitTransition();

  public static final TransitionFactory<AttributeTransitionData> FACTORY = new Factory();

  /** A {@link TransitionFactory} instance that returns the {@link AndroidSplitTransition}. */
  static final class Factory implements TransitionFactory<AttributeTransitionData> {

    @Override
    public ConfigurationTransition create(AttributeTransitionData unused) {
      return INSTANCE;
    }

    @Override
    public TransitionType transitionType() {
      return TransitionType.ATTRIBUTE;
    }

    @Override
    public boolean isTool() {
      return false;
    }

    @Override
    public boolean isSplit() {
      return true;
    }
  }

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

      List<Label> platformsToSplit = androidOptions.androidPlatforms;
      if (platformsToSplit.isEmpty()) {
        // If --android_platforms is unset, instead use only the first value from --platforms.
        Label targetPlatform =
            Iterables.getFirst(buildOptions.get(PlatformOptions.class).platforms, null);
        platformsToSplit = ImmutableList.of(targetPlatform);
      }
      return handleAndroidPlatforms(buildOptions, androidOptions, platformsToSplit);
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
    }
    return result.buildOrThrow();
  }

  private void setCcFlagsFromAndroid(
      AndroidConfiguration.Options androidOptions, BuildOptionsView newOptions) {

    // Set the CC options needed to build native code.
    CppOptions newCppOptions = newOptions.get(CppOptions.class);
    newCppOptions.cppCompiler = androidOptions.cppCompiler;
    newCppOptions.dynamicMode = androidOptions.dynamicMode;

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
