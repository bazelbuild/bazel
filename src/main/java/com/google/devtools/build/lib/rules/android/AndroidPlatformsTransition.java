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
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.StarlarkExposedRuleTransitionFactory;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.analysis.starlark.FunctionTransitionUtil;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.RuleTransitionData;

/**
 * Ensures that Android binaries have a valid target platform by resetting the "--platforms" flag to
 * match the first value from "--android_platforms". This will enable the application to select a
 * valid Android SDK via toolchain resolution. android_binary itself should only need the SDK, not
 * an NDK, so in theory every platform passed to "--android_platforms" should be equivalent.
 */
public final class AndroidPlatformsTransition implements PatchTransition {
  private static final AndroidPlatformsTransition INSTANCE = new AndroidPlatformsTransition();

  /** Machinery to expose the transition to Starlark. */
  public static final class AndroidPlatformsTransitionFactory
      implements StarlarkExposedRuleTransitionFactory {
    @Override
    public PatchTransition create(RuleTransitionData unused) {
      return INSTANCE;
    }
  }

  public static TransitionFactory<RuleTransitionData> create() {
    return new AndroidPlatformsTransitionFactory();
  }

  @Override
  public ImmutableSet<Class<? extends FragmentOptions>> requiresOptionFragments() {
    return ImmutableSet.of(
        AndroidConfiguration.Options.class, PlatformOptions.class, CoreOptions.class);
  }

  @Override
  public BuildOptions patch(BuildOptionsView options, EventHandler eventHandler) {
    AndroidConfiguration.Options androidOptions = options.get(AndroidConfiguration.Options.class);
    BuildOptionsView newOptions = options.clone();
    PlatformOptions newPlatformOptions = newOptions.get(PlatformOptions.class);
    // Set the value of --platforms for this target and its dependencies.
    // 1. If --android_platforms is set, use a value from that.
    // 2. Otherwise, leave --platforms alone (this will probably lead to build errors).
    if (!androidOptions.androidPlatforms.isEmpty()) {
      // If the current value of --platforms is not one of the values of --android_platforms, change
      // it to be the first one. If the curent --platforms is part of --android_platforms, leave it
      // as-is.
      // NOTE: This does not handle aliases at all, so if someone is using aliases with platform
      // definitions this check will break.
      if (!androidOptions.androidPlatforms.containsAll(newPlatformOptions.platforms)) {
        newPlatformOptions.platforms = ImmutableList.of(androidOptions.androidPlatforms.get(0));
      }
    }

    if (androidOptions.androidPlatformsTransitionsUpdateAffected) {
      ImmutableSet.Builder<String> affected = ImmutableSet.builder();
      if (!options
          .get(PlatformOptions.class)
          .platforms
          .equals(newOptions.get(PlatformOptions.class).platforms)) {
        affected.add("//command_line_option:platforms");
      }
      FunctionTransitionUtil.updateAffectedByStarlarkTransition(
          newOptions.get(CoreOptions.class), affected.build());
    }

    return newOptions.underlying();
  }

  @Override
  public String reasonForOverride() {
    return "properly set the target platform for Android binaries";
  }
}
