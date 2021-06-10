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
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.TransitionFactories;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.cpp.CppOptions;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidPlatformsTransitionApi;

/**
 * Ensures that Android binaries have a valid target platform by resetting the "--platforms" flag to
 * match the first value from "--android_platforms". This will enable the application to select a
 * valid Android SDK via toolchain resolution. android_binary itself should only need the SDK, not
 * an NDK, so in theory every platform passed to "--android_platforms" should be equivalent.
 */
public final class AndroidPlatformsTransition
    implements PatchTransition, AndroidPlatformsTransitionApi {

  public static TransitionFactory<Rule> create() {
    return TransitionFactories.of(new AndroidPlatformsTransition());
  }

  @Override
  public ImmutableSet<Class<? extends FragmentOptions>> requiresOptionFragments() {
    return ImmutableSet.of(
        AndroidConfiguration.Options.class, PlatformOptions.class, CppOptions.class);
  }

  @Override
  public BuildOptions patch(BuildOptionsView options, EventHandler eventHandler) {
    AndroidConfiguration.Options androidOptions = options.get(AndroidConfiguration.Options.class);
    if (androidOptions.androidPlatforms.isEmpty()) {
      // No change.
      return options.underlying();
    }

    BuildOptionsView newOptions = options.clone();
    PlatformOptions newPlatformOptions = newOptions.get(PlatformOptions.class);
    newPlatformOptions.platforms = ImmutableList.of(androidOptions.androidPlatforms.get(0));

    // If we are using toolchain resolution for Android, also use it for CPP.
    // This needs to be before the AndroidBinary is analyzed so that all native dependencies
    // use the same configuration.
    if (androidOptions.incompatibleUseToolchainResolution) {
      newOptions.get(CppOptions.class).enableCcToolchainResolution = true;
    }

    return newOptions.underlying();
  }

  @Override
  public String reasonForOverride() {
    return "properly set the target platform for Android binaries";
  }
}
