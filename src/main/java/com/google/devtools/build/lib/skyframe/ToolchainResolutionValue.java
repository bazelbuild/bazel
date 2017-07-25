// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/** A value which represents the selected toolchain for a specific target and execution platform. */
@AutoValue
public abstract class ToolchainResolutionValue implements SkyValue {

  // A key representing the input data.
  public static SkyKey key(
      BuildConfiguration configuration,
      Label toolchainType,
      PlatformInfo targetPlatform,
      PlatformInfo execPlatform) {
    return ToolchainResolutionKey.create(
        configuration, toolchainType, targetPlatform, execPlatform);
  }

  /** {@link SkyKey} implementation used for {@link ToolchainResolutionFunction}. */
  @AutoValue
  public abstract static class ToolchainResolutionKey implements SkyKey {
    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.TOOLCHAIN_RESOLUTION;
    }

    public abstract BuildConfiguration configuration();

    public abstract Label toolchainType();

    public abstract PlatformInfo targetPlatform();

    public abstract PlatformInfo execPlatform();

    public static ToolchainResolutionKey create(
        BuildConfiguration configuration,
        Label toolchainType,
        PlatformInfo targetPlatform,
        PlatformInfo execPlatform) {
      return new AutoValue_ToolchainResolutionValue_ToolchainResolutionKey(
          configuration, toolchainType, targetPlatform, execPlatform);
    }
  }

  public static ToolchainResolutionValue create(Label toolchainLabel) {
    return new AutoValue_ToolchainResolutionValue(toolchainLabel);
  }

  public abstract Label toolchainLabel();
}
