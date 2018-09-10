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

package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.skylarkbuildapi.platform.PlatformConfigurationApi;
import java.util.List;

/** A configuration fragment describing the current platform configuration. */
@ThreadSafety.Immutable
public class PlatformConfiguration extends BuildConfiguration.Fragment
    implements PlatformConfigurationApi {
  private final Label hostPlatform;
  private final ImmutableList<String> extraExecutionPlatforms;
  private final Label targetPlatform;
  private final ImmutableList<String> extraToolchains;
  private final ImmutableList<Label> enabledToolchainTypes;

  PlatformConfiguration(
      Label hostPlatform,
      ImmutableList<String> extraExecutionPlatforms,
      Label targetPlatform,
      ImmutableList<String> extraToolchains,
      ImmutableList<Label> enabledToolchainTypes) {
    this.hostPlatform = hostPlatform;
    this.extraExecutionPlatforms = extraExecutionPlatforms;
    this.targetPlatform = targetPlatform;
    this.extraToolchains = extraToolchains;
    this.enabledToolchainTypes = enabledToolchainTypes;
  }

  @Override
  public Label getHostPlatform() {
    return hostPlatform;
  }

  /**
   * Target patterns that select additional platforms that will be made available for action
   * execution.
   */
  public ImmutableList<String> getExtraExecutionPlatforms() {
    return extraExecutionPlatforms;
  }

  /**
   * Returns the single target platform used in this configuration. The flag is multi-valued for
   * future handling of multiple target platforms but any given configuration should only be
   * concerned with a single target platform.
   */
  @Override
  public Label getTargetPlatform() {
    return targetPlatform;
  }

  @Override
  public ImmutableList<Label> getTargetPlatforms() {
    return ImmutableList.of(targetPlatform);
  }

  /**
   * Target patterns that select additional toolchains that will be considered during toolchain
   * resolution.
   */
  public ImmutableList<String> getExtraToolchains() {
    return extraToolchains;
  }

  @Override
  public List<Label> getEnabledToolchainTypes() {
    return enabledToolchainTypes;
  }

  public boolean isToolchainTypeEnabled(Label toolchainType) {
    return getEnabledToolchainTypes().contains(toolchainType);
  }
}
