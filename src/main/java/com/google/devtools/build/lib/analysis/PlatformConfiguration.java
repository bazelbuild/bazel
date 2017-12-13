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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.PlatformOptions.ToolchainResolutionOverride;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import java.util.List;

/** A configuration fragment describing the current platform configuration. */
@ThreadSafety.Immutable
@SkylarkModule(
  name = "platform",
  doc = "The platform configuration.",
  category = SkylarkModuleCategory.CONFIGURATION_FRAGMENT
)
public class PlatformConfiguration extends BuildConfiguration.Fragment {

  private final Label executionPlatform;
  private final ImmutableList<Label> targetPlatforms;
  private final ImmutableList<Label> extraToolchains;
  private final ImmutableMap<Label, Label> toolchainResolutionOverrides;
  private final ImmutableList<Label> enabledToolchainTypes;

  public PlatformConfiguration(
      Label executionPlatform,
      List<Label> targetPlatforms,
      List<Label> extraToolchains,
      List<ToolchainResolutionOverride> overrides,
      List<Label> enabledToolchainTypes) {

    this.executionPlatform = executionPlatform;
    this.targetPlatforms = ImmutableList.copyOf(targetPlatforms);
    this.extraToolchains = ImmutableList.copyOf(extraToolchains);
    this.toolchainResolutionOverrides = convertOverrides(overrides);
    this.enabledToolchainTypes = ImmutableList.copyOf(enabledToolchainTypes);
  }

  private static ImmutableMap<Label, Label> convertOverrides(
      List<ToolchainResolutionOverride> overrides) {
    ImmutableMap.Builder<Label, Label> builder = new ImmutableMap.Builder<>();
    for (ToolchainResolutionOverride override : overrides) {
      builder.put(override.toolchainType(), override.toolchainLabel());
    }

    return builder.build();
  }

  @SkylarkCallable(
    name = "execution_platform",
    structField = true,
    doc = "The current execution platform"
  )
  public Label getExecutionPlatform() {
    return executionPlatform;
  }

  @SkylarkCallable(name = "platforms", structField = true, doc = "The current target platforms")
  public ImmutableList<Label> getTargetPlatforms() {
    return targetPlatforms;
  }

  /** Additional toolchains that should be considered during toolchain resolution. */
  public ImmutableList<Label> getExtraToolchains() {
    return extraToolchains;
  }

  /** Returns {@code true} if the given toolchain type has a manual override set. */
  public boolean hasToolchainOverride(Label toolchainType) {
    return toolchainResolutionOverrides.containsKey(toolchainType);
  }

  /** Returns the {@link Label} of the toolchain to use for the given toolchain type. */
  public Label getToolchainOverride(Label toolchainType) {
    return toolchainResolutionOverrides.get(toolchainType);
  }

  @SkylarkCallable(
    name = "enabled_toolchain_types",
    structField = true,
    doc = "The set of toolchain types enabled for platform-based toolchain selection."
  )
  public List<Label> getEnabledToolchainTypes() {
    return enabledToolchainTypes;
  }
}
