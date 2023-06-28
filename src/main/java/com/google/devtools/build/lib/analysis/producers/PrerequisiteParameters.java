// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.producers;

import static com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil.configurationId;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ToolchainCollection;
import com.google.devtools.build.lib.analysis.ToolchainContext;
import com.google.devtools.build.lib.analysis.TransitiveDependencyState;
import com.google.devtools.build.lib.analysis.config.StarlarkTransitionCache;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skyframe.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import javax.annotation.Nullable;
import net.starlark.java.syntax.Location;

/** Common parameters for computing prerequisites. */
public final class PrerequisiteParameters {
  private final ConfiguredTargetKey configuredTargetKey;
  @Nullable private final Rule associatedRule;

  private final ImmutableList<Aspect> aspects;
  private final StarlarkTransitionCache transitionCache;
  @Nullable private final ToolchainCollection<ToolchainContext> toolchainContexts;

  @Nullable private final ConfiguredAttributeMapper attributeMap;
  private final TransitiveDependencyState transitiveState;

  public PrerequisiteParameters(
      ConfiguredTargetKey configuredTargetKey,
      @Nullable Rule associatedRule,
      Iterable<Aspect> aspects,
      StarlarkTransitionCache transitionCache,
      @Nullable ToolchainCollection<ToolchainContext> toolchainContexts,
      @Nullable ConfiguredAttributeMapper attributeMap,
      TransitiveDependencyState transitiveState) {
    this.configuredTargetKey = configuredTargetKey;
    this.associatedRule = associatedRule;
    this.aspects = ImmutableList.copyOf(aspects);
    this.transitionCache = transitionCache;
    this.toolchainContexts = toolchainContexts;
    this.attributeMap = attributeMap;
    this.transitiveState = transitiveState;
  }

  public Label label() {
    return configuredTargetKey.getLabel();
  }

  @Nullable
  public Rule associatedRule() {
    return associatedRule;
  }

  @Nullable
  public BuildConfigurationKey configurationKey() {
    return configuredTargetKey.getConfigurationKey();
  }

  public ImmutableList<Aspect> aspects() {
    return aspects;
  }

  public StarlarkTransitionCache transitionCache() {
    return transitionCache;
  }

  @Nullable
  public ToolchainCollection<ToolchainContext> toolchainContexts() {
    return toolchainContexts;
  }

  @Nullable // Non-null for rules, and output files when there are aspects that apply to files.
  public ConfiguredAttributeMapper attributeMap() {
    return attributeMap;
  }

  @Nullable
  public Location location() {
    if (associatedRule == null) {
      return null;
    }
    return associatedRule.getLocation();
  }

  public BuildEventId eventId() {
    return configurationId(configurationKey());
  }

  @Nullable
  public Label getExecutionPlatformLabel(String execGroup) {
    var platform = toolchainContexts.getToolchainContext(execGroup).executionPlatform();
    if (platform == null) {
      return null;
    }
    return platform.label();
  }

  public TransitiveDependencyState transitiveState() {
    return transitiveState;
  }
}
