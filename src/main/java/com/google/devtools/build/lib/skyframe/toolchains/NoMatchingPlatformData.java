// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.toolchains;

import static java.util.stream.Collectors.joining;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.ToolchainTypeRequirement;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;

/** Contains information related to missing execution platform. */
@AutoValue
public abstract class NoMatchingPlatformData {
  abstract ImmutableSet<ToolchainTypeRequirement> toolchainTypes();

  abstract ImmutableList<ConfiguredTargetKey> availableExecutionPlatformKeys();

  abstract ConfiguredTargetKey targetPlatformKey();

  static Builder builder() {
    return new AutoValue_NoMatchingPlatformData.Builder();
  }

  @AutoValue.Builder
  abstract static class Builder {
    abstract Builder setToolchainTypes(ImmutableSet<ToolchainTypeRequirement> toolchainTypes);

    abstract Builder setAvailableExecutionPlatformKeys(
        ImmutableList<ConfiguredTargetKey> availableExecutionPlatformKeys);

    abstract Builder setTargetPlatformKey(ConfiguredTargetKey targetPlatformKey);

    abstract NoMatchingPlatformData build();
  }

  String formatError() {
    if (this.toolchainTypes().isEmpty()) {
      return String.format(
          "Unable to find an execution platform for target platform %s"
              + " from available execution platforms [%s]",
          this.targetPlatformKey().getLabel(),
          this.availableExecutionPlatformKeys().stream()
              .map(key -> key.getLabel().toString())
              .collect(joining(", ")));
    }
    return String.format(
        "Unable to find an execution platform for toolchains [%s] and target platform %s"
            + " from available execution platforms [%s]",
        this.toolchainTypes().stream()
            .map(ToolchainTypeRequirement::toolchainType)
            .map(Label::toString)
            .collect(joining(", ")),
        this.targetPlatformKey().getLabel(),
        this.availableExecutionPlatformKeys().stream()
            .map(key -> key.getLabel().toString())
            .collect(joining(", ")));
  }
}
