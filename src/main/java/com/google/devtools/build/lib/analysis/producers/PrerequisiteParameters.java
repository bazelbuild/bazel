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

import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skyframe.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import javax.annotation.Nullable;
import net.starlark.java.syntax.Location;

/** Common parameters for computing prerequisites. */
public final class PrerequisiteParameters {
  private final ConfiguredTargetKey configuredTargetKey;
  @Nullable private final Rule associatedRule;

  private final TransitiveDependencyState transitiveState;

  public PrerequisiteParameters(
      ConfiguredTargetKey configuredTargetKey,
      @Nullable Rule associatedRule,
      TransitiveDependencyState transitiveState) {
    this.configuredTargetKey = configuredTargetKey;
    this.associatedRule = associatedRule;
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

  public TransitiveDependencyState transitiveState() {
    return transitiveState;
  }
}
