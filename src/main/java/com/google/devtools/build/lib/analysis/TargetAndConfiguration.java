// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * Refers to the pair of a target and a configuration. Not the same as {@link ConfiguredTarget} -
 * that also contains the result of the analysis phase.
 */
@Immutable
public final class TargetAndConfiguration {
  private final Target target;
  @Nullable private final BuildConfiguration configuration;

  public TargetAndConfiguration(Target target, @Nullable BuildConfiguration configuration) {
    this.target = Preconditions.checkNotNull(target);
    this.configuration = configuration;
  }

  // The node name in the graph. The name should be unique.
  // It is not suitable for user display.
  public String getName() {
    return target.getLabel() + " "
        + (configuration == null ? "null" : configuration.checksum());
  }

  @Override
  public boolean equals(Object that) {
    if (this == that) {
      return true;
    }
    if (!(that instanceof TargetAndConfiguration)) {
      return false;
    }

    TargetAndConfiguration thatNode = (TargetAndConfiguration) that;
    return thatNode.target.getLabel().equals(this.target.getLabel()) &&
        thatNode.configuration == this.configuration;
  }

  @Override
  public int hashCode() {
    return Objects.hash(target.getLabel(), configuration);
  }

  @Override
  public String toString() {
    return target.getLabel() + " (" + configuration + ")";
  }

  public Target getTarget() {
    return target;
  }

  public Label getLabel() {
    return target.getLabel();
  }

  @Nullable
  public BuildConfiguration getConfiguration() {
    return configuration;
  }

  public ConfiguredTargetKey getConfiguredTargetKey() {
    return ConfiguredTargetKey.builder()
        .setLabel(getLabel())
        .setConfiguration(getConfiguration())
        .build();
  }
}
