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

import com.google.common.base.MoreObjects;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.Objects;

import javax.annotation.Nullable;

/**
* A (label,configuration) pair.
*/
public final class LabelAndConfiguration {
  private final Label label;
  @Nullable
  private final BuildConfiguration configuration;

  private LabelAndConfiguration(Label label, @Nullable BuildConfiguration configuration) {
    this.label = Preconditions.checkNotNull(label);
    this.configuration = configuration;
  }

  public LabelAndConfiguration(ConfiguredTarget rule) {
    this(rule.getTarget().getLabel(), rule.getConfiguration());
  }

  public Label getLabel() {
    return label;
  }

  @Nullable
  public BuildConfiguration getConfiguration() {
    return configuration;
  }

  @Override
  public int hashCode() {
    int configVal = configuration == null ? 79 : configuration.hashCode();
    return 31 * label.hashCode() + configVal;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (!(obj instanceof LabelAndConfiguration)) {
      return false;
    }
    LabelAndConfiguration other = (LabelAndConfiguration) obj;
    return Objects.equals(label, other.label) && Objects.equals(configuration, other.configuration);
  }

  public static LabelAndConfiguration of(
      Label label, @Nullable BuildConfiguration configuration) {
    return new LabelAndConfiguration(label, configuration);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("label", label)
        .add("configuration", configuration)
        .toString();
  }
}
