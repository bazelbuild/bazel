// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.skyframe.NodeType;

import java.util.Objects;

import javax.annotation.Nullable;

/**
 * A (Label, Configuration) pair.
 */
public final class LabelAndConfiguration extends ActionLookupNode.ActionLookupKey {
  private final Label label;
  @Nullable private final BuildConfiguration configuration;

  public LabelAndConfiguration(Label label, @Nullable BuildConfiguration configuration) {
    this.label = Preconditions.checkNotNull(label);
    this.configuration = configuration;
  }

  public LabelAndConfiguration(ConfiguredTarget rule) {
    this(rule.getTarget().getLabel(), rule.getConfiguration());
  }

  @Override
  public Label getLabel() {
    return label;
  }

  @Override
  NodeType getType() {
    return NodeTypes.CONFIGURED_TARGET;
  }

  @Nullable
  public BuildConfiguration getConfiguration() {
    return configuration;
  }

  @Override
  public int hashCode() {
    return Objects.hash(label, configuration);
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

  public String prettyPrint() {
    if (label == null) {
      return "null";
    }
    return (configuration != null && configuration.isHostConfiguration())
        ? (label.toString() + " (host)") : label.toString();
  }

  @Override
  public String toString() {
    return label + " " + (configuration == null ? "null" : configuration.shortCacheKey());
  }
}
