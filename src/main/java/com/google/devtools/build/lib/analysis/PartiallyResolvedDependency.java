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
package com.google.devtools.build.lib.analysis;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Aspect;
import java.util.List;
import javax.annotation.Nullable;

/**
 * What we know about a dependency edge after factoring in the properties of the configured target
 * that the edge originates from, but not the properties of target it points to.
 */
@AutoValue
public abstract class PartiallyResolvedDependency implements BaseDependencySpecification {
  public abstract ImmutableList<Aspect> getPropagatingAspects();

  /** A Builder to create instances of PartiallyResolvedDependency. */
  @AutoValue.Builder
  abstract static class Builder {
    abstract Builder setLabel(Label label);

    abstract Builder setTransition(ConfigurationTransition transition);

    abstract Builder setPropagatingAspects(List<Aspect> propagatingAspects);

    abstract Builder setExecutionPlatformLabel(@Nullable Label executionPlatformLabel);

    abstract PartiallyResolvedDependency build();
  }

  static Builder builder() {
    return new AutoValue_PartiallyResolvedDependency.Builder()
        .setPropagatingAspects(ImmutableList.of());
  }

  public DependencyKey.Builder getDependencyKeyBuilder() {
    return DependencyKey.builder()
        .setLabel(getLabel())
        .setExecutionPlatformLabel(getExecutionPlatformLabel());
  }
}
