// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.cmdline.Label;
import javax.annotation.Nullable;

/**
 * Helper class which contains data used by a {@link TransitionFactory} to create a transition for
 * attributes.
 */
@AutoValue
public abstract class AttributeTransitionData implements TransitionFactory.Data {
  /** Returns the {@link AttributeMap} which can be used to create a transition. */
  public abstract AttributeMap attributes();

  /**
   * Returns the {@link Label} of the execution platform used by the configured target this
   * transition factory is part of.
   */
  @Nullable
  public abstract Label executionPlatform();

  /** Returns a new {@link Builder} for {@link AttributeTransitionData}. */
  public static Builder builder() {
    return new AutoValue_AttributeTransitionData.Builder();
  }

  /** Builder class for {@link AttributeTransitionData}. */
  @AutoValue.Builder
  public abstract static class Builder {
    /** Sets the attributes. */
    public abstract Builder attributes(AttributeMap attributes);

    /** Sets the execution platform label. */
    public abstract Builder executionPlatform(@Nullable Label executionPlatform);

    /** Returns the new {@link AttributeTransitionData}. */
    public abstract AttributeTransitionData build();
  }
}
