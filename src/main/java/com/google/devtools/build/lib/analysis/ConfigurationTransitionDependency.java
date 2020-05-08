// Copyright 2020 The Bazel Authors. All rights reserved.
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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * Implementation of a dependency with a given configuration transition.
 */
@AutoValue
public abstract class ConfigurationTransitionDependency {
  /** Returns the label of the target this dependency points to. */
  public abstract Label getLabel();

  /** Returns the configuration transition used by this dependency. */
  public abstract ConfigurationTransition getTransition();

  /** Returns any aspects that need to be propogated along this dependency. */
  public abstract AspectCollection getAspects();

  /** Returns a new Builder for creating instances. */
  public static ConfigurationTransitionBuilder builder() {
    return new AutoValue_ConfigurationTransitionDependency.Builder()
        .setAspects(AspectCollection.EMPTY);
  }

  /** Builder to assist in creating dependency instances with a configuration transition. */
  @AutoValue.Builder
  public static abstract class ConfigurationTransitionBuilder {
    /** Sets the label of the target this dependency points to. */
    public abstract ConfigurationTransitionBuilder setLabel(Label label);

    /** Set the configuration transition used by this dependency. */
    public abstract ConfigurationTransitionBuilder setTransition(ConfigurationTransition transition);

    /**
     * Add aspects to this Dependency.
     */
    public abstract ConfigurationTransitionBuilder setAspects(AspectCollection aspectCollection);

    /** Returns the full Dependency instance. */
    public abstract ConfigurationTransitionDependency build();
  }
}
