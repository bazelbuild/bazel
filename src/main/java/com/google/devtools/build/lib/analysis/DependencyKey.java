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
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.cmdline.Label;

/**
 * Information about a dependency, including the label and configuration transition. This will be
 * used to create the actual {@link Dependency}.
 */
@AutoValue
public abstract class DependencyKey {

  /** Builder to help construct instances of {@link DependencyKey}. */
  @AutoValue.Builder
  public interface Builder {
    /** Sets the label of the target this dependency points to. */
    Builder setLabel(Label label);

    /** Sets the transition to use when evaluating the target this dependency points to. */
    Builder setTransition(ConfigurationTransition transition);

    /** Sets the aspects that are propagating to the target this dependency points to. */
    Builder setAspects(AspectCollection aspectCollection);

    /** Returns the new instance. */
    DependencyKey build();
  }

  /** Returns a new {@link Builder} to construct instances of {@link DependencyKey}. */
  public static Builder builder() {
    return new AutoValue_DependencyKey.Builder().setAspects(AspectCollection.EMPTY);
  }

  /** Returns the label of the target this dependency points to. */
  public abstract Label getLabel();

  /** Returns the transition to use when evaluating the target this dependency points to. */
  public abstract ConfigurationTransition getTransition();

  /** Returns the aspects that are propagating to the target this dependency points to. */
  public abstract AspectCollection getAspects();
}
