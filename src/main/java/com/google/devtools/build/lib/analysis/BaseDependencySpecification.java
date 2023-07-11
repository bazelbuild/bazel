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

import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.cmdline.Label;
import javax.annotation.Nullable;

/**
 * Common specification of a dependency.
 *
 * <p>Omits which aspects are needed, which has an implementation-determined type.
 */
public interface BaseDependencySpecification {
  /** Returns the label of the target this dependency points to. */
  public abstract Label getLabel();

  /** Returns the transition to use when evaluating the target this dependency points to. */
  public abstract ConfigurationTransition getTransition();

  /**
   * Returns the execution platform {@link Label} that this dependency should use as an override for
   * toolchain resolution.
   */
  @Nullable
  public abstract Label getExecutionPlatformLabel();
}
