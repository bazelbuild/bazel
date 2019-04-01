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

/**
 * Helper class which contains data used by a {@link TransitionFactory} to create a transition for
 * rules and attributes.
 */
// This class is in lib.packages in order to access AttributeMap, which is not available to
// the lib.analysis.config.transitions package.
@AutoValue
public abstract class RuleTransitionData {
  /** Returns the {@link AttributeMap} which can be used to create a transition. */
  public abstract AttributeMap attributes();

  // TODO(https://github.com/bazelbuild/bazel/issues/7814): Add further data fields as needed by
  // transition factory instances.

  /** Returns a new {@link RuleTransitionData} instance. */
  public static RuleTransitionData create(AttributeMap attributes) {
    return new AutoValue_RuleTransitionData(attributes);
  }
}
