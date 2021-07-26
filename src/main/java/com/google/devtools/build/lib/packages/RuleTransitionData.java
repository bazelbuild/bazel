// Copyright 2021 The Bazel Authors. All rights reserved.
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
 * rules.
 */
@AutoValue
public abstract class RuleTransitionData implements TransitionFactory.Data {

  public static RuleTransitionData create(Rule rule) {
    return new AutoValue_RuleTransitionData(rule);
  }

  public abstract Rule rule();
}
