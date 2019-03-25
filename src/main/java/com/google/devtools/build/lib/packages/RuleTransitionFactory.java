// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;

/**
 * Customizable transition which accepts the rule it will be executing on.
 */
public interface RuleTransitionFactory {
  /**
   * Generates a transition to be used when entering the given rule.
   *
   * <p>This cannot be a {@link
   * com.google.devtools.build.lib.analysis.config.transitions.SplitTransition} because splits are
   * conceptually a property of the <i>parent<i> rule. In other words, it makes sense for a parent
   * to say "build my deps in configurations A and B". But it doesn't make sense for a dep to say
   * "build myself in configurations A and B" if its parent doesn't know how to intelligently handle
   * the results.
   *
   * <p>If this class determines that no transition should be performed, it should return {@code
   * NoTransition.INSTANCE}.
   */
  // TODO(bazel-team): Refactor to only take an AttributeMap. Currently the entire Rule is consumed
  // by StarlarkRuleTransitionProvider and TestTrimmingTransitionFactory.
  PatchTransition buildTransitionFor(Rule rule);
}
