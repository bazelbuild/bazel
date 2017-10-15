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

import com.google.devtools.build.lib.packages.Attribute.Transition;
import javax.annotation.Nullable;

/**
 * Customizable transition which accepts the rule's nonconfigurable attributes.
 */
public interface RuleTransitionFactory {
  /**
   * Generates a transition to be used when entering the given rule.
   *
   * <p>This transition must be a PatchTransition, but that class is not accessible in this package.
   * If this class determines that no transition should be performed, it should return {@code null}.
   */
  @Nullable
  Transition buildTransitionFor(Rule rule);
}
