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


package com.google.devtools.build.lib.analysis.config;

import com.google.devtools.build.lib.analysis.config.transitions.ComposingPatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.Transition;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleTransitionFactory;

/**
 * A {@link RuleTransitionFactory} that composes other {@link RuleTransitionFactory}s.
 */
public class ComposingRuleTransitionFactory implements RuleTransitionFactory {

  private final RuleTransitionFactory rtf1;
  private final RuleTransitionFactory rtf2;

  /**
   * Creates a factory that applies the given factories in-order ({@code rtf1} first,
   * {@code rtf2} second).
   */
  public ComposingRuleTransitionFactory(RuleTransitionFactory rtf1, RuleTransitionFactory rtf2) {
    this.rtf1 = rtf1;
    this.rtf2 = rtf2;
  }

  @Override
  public Transition buildTransitionFor(Rule rule) {
    PatchTransition transition1 = (PatchTransition) rtf1.buildTransitionFor(rule);
    PatchTransition transition2 = (PatchTransition) rtf2.buildTransitionFor(rule);

    if (transition1 == null) {
      return transition2;
    }

    if (transition2 == null) {
      return transition1;
    }

    return new ComposingPatchTransition(
        (PatchTransition) rtf1.buildTransitionFor(rule),
        (PatchTransition) rtf2.buildTransitionFor(rule));
  }
}
