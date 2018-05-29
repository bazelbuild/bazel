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

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleTransitionFactory;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/** A {@link RuleTransitionFactory} that composes other {@link RuleTransitionFactory}s. */
@AutoCodec
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
  public PatchTransition buildTransitionFor(Rule rule) {
    ConfigurationTransition composedTransition = TransitionResolver.composeTransitions(
        rtf1.buildTransitionFor(rule), rtf2.buildTransitionFor(rule));
    // Even though we know the composed transition isn't a split (because neither of its children
    // can be splits), the returned type is a generic ConfigurationTransition. So cast that back to
    // a PatchTransition here.
    //
    // We could alternatively change RuleTransitionFactory's signature to a ConfigurationTransition.
    // But it's nice to strongly enforce the interface expectation that rule transitions don't
    // split.
    return (BuildOptions options) -> Iterables.getOnlyElement(composedTransition.apply(options));
  }
}
