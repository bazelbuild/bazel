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
import com.google.devtools.build.lib.analysis.config.transitions.ComposingTransition;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleTransitionFactory;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/** A {@link RuleTransitionFactory} that composes other {@link RuleTransitionFactory}s. */
// TODO(https://github.com/bazelbuild/bazel/issues/7814): Replace with ComposingTransitionFactory.
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
    ConfigurationTransition composed =
        ComposingTransition.of(rtf1.buildTransitionFor(rule), rtf2.buildTransitionFor(rule));
    if (composed instanceof PatchTransition) {
      // This is one of the two input transitions. Especially if it's a NoTransition or
      // HostTransition, we should give it back so it can be specially identified as described
      // in composeTransitions.
      return (PatchTransition) composed;
    } else {
      // This is a composed transition, but we need a composed transition which is both a
      // PatchTransition and can be registered as equal to another instance of the same composed
      // transition.
      return new AsPatchTransition(composed);
    }
  }

  private static final class AsPatchTransition implements PatchTransition {
    private final ConfigurationTransition wrapped;

    private AsPatchTransition(ConfigurationTransition wrapped) {
      this.wrapped = wrapped;
    }

    @Override
    public BuildOptions patch(BuildOptions options) {
      return Iterables.getOnlyElement(wrapped.apply(options));
    }

    @Override
    public int hashCode() {
      return wrapped.hashCode();
    }

    @Override
    public boolean equals(Object other) {
      return other instanceof AsPatchTransition
          && this.wrapped.equals(((AsPatchTransition) other).wrapped);
    }
  }
}
