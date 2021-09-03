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

package com.google.devtools.build.lib.analysis.config.transitions;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleTransitionData;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.StarlarkValue;

/** A {@link TransitionFactory} that is exposed to Starlark. */
@StarlarkBuiltin(
    name = "native_rule_transition",
    category = DocCategory.BUILTIN,
    doc =
        "Represents a native transition that can be applied to a Starlark rule as an incoming edge."
            + " This is a valid value for cfg in rule() but not attr().")
public interface StarlarkExposedRuleTransitionFactory
    extends TransitionFactory<RuleTransitionData>, StarlarkValue {

  /**
   * Called during RuleClass construction, when the transition is used by a Starlark rule cfg.
   *
   * <p>Intent is to allow transition to e.g. add or validate attributes it needs to function, add
   * an AllowlistChecker to even engage the transtion, etc.
   */
  default void addToStarlarkRule(RuleDefinitionEnvironment env, RuleClass.Builder builder) {}
}
