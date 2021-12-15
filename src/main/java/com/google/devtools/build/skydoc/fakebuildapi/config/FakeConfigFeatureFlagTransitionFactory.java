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

package com.google.devtools.build.skydoc.fakebuildapi.config;

import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.StarlarkExposedRuleTransitionFactory;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleTransitionData;

/** Fake callable implementation of {@link StarlarkExposedRuleTransitionFactory}. */
public class FakeConfigFeatureFlagTransitionFactory
    implements StarlarkExposedRuleTransitionFactory {

  @Override
  public void addToStarlarkRule(RuleDefinitionEnvironment ctx, RuleClass.Builder builder) {}

  @Override
  public PatchTransition create(RuleTransitionData ruleData) {
    return null;
  }
}
