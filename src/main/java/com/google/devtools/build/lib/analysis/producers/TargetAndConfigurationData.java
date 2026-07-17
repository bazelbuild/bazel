// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.producers;

import com.google.devtools.build.lib.analysis.TransitiveDependencyState;
import com.google.devtools.build.lib.analysis.config.StarlarkTransitionCache;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.packages.RuleTransitionData;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import javax.annotation.Nullable;

/** Data shared by {@link TargetAndConfigurationProducer} and {@link RuleTransitionApplier}. */
public interface TargetAndConfigurationData {
  public ConfiguredTargetKey getPreRuleTransitionKey();

  @Nullable
  public TransitionFactory<RuleTransitionData> getTrimmingTransitionFactory();

  public PatchTransition getToolchainTaggedTrimmingTransition();

  public StarlarkTransitionCache getTransitionCache();

  public TransitiveDependencyState getTransitiveState();
}
