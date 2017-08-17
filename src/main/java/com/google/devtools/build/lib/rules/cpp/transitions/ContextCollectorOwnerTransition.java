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

package com.google.devtools.build.lib.rules.cpp.transitions;

import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.PatchTransition;
import com.google.devtools.build.lib.rules.cpp.CppOptions;
import com.google.devtools.build.lib.rules.cpp.CppOptions.LipoConfigurationState;

/**
 * Configuration transition that creates the "artifact owner" configuration from the LIPO
 * context collector configuration.
 *
 * <p>The context collector creates C++ output artifacts but doesn't create the actions that
 * generate those artifacts (this is what {@link BuildConfiguration#isActionsEnabled()} means).
 * Those actions are the responsibility of the target configuration. This transition produces that
 * config so artifacts created by the context collector can be associated with the the right
 * "owner". Also see {@link BuildConfiguration#getArtifactOwnerTransition()}.
 *
 * <p>This is a no-op for all configurations but the context collector.
 */
public class ContextCollectorOwnerTransition implements PatchTransition {
  public static final ContextCollectorOwnerTransition INSTANCE =
      new ContextCollectorOwnerTransition();

  @Override
  public BuildOptions apply(BuildOptions options) {
    // If this target and its transitive closure don't have C++ options, there's no context
    // collector configuration to change.
    if (!options.contains(CppOptions.class)) {
      return options;
    }
    if (!options.get(CppOptions.class).isLipoContextCollector()) {
      return options;
    }
    BuildOptions ownerOptions = options.clone();
    ownerOptions.get(CppOptions.class).lipoConfigurationState = LipoConfigurationState.APPLY_LIPO;
    return ownerOptions;
  }

  @Override
  public boolean defaultsToSelf() {
    return false;
  }
}
