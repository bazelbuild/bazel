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

import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.PatchTransition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.cpp.CppOptions;
import com.google.devtools.build.lib.rules.cpp.CppOptions.LipoConfigurationState;

/**
 * Configuration transition that turns on LIPO/FDO settings for configurations that have them
 * disabled.
 */
public class EnableLipoTransition implements PatchTransition {
  private final Label ruleLabel;

  /**
   * Creates a new transition that only triggers on the given rule. This can be used for
   * restricting this transition to the LIPO context binary.
   */
  public EnableLipoTransition(Label ruleLabel) {
    this.ruleLabel = ruleLabel;
  }

  @Override
  public BuildOptions apply(BuildOptions options) {
    CppOptions cppOptions = options.get(CppOptions.class);
    if (!cppOptions.isDataConfigurationForLipoOptimization()
        || !ruleLabel.equals(cppOptions.getLipoContextForBuild())) {
      return options;
    }
    BuildOptions lipoEnabledOptions = options.clone();
    lipoEnabledOptions.get(CppOptions.class).lipoConfigurationState =
        LipoConfigurationState.APPLY_LIPO;
    return lipoEnabledOptions;
  }

  @Override
  public boolean defaultsToSelf() {
    return false;
  }
}
