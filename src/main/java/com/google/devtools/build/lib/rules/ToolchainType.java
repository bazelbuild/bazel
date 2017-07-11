// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.TreeMap;

/**
 * Abstract base class for {@code toolchain_type}.
 */
public class ToolchainType implements RuleConfiguredTargetFactory {
  private final ImmutableMap<Label, Class<? extends BuildConfiguration.Fragment>> fragmentMap;
  private final ImmutableMap<Label, ImmutableMap<String, String>> hardcodedVariableMap;

  protected ToolchainType(
      ImmutableMap<Label, Class<? extends BuildConfiguration.Fragment>> fragmentMap,
      ImmutableMap<Label, ImmutableMap<String, String>> hardcodedVariableMap) {
    this.fragmentMap = fragmentMap;
    this.hardcodedVariableMap = hardcodedVariableMap;
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    // This cannot be an ImmutableMap.Builder because that asserts when a key is duplicated
    TreeMap<String, String> makeVariables = new TreeMap<>();
    Class<? extends BuildConfiguration.Fragment> fragmentClass =
        fragmentMap.get(ruleContext.getLabel());
    if (fragmentClass != null) {
      BuildConfiguration.Fragment fragment = ruleContext.getFragment(fragmentClass);
      ImmutableMap.Builder<String, String> fragmentBuilder = ImmutableMap.builder();
      fragment.addGlobalMakeVariables(fragmentBuilder);
      makeVariables.putAll(fragmentBuilder.build());
    }

    ImmutableMap<String, String> hardcodedVariables =
        hardcodedVariableMap.get(ruleContext.getLabel());
    if (hardcodedVariables != null) {
      makeVariables.putAll(hardcodedVariables);
    }
    // This will eventually go to Skyframe using getAnalysisEnvironment.getSkyframeEnv() to figure
    // out the lookup rule -> toolchain rule mapping. For now, it only provides Make variables that
    // come from BuildConfiguration so no need to ask Skyframe.
    return new RuleConfiguredTargetBuilder(ruleContext)
        .addNativeDeclaredProvider(new MakeVariableProvider(ImmutableMap.copyOf(makeVariables)))
        .addProvider(RunfilesProvider.simple(Runfiles.EMPTY))
        .build();
  }
}
