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

package com.google.devtools.build.lib.rules;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TemplateVariableInfo;
import com.google.devtools.build.lib.analysis.ToolchainContext.ResolvedToolchainProviders;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Options;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Options.MakeVariableSource;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.TreeMap;

/**
 * Abstract base class for {@code toolchain_type}.
 */
public class ToolchainType implements RuleConfiguredTargetFactory {
  private ImmutableMap<Label, Class<? extends BuildConfiguration.Fragment>> fragmentMap;
  private final Function<RuleContext, ImmutableMap<Label, Class<? extends Fragment>>>
      fragmentMapFromRuleContext;
  private final ImmutableMap<Label, ImmutableMap<String, String>> hardcodedVariableMap;

  protected ToolchainType(
      ImmutableMap<Label, Class<? extends BuildConfiguration.Fragment>> fragmentMap,
      ImmutableMap<Label, ImmutableMap<String, String>> hardcodedVariableMap) {
    this.fragmentMap = fragmentMap;
    this.fragmentMapFromRuleContext = null;
    this.hardcodedVariableMap = hardcodedVariableMap;
  }

  // This constructor is required to allow for toolchains that depend on the tools repository,
  // which depends on RuleContext.
  protected ToolchainType(
      Function<RuleContext, ImmutableMap<Label, Class<? extends Fragment>>>
          fragmentMapFromRuleContext,
      ImmutableMap<Label, ImmutableMap<String, String>> hardcodedVariableMap) {
    this.fragmentMap = null;
    this.fragmentMapFromRuleContext = fragmentMapFromRuleContext;
    this.hardcodedVariableMap = hardcodedVariableMap;
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    if (fragmentMap == null && fragmentMapFromRuleContext != null) {
      this.fragmentMap = fragmentMapFromRuleContext.apply(ruleContext);
    }

    // This cannot be an ImmutableMap.Builder because that asserts when a key is duplicated
    TreeMap<String, String> makeVariables = new TreeMap<>();
    ImmutableMap.Builder<String, String> fragmentBuilder = ImmutableMap.builder();

    // If this toolchain type is enabled, we can derive make variables from it.  Otherwise, we
    // derive make variables from the corresponding configuration fragment.
    if (ruleContext.getConfiguration().getOptions().get(Options.class).makeVariableSource
            == MakeVariableSource.TOOLCHAIN
        && ruleContext
            .getFragment(PlatformConfiguration.class)
            .getEnabledToolchainTypes()
            .contains(ruleContext.getLabel())) {
      ResolvedToolchainProviders providers =
          (ResolvedToolchainProviders)
              ruleContext.getToolchainContext().getResolvedToolchainProviders();
      providers.getForToolchainType(ruleContext.getLabel()).addGlobalMakeVariables(fragmentBuilder);
    } else {
      Class<? extends BuildConfiguration.Fragment> fragmentClass =
          fragmentMap.get(ruleContext.getLabel());
      if (fragmentClass != null) {
        BuildConfiguration.Fragment fragment = ruleContext.getFragment(fragmentClass);
        fragment.addGlobalMakeVariables(fragmentBuilder);
      }
    }
    makeVariables.putAll(fragmentBuilder.build());

    ImmutableMap<String, String> hardcodedVariables =
        hardcodedVariableMap.get(ruleContext.getLabel());
    if (hardcodedVariables != null) {
      makeVariables.putAll(hardcodedVariables);
    }
    // This will eventually go to Skyframe using getAnalysisEnvironment.getSkyframeEnv() to figure
    // out the lookup rule -> toolchain rule mapping. For now, it only provides Make variables that
    // come from BuildConfiguration so no need to ask Skyframe.
    return new RuleConfiguredTargetBuilder(ruleContext)
        .addNativeDeclaredProvider(new TemplateVariableInfo(ImmutableMap.copyOf(makeVariables)))
        .addProvider(RunfilesProvider.simple(Runfiles.EMPTY))
        .build();
  }
}
