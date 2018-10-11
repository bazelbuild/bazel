// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;

/**
 * Implementation of the {@code cc_toolchain_suite} rule.
 *
 * <p>This is currently a no-op because the logic that transforms this rule into something that can
 * be understood by the {@code cc_*} rules is in
 * {@link com.google.devtools.build.lib.rules.cpp.CppConfigurationLoader}.
 */
public class CcToolchainSuite implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    NestedSetBuilder<Artifact> filesToBuild = NestedSetBuilder.stableOrder();
    for (TransitiveInfoCollection dep : ruleContext.getPrerequisiteMap("toolchains").values()) {
      CcToolchainProvider provider = (CcToolchainProvider) dep.get(ToolchainInfo.PROVIDER);
      if (provider != null) {
        filesToBuild.addTransitive(provider.getCrosstool());
      }
    }

    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(filesToBuild.build())
        .add(RunfilesProvider.class, RunfilesProvider.EMPTY)
        .build();
  }
}
