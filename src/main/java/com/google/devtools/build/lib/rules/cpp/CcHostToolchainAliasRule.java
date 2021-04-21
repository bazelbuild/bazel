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

package com.google.devtools.build.lib.rules.cpp;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;

import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.MiddlemanProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TemplateVariableInfo;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.packages.RuleClass;

/** Implementation of the {@code cc_toolchain_alias} rule. */
public class CcHostToolchainAliasRule implements RuleDefinition {

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(CppConfiguration.class)
        .removeAttribute("licenses")
        .removeAttribute("distribs")
        .add(
            attr("$cc_toolchain_alias", LABEL)
                .cfg(HostTransition.createFactory())
                .value(env.getToolsLabel("//tools/cpp:current_cc_toolchain")))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return Metadata.builder()
        .name("cc_host_toolchain_alias")
        .ancestors(BaseRuleClasses.NativeBuildRule.class)
        .factoryClass(CcHostToolchainAlias.class)
        .build();
  }

  /** Implementation of cc_host_toolchain_alias. */
  public static class CcHostToolchainAlias implements RuleConfiguredTargetFactory {
    @Override
    public ConfiguredTarget create(RuleContext ruleContext)
        throws InterruptedException, RuleErrorException, ActionConflictException {

      TransitiveInfoCollection ccToolchainAlias =
          ruleContext.getPrerequisite("$cc_toolchain_alias");
      CcToolchainProvider ccToolchainProvider = ccToolchainAlias.get(CcToolchainProvider.PROVIDER);
      TemplateVariableInfo templateVariableInfo =
          ccToolchainAlias.get(TemplateVariableInfo.PROVIDER);
      ToolchainInfo toolchain = ccToolchainAlias.get(ToolchainInfo.PROVIDER);

      return new RuleConfiguredTargetBuilder(ruleContext)
          .addProvider(RunfilesProvider.simple(Runfiles.EMPTY))
          .addNativeDeclaredProvider(ccToolchainProvider)
          .addNativeDeclaredProvider(toolchain)
          .addNativeDeclaredProvider(templateVariableInfo)
          .setFilesToBuild(ccToolchainProvider.getAllFiles())
          .addProvider(new MiddlemanProvider(ccToolchainProvider.getAllFilesMiddleman()))
          .build();
    }
  }
}
