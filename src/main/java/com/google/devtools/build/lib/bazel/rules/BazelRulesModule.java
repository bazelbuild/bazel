// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCppRuleClasses;
import com.google.devtools.build.lib.bazel.rules.sh.BazelShRuleClasses;
import com.google.devtools.build.lib.rules.cpp.FdoSupportFunction;
import com.google.devtools.build.lib.rules.cpp.FdoSupportValue;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.WorkspaceBuilder;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;
import java.io.IOException;

/**
 * Module implementing the rule set of Bazel.
 */
public class BazelRulesModule extends BlazeModule {
  /** This is where deprecated options go to die. */
  public static class GraveyardOptions extends OptionsBase {
    @Deprecated
    @Option(
        name = "direct_run",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.UNKNOWN},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated no-op.")
    public boolean directRun;

    @Deprecated
    @Option(
        name = "glibc",
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.UNKNOWN},
        metadataTags = {OptionMetadataTag.DEPRECATED},
        help = "Deprecated no-op.")
    public String glibc;
  }

  @Override
  public void initializeRuleClasses(ConfiguredRuleClassProvider.Builder builder) {
    builder.setToolsRepository(BazelRuleClassProvider.TOOLS_REPOSITORY);
    BazelRuleClassProvider.setup(builder);
    try {
      // Load auto-configuration files, it is made outside of the rule class provider so that it
      // will not be loaded for our Java tests.
      builder.addWorkspaceFileSuffix(
          ResourceFileLoader.loadResource(BazelCppRuleClasses.class, "cc_configure.WORKSPACE"));
      builder.addWorkspaceFileSuffix(
          ResourceFileLoader.loadResource(BazelRulesModule.class, "xcode_configure.WORKSPACE"));
      builder.addWorkspaceFileSuffix(
          ResourceFileLoader.loadResource(BazelShRuleClasses.class, "sh_configure.WORKSPACE"));
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }

  @Override
  public void workspaceInit(
      BlazeRuntime runtime, BlazeDirectories directories, WorkspaceBuilder builder) {
    builder.addSkyFunction(FdoSupportValue.SKYFUNCTION, new FdoSupportFunction(directories));
  }

  @Override
  public BuildOptions getDefaultBuildOptions(BlazeRuntime blazeRuntime) {
    return DefaultBuildOptionsForDiffing.getDefaultBuildOptionsForFragments(
        blazeRuntime.getRuleClassProvider().getConfigurationOptions());
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return "build".equals(command.name())
        ? ImmutableList.of(GraveyardOptions.class) : ImmutableList.of();
  }
}
