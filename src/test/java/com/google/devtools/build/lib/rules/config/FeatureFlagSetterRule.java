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
// limitations under the License

package com.google.devtools.build.lib.rules.config;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_KEYED_STRING_DICT;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.PrerequisiteArtifacts;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.RuleClass;

/** Rule introducing a transition to set feature flags for itself and dependencies. */
public final class FeatureFlagSetterRule implements RuleDefinition, RuleConfiguredTargetFactory {

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(ConfigFeatureFlagConfiguration.class)
        .cfg(new ConfigFeatureFlagTransitionFactory("flag_values"))
        .add(attr("deps", LABEL_LIST).allowedFileTypes())
        .add(attr("exports_setting", LABEL).allowedRuleClasses("config_setting").allowedFileTypes())
        .add(
            attr("exports_flag", LABEL)
                .allowedRuleClasses("config_feature_flag")
                .allowedFileTypes())
        .add(
            attr("flag_values", LABEL_KEYED_STRING_DICT)
                .allowedRuleClasses("config_feature_flag")
                .allowedFileTypes()
                .nonconfigurable("used in RuleTransitionFactory")
                .value(ImmutableMap.<Label, String>of()))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("feature_flag_setter")
        .ancestors(BaseRuleClasses.NativeBuildRule.class)
        .factoryClass(FeatureFlagSetterRule.class)
        .build();
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    TransitiveInfoCollection exportedFlag = ruleContext.getPrerequisite("exports_flag");
    ConfigFeatureFlagProvider exportedFlagProvider =
        exportedFlag != null ? ConfigFeatureFlagProvider.fromTarget(exportedFlag) : null;

    TransitiveInfoCollection exportedSetting = ruleContext.getPrerequisite("exports_setting");
    ConfigMatchingProvider exportedSettingProvider =
        exportedSetting != null ? exportedSetting.getProvider(ConfigMatchingProvider.class) : null;

    RuleConfiguredTargetBuilder builder =
        new RuleConfiguredTargetBuilder(ruleContext)
            .setFilesToBuild(PrerequisiteArtifacts.nestedSet(ruleContext, "deps"))
            .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY);
    if (exportedFlagProvider != null) {
      builder.addNativeDeclaredProvider(exportedFlagProvider);
    }
    if (exportedSettingProvider != null) {
      builder.addProvider(ConfigMatchingProvider.class, exportedSettingProvider);
    }
    return builder.build();
  }
}
