// Copyright 2018 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.RuleClass.Builder.STARLARK_BUILD_SETTING_DEFAULT_ATTR_NAME;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.devtools.build.lib.actions.ActionConflictException;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.BuildSettingProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.LabelLateBoundDefault;
import com.google.devtools.build.lib.packages.BuildSetting;
import com.google.devtools.build.lib.packages.RuleClass;
import java.util.function.Function;

/** Implements template for creating custom alias rules. */
public final class LateBoundAlias implements RuleConfiguredTargetFactory {

  private static final String ATTRIBUTE_NAME = ":alias";

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws ActionConflictException, InterruptedException {
    ConfiguredTarget actual = (ConfiguredTarget) ruleContext.getPrerequisite(ATTRIBUTE_NAME);
    if (actual == null) {
      return createEmptyConfiguredTarget(ruleContext);
    }

    if (!ruleContext.getRule().isBuildSetting()) {
      return AliasConfiguredTarget.create(ruleContext, actual, ruleContext.getVisibility());
    }

    // This makes label_setting and label_flag work with select().
    BuildSetting buildSetting = ruleContext.getRule().getRuleClassObject().getBuildSetting();
    Object defaultValue =
        ruleContext
            .attributes()
            .get(STARLARK_BUILD_SETTING_DEFAULT_ATTR_NAME, buildSetting.getType());
    return AliasConfiguredTarget.createWithOverrides(
        ruleContext,
        actual,
        ruleContext.getVisibility(),
        ImmutableClassToInstanceMap.of(
            BuildSettingProvider.class,
            new BuildSettingProvider(buildSetting, defaultValue, ruleContext.getLabel())));
  }

  private static ConfiguredTarget createEmptyConfiguredTarget(RuleContext ruleContext)
      throws ActionConflictException, InterruptedException {
    return new RuleConfiguredTargetBuilder(ruleContext)
        .addProvider(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY))
        .build();
  }

  /** Rule definition for custom alias rules. */
  public abstract static class CommonAliasRule<FragmentT extends Fragment>
      extends AbstractAliasRule {
    private static final String ATTRIBUTE_NAME = ":alias";

    private final Function<RuleDefinitionEnvironment, LabelLateBoundDefault<FragmentT>>
        labelResolver;
    private final Class<FragmentT> fragmentClass;

    protected CommonAliasRule(
        String ruleName,
        Function<RuleDefinitionEnvironment, LabelLateBoundDefault<FragmentT>> labelResolver,
        Class<FragmentT> fragmentClass) {
      super(ruleName);
      this.labelResolver = Preconditions.checkNotNull(labelResolver);
      this.fragmentClass = Preconditions.checkNotNull(fragmentClass);
    }

    protected Attribute.Builder<Label> makeAttribute(RuleDefinitionEnvironment environment) {
      return attr(ATTRIBUTE_NAME, LABEL).value(labelResolver.apply(environment));
    }

    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
      Attribute attribute = makeAttribute(environment).build();
      Preconditions.checkArgument(attribute.getName().equals(ATTRIBUTE_NAME));
      Preconditions.checkArgument(attribute.getType().equals(LABEL));

      return builder
          .requiresConfigurationFragments(fragmentClass)
          .removeAttribute("licenses")
          .removeAttribute("distribs")
          .advertiseProvider(AliasProvider.class)
          .addAttribute(attribute)
          .build();
    }
  }

  abstract static class AbstractAliasRule implements RuleDefinition {
    private final String ruleName;

    AbstractAliasRule(String ruleName) {
      this.ruleName = Preconditions.checkNotNull(ruleName);
    }

    @Override
    public final Metadata getMetadata() {
      return Metadata.builder()
          .name(ruleName)
          .ancestors(BaseRuleClasses.NativeBuildRule.class)
          .factoryClass(LateBoundAlias.class)
          .build();
    }
  }
}
