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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.collect.nestedset.Order.STABLE_ORDER;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_KEYED_STRING_DICT;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.SkylarkRuleContext;
import com.google.devtools.build.lib.skylark.util.SkylarkTestCase;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the config_feature_flag rule. */
@RunWith(JUnit4.class)
public final class ConfigFeatureFlagTest extends SkylarkTestCase {

  /** Rule introducing a transition to set feature flags for dependencies. */
  public static final class FeatureFlagSetter
      implements RuleDefinition, RuleConfiguredTargetFactory {

    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .requiresConfigurationFragments(ConfigFeatureFlagConfiguration.class)
          .cfg(new ConfigFeatureFlagTransitionFactory("flag_values"))
          .add(attr("deps", LABEL_LIST).allowedFileTypes())
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
          .ancestors(BaseRuleClasses.BaseRule.class)
          .factoryClass(FeatureFlagSetter.class)
          .build();
    }

    @Override
    public ConfiguredTarget create(RuleContext ruleContext)
        throws InterruptedException, RuleErrorException {
      TransitiveInfoCollection dep = ruleContext.getPrerequisite("exports_flag", Mode.TARGET);
      ConfigFeatureFlagProvider exportedProvider =
          dep != null ? ConfigFeatureFlagProvider.fromTarget(dep) : null;
      RuleConfiguredTargetBuilder builder =
          new RuleConfiguredTargetBuilder(ruleContext)
              .setFilesToBuild(NestedSetBuilder.<Artifact>emptySet(STABLE_ORDER))
              .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY);
      if (exportedProvider != null) {
        builder.addNativeDeclaredProvider(exportedProvider);
      }
      return builder.build();
    }
  }

  @Before
  public void useDynamicConfigurations() throws Exception {
    useConfiguration("--experimental_dynamic_configs=on");
  }

  @Override
  protected ConfiguredRuleClassProvider getRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder =
        new ConfiguredRuleClassProvider.Builder().addRuleDefinition(new FeatureFlagSetter());
    TestRuleClassProvider.addStandardRules(builder);
    return builder.build();
  }

  @Test
  public void configFeatureFlagProvider_fromTargetReturnsNullIfTargetDoesNotExportProvider()
      throws Exception {
    scratch.file(
        "test/BUILD",
        "feature_flag_setter(",
        "    name = 'top',",
        "    flag_values = {",
        "    },",
        ")");
    assertThat(ConfigFeatureFlagProvider.fromTarget(getConfiguredTarget("//test:top"))).isNull();
  }

  @Test
  public void configFeatureFlagProvider_containsValueFromConfiguration() throws Exception {
    scratch.file(
        "test/BUILD",
        "feature_flag_setter(",
        "    name = 'top',",
        "    exports_flag = ':flag',",
        "    flag_values = {",
        "        ':flag': 'configured',",
        "    },",
        ")",
        "config_feature_flag(",
        "    name = 'flag',",
        "    allowed_values = ['default', 'configured', 'other'],",
        "    default_value = 'default',",
        ")");
    assertThat(ConfigFeatureFlagProvider.fromTarget(getConfiguredTarget("//test:top")).getValue())
        .isEqualTo("configured");
  }

  @Test
  public void configFeatureFlagProvider_valueIsAccessibleFromSkylark() throws Exception {
    scratch.file(
        "test/wrapper.bzl",
        "def _flag_reading_wrapper_impl(ctx):",
        "  pass",
        "flag_reading_wrapper = rule(",
        "  implementation = _flag_reading_wrapper_impl,",
        "  attrs = {'flag': attr.label()},",
        ")");
    scratch.file(
        "test/BUILD",
        "load(':wrapper.bzl', 'flag_reading_wrapper')",
        "feature_flag_setter(",
        "    name = 'top',",
        "    deps = [':wrapper'],",
        "    flag_values = {",
        "        ':flag': 'configured',",
        "    },",
        ")",
        "flag_reading_wrapper(",
        "    name = 'wrapper',",
        "    flag = ':flag',",
        ")",
        "config_feature_flag(",
        "    name = 'flag',",
        "    allowed_values = ['default', 'configured', 'other'],",
        "    default_value = 'default',",
        ")");
    ConfiguredTarget top = getConfiguredTarget("//test:top");
    ConfiguredTarget wrapper =
        (ConfiguredTarget) Iterables.getOnlyElement(getPrerequisites(top, "deps"));
    SkylarkRuleContext ctx = new SkylarkRuleContext(getRuleContextForSkylark(wrapper), null);
    update("ruleContext", ctx);
    update("config_common", new ConfigSkylarkCommon());
    String value = (String) eval("ruleContext.attr.flag[config_common.FeatureFlagInfo].value");
    assertThat(value).isEqualTo("configured");
  }

  @Test
  public void configFeatureFlagProvider_validatesValuesUsingAllowedValuesAttribute()
      throws Exception {
    scratch.file(
        "test/BUILD",
        "config_feature_flag(",
        "    name = 'flag',",
        "    allowed_values = ['default', 'configured', 'other'],",
        "    default_value = 'default',",
        ")");
    ConfigFeatureFlagProvider provider =
        ConfigFeatureFlagProvider.fromTarget(getConfiguredTarget("//test:flag"));
    assertThat(provider.isValidValue("default")).isTrue();
    assertThat(provider.isValidValue("configured")).isTrue();
    assertThat(provider.isValidValue("other")).isTrue();

    assertThat(provider.isValidValue("absent")).isFalse();
    assertThat(provider.isValidValue("conFigured")).isFalse();
    assertThat(provider.isValidValue("  other")).isFalse();
  }

  @Test
  public void configFeatureFlagProvider_valueValidationIsPossibleFromSkylark() throws Exception {
    scratch.file(
        "test/wrapper.bzl",
        "def _flag_reading_wrapper_impl(ctx):",
        "  pass",
        "flag_reading_wrapper = rule(",
        "  implementation = _flag_reading_wrapper_impl,",
        "  attrs = {'flag': attr.label()},",
        ")");
    scratch.file(
        "test/BUILD",
        "load(':wrapper.bzl', 'flag_reading_wrapper')",
        "flag_reading_wrapper(",
        "    name = 'wrapper',",
        "    flag = ':flag',",
        ")",
        "config_feature_flag(",
        "    name = 'flag',",
        "    allowed_values = ['default', 'configured', 'other'],",
        "    default_value = 'default',",
        ")");
    SkylarkRuleContext ctx = createRuleContext("//test:wrapper");
    update("ruleContext", ctx);
    update("config_common", new ConfigSkylarkCommon());
    String provider = "ruleContext.attr.flag[config_common.FeatureFlagInfo]";
    Boolean isDefaultValid = (Boolean) eval(provider + ".is_valid_value('default')");
    Boolean isConfiguredValid = (Boolean) eval(provider + ".is_valid_value('configured')");
    Boolean isOtherValid = (Boolean) eval(provider + ".is_valid_value('other')");
    Boolean isAbsentValid = (Boolean) eval(provider + ".is_valid_value('absent')");
    Boolean isIncorrectCapitalizationValid =
        (Boolean) eval(provider + ".is_valid_value('conFigured')");
    Boolean isIncorrectSpacingValid = (Boolean) eval(provider + ".is_valid_value('  other')");

    assertThat(isDefaultValid).isTrue();
    assertThat(isConfiguredValid).isTrue();
    assertThat(isOtherValid).isTrue();

    assertThat(isAbsentValid).isFalse();
    assertThat(isIncorrectCapitalizationValid).isFalse();
    assertThat(isIncorrectSpacingValid).isFalse();
  }

  @Test
  public void configFeatureFlagProvider_usesDefaultValueIfConfigurationDoesntSetValue()
      throws Exception {
    scratch.file(
        "test/BUILD",
        "feature_flag_setter(",
        "    name = 'top',",
        "    exports_flag = ':flag',",
        "    flag_values = {",
        "        ':other': 'configured',",
        "    },",
        ")",
        "config_feature_flag(",
        "    name = 'flag',",
        "    allowed_values = ['other', 'default', 'configured'],",
        "    default_value = 'default',",
        ")",
        "config_feature_flag(",
        "    name = 'other',",
        "    allowed_values = ['default', 'configured', 'other'],",
        "    default_value = 'default',",
        ")");
    assertThat(ConfigFeatureFlagProvider.fromTarget(getConfiguredTarget("//test:top")).getValue())
        .isEqualTo("default");
  }

  @Test
  public void allowedValuesAttribute_cannotBeEmpty() throws Exception {
    reporter.removeHandler(failFastHandler); // expecting an error
    scratch.file(
        "test/BUILD",
        "config_feature_flag(",
        "    name = 'flag',",
        "    allowed_values = [],",
        "    default_value = 'default',",
        ")");
    assertThat(getConfiguredTarget("//test:flag")).isNull();
    assertContainsEvent(
        "in allowed_values attribute of config_feature_flag rule //test:flag: "
            + "attribute must be non empty");
  }

  @Test
  public void allowedValuesAttribute_cannotContainDuplicates() throws Exception {
    reporter.removeHandler(failFastHandler); // expecting an error
    scratch.file(
        "test/BUILD",
        "config_feature_flag(",
        "    name = 'flag',",
        "    allowed_values = ['double', 'double', 'toil', 'trouble'],",
        "    default_value = 'trouble',",
        ")");
    assertThat(getConfiguredTarget("//test:flag")).isNull();
    assertContainsEvent(
        "in allowed_values attribute of config_feature_flag rule //test:flag: "
            + "cannot contain duplicates, but contained multiple of ['double']");
  }

  @Test
  public void defaultValueAttribute_mustBeMemberOfAllowedValues() throws Exception {
    reporter.removeHandler(failFastHandler); // expecting an error
    scratch.file(
        "test/BUILD",
        "feature_flag_setter(",
        "    name = 'top',",
        "    exports_flag = ':flag',",
        "    flag_values = {",
        "        ':flag': 'legal',",
        "    },",
        ")",
        "config_feature_flag(",
        "    name = 'flag',",
        "    allowed_values = ['legal', 'eagle'],",
        "    default_value = 'beagle',",
        ")");
    assertThat(getConfiguredTarget("//test:top")).isNull();
    assertContainsEvent(
        "in default_value attribute of config_feature_flag rule //test:flag: "
            + "must be one of ['eagle', 'legal'], but was 'beagle'");
  }

  @Test
  public void configurationValue_mustBeMemberOfAllowedValues() throws Exception {
    reporter.removeHandler(failFastHandler); // expecting an error
    scratch.file(
        "test/BUILD",
        "feature_flag_setter(",
        "    name = 'top',",
        "    exports_flag = ':flag',",
        "    flag_values = {",
        "        ':flag': 'invalid',",
        "    },",
        ")",
        "config_feature_flag(",
        "    name = 'flag',",
        "    allowed_values = ['default', 'configured', 'other'],",
        "    default_value = 'default',",
        ")");
    assertThat(getConfiguredTarget("//test:top")).isNull();
    // TODO(mstaib): when configurationError is implemented, switch to testing for that
    assertContainsEvent(
        "in config_feature_flag rule //test:flag: "
            + "value must be one of ['configured', 'default', 'other'], but was 'invalid'");
  }
}
