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

package com.google.devtools.build.lib.testutil;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.OUTPUT_LIST;
import static com.google.devtools.build.lib.syntax.Type.INTEGER;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TemplateVariableInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileTypeSet;
import java.lang.reflect.Method;
import java.util.Map;

/**
 * Helper class to provide a RuleClassProvider for tests.
 */
public class TestRuleClassProvider {
  private static ConfiguredRuleClassProvider ruleProvider = null;
  private static ConfiguredRuleClassProvider ruleProviderWithClearedSuffix = null;

  /**
   * Adds all the rule classes supported internally within the build tool to the given builder.
   */
  public static void addStandardRules(ConfiguredRuleClassProvider.Builder builder) {
    try {
      Class<?> providerClass = Class.forName(TestConstants.TEST_RULE_CLASS_PROVIDER);
      // The method setup in the rule class provider requires the tools repository to be set
      // beforehand.
      builder.setToolsRepository(TestConstants.TOOLS_REPOSITORY);
      Method setupMethod = providerClass.getMethod("setup",
          ConfiguredRuleClassProvider.Builder.class);
      setupMethod.invoke(null, builder);
    } catch (Exception e) {
      throw new IllegalStateException(e);
    }
  }

  private static ConfiguredRuleClassProvider createRuleClassProvider(boolean clearSuffix) {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    addStandardRules(builder);
    builder.addRuleDefinition(new TestingDummyRule());
    builder.addRuleDefinition(new MockToolchainRule());
    if (clearSuffix) {
      builder.clearWorkspaceFileSuffixForTesting();
    }
    return builder.build();
  }

  /** Return a rule class provider. */
  public static ConfiguredRuleClassProvider getRuleClassProvider(boolean clearSuffix) {
    if (clearSuffix) {
      if (ruleProviderWithClearedSuffix == null) {
        ruleProviderWithClearedSuffix = createRuleClassProvider(true);
      }
      return ruleProviderWithClearedSuffix;
    }
    if (ruleProvider == null) {
      ruleProvider = createRuleClassProvider(false);
    }
    return ruleProvider;
  }

  /** Return a rule class provider. */
  public static ConfiguredRuleClassProvider getRuleClassProvider() {
    return getRuleClassProvider(false);
  }

  /**
   * A dummy rule with some dummy attributes.
   */
  public static final class TestingDummyRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .setUndocumented()
          .add(attr("srcs", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE))
          .add(attr("outs", OUTPUT_LIST))
          .add(attr("dummystrings", STRING_LIST))
          .add(attr("dummyinteger", INTEGER))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("testing_dummy_rule")
          .ancestors(BaseRuleClasses.RuleBase.class)
          .factoryClass(UnknownRuleConfiguredTarget.class)
          .build();
    }
  }

  /**
   * Stub rule to test Make variable expansion.
   */
  public static final class MakeVariableTester implements RuleConfiguredTargetFactory {

    @Override
    public ConfiguredTarget create(RuleContext ruleContext)
        throws InterruptedException, RuleErrorException, ActionConflictException {
      Map<String, String> variables = ruleContext.attributes().get("variables", Type.STRING_DICT);
      return new RuleConfiguredTargetBuilder(ruleContext)
          .setFilesToBuild(NestedSetBuilder.emptySet(Order.STABLE_ORDER))
          .addProvider(RunfilesProvider.EMPTY)
          .addNativeDeclaredProvider(
              new TemplateVariableInfo(ImmutableMap.copyOf(variables), Location.BUILTIN))
          .build();
    }
  }

  /**
   * Definition of a stub rule to test Make variable expansion.
   */
  public static final class MakeVariableTesterRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          .advertiseProvider(TemplateVariableInfo.class)
          .add(attr("variables", Type.STRING_DICT))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return Metadata.builder()
          .name("make_variable_tester")
          .ancestors(
              BaseRuleClasses.BaseRule.class, BaseRuleClasses.MakeVariableExpandingRule.class)
          .factoryClass(MakeVariableTester.class)
          .build();
    }
  }

  /** A mock rule that requires a toolchain. */
  public static class MockToolchainRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .requiresConfigurationFragments(PlatformConfiguration.class)
          .addRequiredToolchains(
              ImmutableList.of(Label.parseAbsoluteUnchecked("//toolchain:test_toolchain")))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("mock_toolchain_rule")
          .factoryClass(UnknownRuleConfiguredTarget.class)
          .ancestors(BaseRuleClasses.RuleBase.class)
          .build();
    }
  }
}
