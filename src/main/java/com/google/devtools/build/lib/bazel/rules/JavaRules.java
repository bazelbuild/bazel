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
package com.google.devtools.build.lib.bazel.rules;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.RuleSet;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaBinaryRule;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaBuildInfoFactory;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaImportRule;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaLibraryRule;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaPluginRule;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaRuleClasses;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaSemantics;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaTestRule;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaToolchain;
import com.google.devtools.build.lib.rules.core.CoreRules;
import com.google.devtools.build.lib.rules.extra.ActionListenerRule;
import com.google.devtools.build.lib.rules.extra.ExtraActionRule;
import com.google.devtools.build.lib.rules.java.JavaCcLinkParamsProvider;
import com.google.devtools.build.lib.rules.java.JavaConfigurationLoader;
import com.google.devtools.build.lib.rules.java.JavaImportBaseRule;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaOptions;
import com.google.devtools.build.lib.rules.java.JavaPackageConfigurationRule;
import com.google.devtools.build.lib.rules.java.JavaRuleClasses.IjarBaseRule;
import com.google.devtools.build.lib.rules.java.JavaRuleClasses.JavaHostRuntimeBaseRule;
import com.google.devtools.build.lib.rules.java.JavaRuleClasses.JavaRuntimeBaseRule;
import com.google.devtools.build.lib.rules.java.JavaRuleClasses.JavaToolchainBaseRule;
import com.google.devtools.build.lib.rules.java.JavaRuntimeAliasRule;
import com.google.devtools.build.lib.rules.java.JavaRuntimeRule;
import com.google.devtools.build.lib.rules.java.JavaSkylarkCommon;
import com.google.devtools.build.lib.rules.java.JavaToolchainAliasRule;
import com.google.devtools.build.lib.rules.java.JavaToolchainRule;
import com.google.devtools.build.lib.rules.java.ProguardLibraryRule;
import com.google.devtools.build.lib.rules.java.proto.JavaProtoSkylarkCommon;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaBootstrap;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import java.io.IOException;

/**
 * Rules for Java support in Bazel.
 */
public class JavaRules implements RuleSet {
  public static final JavaRules INSTANCE = new JavaRules();

  private JavaRules() {
    // Use the static INSTANCE field instead.
  }

  @Override
  public void init(ConfiguredRuleClassProvider.Builder builder) {
    builder.addConfigurationOptions(JavaOptions.class);
    builder.addConfigurationFragment(new JavaConfigurationLoader());
    // TestConfiguration is only required by the java rules to know when the persistent test runner
    // is enabled.
    builder.addConfigurationFragment(new TestConfiguration.Loader());

    builder.addBuildInfoFactory(new BazelJavaBuildInfoFactory());

    builder.addRuleDefinition(new BazelJavaRuleClasses.BaseJavaBinaryRule());
    builder.addRuleDefinition(new IjarBaseRule());
    builder.addRuleDefinition(new JavaToolchainBaseRule());
    builder.addRuleDefinition(new JavaRuntimeBaseRule());
    builder.addRuleDefinition(new JavaHostRuntimeBaseRule());
    builder.addRuleDefinition(new BazelJavaRuleClasses.JavaBaseRule());
    builder.addRuleDefinition(new ProguardLibraryRule());
    builder.addRuleDefinition(new JavaImportBaseRule());
    builder.addRuleDefinition(new BazelJavaRuleClasses.JavaRule());
    builder.addRuleDefinition(new BazelJavaBinaryRule());
    builder.addRuleDefinition(new BazelJavaLibraryRule());
    builder.addRuleDefinition(new BazelJavaImportRule());
    builder.addRuleDefinition(new BazelJavaTestRule());
    builder.addRuleDefinition(new BazelJavaPluginRule());
    builder.addRuleDefinition(JavaToolchainRule.create(BazelJavaToolchain.class));
    builder.addRuleDefinition(new JavaPackageConfigurationRule());
    builder.addRuleDefinition(new JavaRuntimeRule());
    builder.addRuleDefinition(new JavaRuntimeAliasRule());
    builder.addRuleDefinition(new JavaToolchainAliasRule());

    builder.addRuleDefinition(new ExtraActionRule());
    builder.addRuleDefinition(new ActionListenerRule());

    builder.addSkylarkBootstrap(
        new JavaBootstrap(
            new JavaSkylarkCommon(BazelJavaSemantics.INSTANCE),
            JavaInfo.PROVIDER,
            new JavaProtoSkylarkCommon(),
            JavaCcLinkParamsProvider.PROVIDER));

    try {
      builder.addWorkspaceFileSuffix(
          ResourceFileLoader.loadResource(BazelJavaRuleClasses.class, "jdk.WORKSPACE"));
      builder.addWorkspaceFileSuffix(
          ResourceFileLoader.loadResource(JavaRules.class, "coverage.WORKSPACE"));
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }

  @Override
  public ImmutableList<RuleSet> requires() {
    return ImmutableList.of(CoreRules.INSTANCE, CcRules.INSTANCE);
  }
}
