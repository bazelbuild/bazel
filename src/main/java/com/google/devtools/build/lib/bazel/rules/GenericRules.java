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
import com.google.devtools.build.lib.analysis.constraints.EnvironmentRule;
import com.google.devtools.build.lib.bazel.rules.common.BazelFilegroupRule;
import com.google.devtools.build.lib.rules.Alias.AliasRule;
import com.google.devtools.build.lib.rules.core.CoreRules;
import com.google.devtools.build.lib.rules.genquery.GenQueryRule;
import com.google.devtools.build.lib.rules.test.TestSuiteRule;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import java.io.IOException;

/**
 * A set of generic rules that provide miscellaneous capabilities to Bazel.
 */
public class GenericRules implements RuleSet {
  public static final GenericRules INSTANCE = new GenericRules();

  private GenericRules() {
    // Use the static INSTANCE field instead.
  }

  @Override
  public void init(ConfiguredRuleClassProvider.Builder builder) {
    builder.addRuleDefinition(new EnvironmentRule());

    builder.addRuleDefinition(new AliasRule());
    builder.addRuleDefinition(new BazelFilegroupRule());
    builder.addRuleDefinition(new TestSuiteRule());
    builder.addRuleDefinition(new GenQueryRule());

    try {
      builder.addWorkspaceFilePrefix(
          ResourceFileLoader.loadResource(BazelRuleClassProvider.class, "tools.WORKSPACE")
              // Hackily select the java_toolchain based on the host JDK version. JDK 8 and
              // 9 host_javabases require different toolchains, e.g. to use --patch-module
              // instead of -Xbootclasspath/p:.
              .replace(
                  "%java_toolchain%",
                  isJdk8OrEarlier()
                      ? "@bazel_tools//tools/jdk:toolchain_hostjdk8"
                      : "@bazel_tools//tools/jdk:toolchain_hostjdk9"));

    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }

  @Override
  public ImmutableList<RuleSet> requires() {
    return ImmutableList.of(CoreRules.INSTANCE);
  }

  private static boolean isJdk8OrEarlier() {
    return Double.parseDouble(System.getProperty("java.class.version")) <= 52.0;
  }
}
