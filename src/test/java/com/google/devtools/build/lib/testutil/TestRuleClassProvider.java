// Copyright 2014 Google Inc. All rights reserved.
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
import static com.google.devtools.build.lib.packages.Type.INTEGER;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.OUTPUT_LIST;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;

import com.google.devtools.build.lib.bazel.rules.BazelRuleClassProvider;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.view.BaseRuleClasses;
import com.google.devtools.build.lib.view.BlazeRule;
import com.google.devtools.build.lib.view.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.view.RuleDefinition;
import com.google.devtools.build.lib.view.RuleDefinitionEnvironment;

/**
 * Helper class to provide Google3RuleClassProvider for tests. This will statically add rule classes
 * that otherwise are loaded as plugins.
 */
public class TestRuleClassProvider {
  private static ConfiguredRuleClassProvider ruleProvider = null;

  /**
   * Return a Google3RuleClassProvider.
   */
  public static ConfiguredRuleClassProvider getRuleClassProvider() {
    if (ruleProvider == null) {
      ConfiguredRuleClassProvider.Builder builder =
          new ConfiguredRuleClassProvider.Builder();
      BazelRuleClassProvider.setup(builder);
      builder.addRuleDefinition(TestingDummyRule.class);
      ruleProvider = builder.build();
    }
    return ruleProvider;
  }

  @BlazeRule(name = "testing_dummy_rule",
               ancestors = { BaseRuleClasses.RuleBase.class },
               // Instantiated only in tests
               factoryClass = UnknownRuleConfiguredTarget.class)
  public static final class TestingDummyRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .setUndocumented()
          .add(attr("srcs", LABEL_LIST))
          .add(attr("outs", OUTPUT_LIST))
          .add(attr("dummystrings", STRING_LIST))
          .add(attr("dummyinteger", INTEGER))
          .build();
    }
  }
}
