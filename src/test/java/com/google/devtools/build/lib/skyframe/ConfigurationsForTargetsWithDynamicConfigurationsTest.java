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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.util.TestAspects;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Runs an expanded set of ConfigurationsForTargetsTest with trimmed dynamic configurations. */
@TestSpec(size = Suite.SMALL_TESTS)
@RunWith(JUnit4.class)
public class ConfigurationsForTargetsWithDynamicConfigurationsTest
    extends ConfigurationsForTargetsTest {
  @Override
  protected FlagBuilder defaultFlags() {
    return super.defaultFlags().with(Flag.DYNAMIC_CONFIGURATIONS);
  }

  @Test
  public void testRuleClassTransition() throws Exception {
    setRulesAvailableInTests(new TestAspects.BaseRule(),
        new TestAspects.AttributeTransitionRule(),
        new TestAspects.RuleClassTransitionRule());
    scratch.file("a/BUILD",
        "attribute_transition(",
        "   name='attribute',",
        "   without_transition = ':rule_class',",
        ")",
        "rule_class_transition(name='rule_class')");
    List<ConfiguredTarget> deps = getConfiguredDeps("//a:attribute", "without_transition");
    BuildConfiguration ruleclass = Iterables.getOnlyElement(deps).getConfiguration();
    assertThat(ruleclass.getCpu()).isEqualTo("SET BY PATCH");
  }

  @Test
  public void testNonConflictingAttributeAndRuleClassTransitions() throws Exception {
    setRulesAvailableInTests(new TestAspects.BaseRule(),
        new TestAspects.AttributeTransitionRule(),
        new TestAspects.RuleClassTransitionRule());
    scratch.file("a/BUILD",
        "attribute_transition(",
        "   name='attribute',",
        "   with_host_cpu_transition = ':rule_class',",
        ")",
        "rule_class_transition(name='rule_class')");
    List<ConfiguredTarget> deps = getConfiguredDeps("//a:attribute", "with_host_cpu_transition");
    BuildConfiguration ruleclass = Iterables.getOnlyElement(deps).getConfiguration();
    assertThat(ruleclass.getCpu()).isEqualTo("SET BY PATCH");
    assertThat(ruleclass.getHostCpu()).isEqualTo("SET BY SPLIT");
  }

  @Test
  public void testConflictingAttributeAndRuleClassTransitions() throws Exception {
    setRulesAvailableInTests(new TestAspects.BaseRule(),
        new TestAspects.AttributeTransitionRule(),
        new TestAspects.RuleClassTransitionRule());
    scratch.file("a/BUILD",
        "attribute_transition(",
        "   name='attribute',",
        "   with_cpu_transition = ':rule_class',",
        ")",
        "rule_class_transition(name='rule_class')");
    List<ConfiguredTarget> deps = getConfiguredDeps("//a:attribute", "with_cpu_transition");
    BuildConfiguration ruleclass = Iterables.getOnlyElement(deps).getConfiguration();
    assertThat(ruleclass.getCpu()).isEqualTo("SET BY SPLIT");
  }
}
