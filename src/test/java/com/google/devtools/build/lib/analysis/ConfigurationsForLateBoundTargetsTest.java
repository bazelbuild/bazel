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

package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.PatchTransition;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.testutil.TestSpec;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests <target, sourceConfig> -> <dep, depConfig> relationships over latebound attributes.
 *
 * <p>Ideally these tests would be in
 * {@link com.google.devtools.build.lib.skyframe.ConfigurationsForTargetsTest}. But that's a
 * Skyframe test (ConfiguredTargetFunction is a Skyframe function). And the Skyframe library doesn't
 * know anything about latebound attributes. So we need to place these properly under the analysis
 * package.
 */
@TestSpec(size = Suite.SMALL_TESTS)
@RunWith(JUnit4.class)
public class ConfigurationsForLateBoundTargetsTest extends AnalysisTestCase {
  private static final PatchTransition CHANGE_FOO_FLAG_TRANSITION = new PatchTransition() {
    @Override
    public BuildOptions apply(BuildOptions options) {
      BuildOptions toOptions = options.clone();
      toOptions.get(LateBoundSplitUtil.TestOptions.class).fooFlag = "PATCHED!";
      return toOptions;
    }
  };

  /**
   * Mock late-bound attribute resolver that returns a fixed label.
   */
  private static final Attribute.LateBoundLabel<BuildConfiguration> LATEBOUND_VALUE_RESOLVER =
      new Attribute.LateBoundLabel<BuildConfiguration>() {
        @Override
        public Label resolve(Rule rule, AttributeMap attributes, BuildConfiguration config) {
          return Label.parseAbsoluteUnchecked("//foo:latebound_dep");
        }
      };

  /**
   * Rule definition with a latebound dependency.
   */
  private static final RuleDefinition LATE_BOUND_DEP_RULE = (MockRule) () -> MockRule.define(
      "rule_with_latebound_attr",
      (builder, env) -> {
        builder
            .add(
                attr(":latebound_attr", LABEL)
                    .value(LATEBOUND_VALUE_RESOLVER)
                    .cfg(CHANGE_FOO_FLAG_TRANSITION))
            .requiresConfigurationFragments(LateBoundSplitUtil.TestFragment.class);
      });

  @Before
  public void setupCustomLateBoundRules() throws Exception {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addRuleDefinition(LateBoundSplitUtil.RULE_WITH_LATEBOUND_SPLIT_ATTR);
    builder.addRuleDefinition(LateBoundSplitUtil.RULE_WITH_TEST_FRAGMENT);
    builder.addConfigurationFragment(new LateBoundSplitUtil.FragmentLoader());
    builder.addConfigurationOptions(LateBoundSplitUtil.TestOptions.class);
    builder.addRuleDefinition(LATE_BOUND_DEP_RULE);
    useRuleClassProvider(builder.build());
  }

  @Test
  public void lateBoundAttributeInTargetConfiguration() throws Exception {
    scratch.file("foo/BUILD",
        "rule_with_latebound_attr(",
        "    name = 'foo')",
        "rule_with_test_fragment(",
        "    name = 'latebound_dep')");
    update("//foo:foo");
    assertThat(getConfiguredTarget("//foo:foo", getTargetConfiguration())).isNotNull();
    ConfiguredTarget dep = Iterables.getOnlyElement(
        SkyframeExecutorTestUtils.getExistingConfiguredTargets(
            skyframeExecutor, Label.parseAbsolute("//foo:latebound_dep")));
    assertThat(dep.getConfiguration()).isNotEqualTo(getTargetConfiguration());
    assertThat(LateBoundSplitUtil.getOptions(dep.getConfiguration()).fooFlag).isEqualTo("PATCHED!");
  }

  @Test
  public void lateBoundAttributeInHostConfiguration() throws Exception {
    scratch.file("foo/BUILD",
        "genrule(",
        "    name = 'gen',",
        "    srcs = [],",
        "    outs = ['gen.out'],",
        "    cmd = 'echo hi > $@',",
        "    tools = [':foo'])",
        "rule_with_latebound_attr(",
        "    name = 'foo')",
        "rule_with_test_fragment(",
        "    name = 'latebound_dep')");
    update("//foo:gen");
    assertThat(getConfiguredTarget("//foo:foo", getHostConfiguration())).isNotNull();
    ConfiguredTarget dep = Iterables.getOnlyElement(
        SkyframeExecutorTestUtils.getExistingConfiguredTargets(
            skyframeExecutor, Label.parseAbsolute("//foo:latebound_dep")));
    assertThat(dep.getConfiguration()).isEqualTo(getHostConfiguration());
    // This is technically redundant, but slightly stronger in sanity checking that the host
    // configuration doesn't happen to match what the patch would have done.
    assertThat(LateBoundSplitUtil.getOptions(dep.getConfiguration()).fooFlag).isEmpty();
  }

  @Test
  public void lateBoundSplitAttributeInTargetConfiguration() throws Exception {
    scratch.file("foo/BUILD",
        "rule_with_latebound_split(",
        "    name = 'foo')",
        "rule_with_test_fragment(",
        "    name = 'latebound_dep')");
    update("//foo:foo");
    assertThat(getConfiguredTarget("//foo:foo")).isNotNull();
    Iterable<ConfiguredTarget> deps = SkyframeExecutorTestUtils.getExistingConfiguredTargets(
        skyframeExecutor, Label.parseAbsolute("//foo:latebound_dep"));
    assertThat(deps).hasSize(2);
    assertThat(
        ImmutableList.of(
            LateBoundSplitUtil.getOptions(Iterables.get(deps, 0).getConfiguration()).fooFlag,
            LateBoundSplitUtil.getOptions(Iterables.get(deps, 1).getConfiguration()).fooFlag))
        .containsExactly("one", "two");
  }

  @Test
  public void lateBoundSplitAttributeInHostConfiguration() throws Exception {
    scratch.file("foo/BUILD",
        "genrule(",
        "    name = 'gen',",
        "    srcs = [],",
        "    outs = ['gen.out'],",
        "    cmd = 'echo hi > $@',",
        "    tools = [':foo'])",
        "rule_with_latebound_split(",
        "    name = 'foo')",
        "rule_with_test_fragment(",
        "    name = 'latebound_dep')");
    update("//foo:gen");
    assertThat(getConfiguredTarget("//foo:foo", getHostConfiguration())).isNotNull();
    ConfiguredTarget dep = Iterables.getOnlyElement(
        SkyframeExecutorTestUtils.getExistingConfiguredTargets(
            skyframeExecutor, Label.parseAbsolute("//foo:latebound_dep")));
    assertThat(dep.getConfiguration()).isEqualTo(getHostConfiguration());
    // This is technically redundant, but slightly stronger in sanity checking that the host
    // configuration doesn't happen to match what the split would have done.
    assertThat(LateBoundSplitUtil.getOptions(dep.getConfiguration()).fooFlag).isEmpty();
  }
}
