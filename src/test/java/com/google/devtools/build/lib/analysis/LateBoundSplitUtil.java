// Copyright 2016 The Bazel Authors. All rights reserved.
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


import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.RequiresOptions;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;

/**
 * Rule and configuration class definitions for testing late-bound split attributes.
 */
public class LateBoundSplitUtil {
  /** A custom {@link FragmentOptions} with the option to be split. */
  public static class TestOptions extends FragmentOptions { // public for options loader
    @Option(
      name = "foo",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = ""
    )
    public String fooFlag;
  }

  /** The {@link Fragment} that contains the options. */
  @AutoCodec
  @RequiresOptions(options = {TestOptions.class})
  public static class TestFragment extends Fragment {
    private final BuildOptions buildOptions;

    public TestFragment(BuildOptions buildOptions) {
      this.buildOptions = buildOptions;
    }
    // Getter required to satisfy AutoCodec.
    public BuildOptions getBuildOptions() {
      return buildOptions;
    }
  }

  /**
   * The fragment's loader.
   */
  static class FragmentLoader implements ConfigurationFragmentFactory {
    @Override
    public Class<? extends Fragment> creates() {
     return TestFragment.class;
    }
  }

  /**
   * A custom rule that requires {@link TestFragment}.
   */
  static final RuleDefinition RULE_WITH_TEST_FRAGMENT = (MockRule) () -> MockRule.define(
      "rule_with_test_fragment",
      (builder, env) -> builder.requiresConfigurationFragments(TestFragment.class));

  /**
   * Returns a rule class provider with standard test setup plus the above rules/configs.
   */
  static ConfiguredRuleClassProvider getRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addRuleDefinition(RULE_WITH_TEST_FRAGMENT);
    builder.addConfigurationFragment(new FragmentLoader());
    builder.addConfigurationOptions(TestOptions.class);
    return builder.build();
  }

  /**
   * Returns the {@link TestOptions} from the given configuration.
   */
  static TestOptions getOptions(BuildConfiguration config) {
    return config.getOptions().get(TestOptions.class);
  }
}
