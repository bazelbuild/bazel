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

package com.google.devtools.build.lib.analysis.test;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.RunsPerTestConverter;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.PerLabelOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.TriState;
import java.util.List;

/** Test-related options. */
public class TestConfiguration extends Fragment {

  /** Command-line options. */
  public static class TestOptions extends FragmentOptions {
    @Option(
      name = "test_filter",
      allowMultiple = false,
      defaultValue = "null",
      category = "testing",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Specifies a filter to forward to the test framework.  Used to limit "
              + "the tests run. Note that this does not affect which targets are built."
    )
    public String testFilter;

    @Option(
      name = "cache_test_results",
      defaultValue = "auto",
      category = "testing",
      abbrev = 't', // it's useful to toggle this on/off quickly
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "If set to 'auto', Bazel reruns a test if and only if: "
              + "(1) Bazel detects changes in the test or its dependencies, "
              + "(2) the test is marked as external, "
              + "(3) multiple test runs were requested with --runs_per_test, or"
              + "(4) the test previously failed. "
              + "If set to 'yes', Bazel caches all test results except for tests marked as "
              + "external. If set to 'no', Bazel does not cache any test results."
    )
    public TriState cacheTestResults;

    @Deprecated
    @Option(
      name = "test_result_expiration",
      defaultValue = "-1", // No expiration by defualt.
      category = "testing",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "This option is deprecated and has no effect."
    )
    public int testResultExpiration;

    @Option(
      name = "test_arg",
      allowMultiple = true,
      defaultValue = "",
      category = "testing",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Specifies additional options and arguments that should be passed to the test "
              + "executable. Can be used multiple times to specify several arguments. "
              + "If multiple tests are executed, each of them will receive identical arguments. "
              + "Used only by the 'bazel test' command."
    )
    public List<String> testArguments;

    @Option(
      name = "test_sharding_strategy",
      defaultValue = "explicit",
      category = "testing",
      converter = TestActionBuilder.ShardingStrategyConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Specify strategy for test sharding: "
              + "'explicit' to only use sharding if the 'shard_count' BUILD attribute is present. "
              + "'disabled' to never use test sharding. "
              + "'experimental_heuristic' to enable sharding on remotely executed tests without an "
              + "explicit  'shard_count' attribute which link in a supported framework. Considered "
              + "experimental."
    )
    public TestActionBuilder.TestShardingStrategy testShardingStrategy;

    @Option(
      name = "runs_per_test",
      allowMultiple = true,
      defaultValue = "1",
      category = "testing",
      converter = RunsPerTestConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Specifies number of times to run each test. If any of those attempts "
              + "fail for any reason, the whole test would be considered failed. "
              + "Normally the value specified is just an integer. Example: --runs_per_test=3 "
              + "will run all tests 3 times. "
              + "Alternate syntax: regex_filter@runs_per_test. Where runs_per_test stands for "
              + "an integer value and regex_filter stands "
              + "for a list of include and exclude regular expression patterns (Also see "
              + "--instrumentation_filter). Example: "
              + "--runs_per_test=//foo/.*,-//foo/bar/.*@3 runs all tests in //foo/ "
              + "except those under foo/bar three times. "
              + "This option can be passed multiple times. "
    )
    public List<PerLabelOptions> runsPerTest;
  }

  /** Configuration loader for test options */
  public static class Loader implements ConfigurationFragmentFactory {
    @Override
    public Fragment create(ConfigurationEnvironment env, BuildOptions buildOptions)
        throws InvalidConfigurationException {
      return new TestConfiguration(buildOptions.get(TestOptions.class));
    }

    @Override
    public Class<? extends Fragment> creates() {
      return TestConfiguration.class;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
      return ImmutableSet.of(TestOptions.class);
    }
  }

  private final TestOptions options;

  TestConfiguration(TestOptions options) {
    this.options = options;
  }

  @Override
  public void reportInvalidOptions(EventHandler reporter, BuildOptions buildOptions) {
    if (options.testShardingStrategy
        == TestActionBuilder.TestShardingStrategy.EXPERIMENTAL_HEURISTIC) {
      reporter.handle(
          Event.warn(
              "Heuristic sharding is intended as a one-off experimentation tool for determing the "
                  + "benefit from sharding certain tests. Please don't keep this option in your "
                  + ".blazerc or continuous build"));
    }
  }

  public String getTestFilter() {
    return options.testFilter;
  }

  public TriState cacheTestResults() {
    return options.cacheTestResults;
  }

  public List<String> getTestArguments() {
    return options.testArguments;
  }

  public TestActionBuilder.TestShardingStrategy testShardingStrategy() {
    return options.testShardingStrategy;
  }

  /**
   * @return number of times the given test should run. If the test doesn't match any of the
   *     filters, runs it once.
   */
  public int getRunsPerTestForLabel(Label label) {
    for (PerLabelOptions perLabelRuns : Lists.reverse(options.runsPerTest)) {
      if (perLabelRuns.isIncluded(label)) {
        return Integer.parseInt(Iterables.getOnlyElement(perLabelRuns.getOptions()));
      }
    }
    return 1;
  }
}
