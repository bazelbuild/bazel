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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.OptionsDiffPredicate;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.LabelConverter;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.PerLabelOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.TestTimeout;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.TriState;
import java.time.Duration;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/** Test-related options. */
public class TestConfiguration extends Fragment {
  public static final OptionsDiffPredicate SHOULD_INVALIDATE_FOR_OPTION_DIFF =
      (options, changedOption, oldValue, newValue) -> {
        if (TestOptions.TRIM_TEST_CONFIGURATION.equals(changedOption)) {
          // changes in --trim_test_configuration itself always prompt invalidation
          return true;
        }
        if (!changedOption.getField().getDeclaringClass().equals(TestOptions.class)) {
          // options outside of TestOptions always prompt invalidation
          return true;
        }
        // other options in TestOptions require invalidation when --trim_test_configuration is off
        return !options.get(TestOptions.class).trimTestConfiguration;
      };

  /** Command-line options. */
  public static class TestOptions extends FragmentOptions {
    private static final OptionDefinition TRIM_TEST_CONFIGURATION =
        OptionsParser.getOptionDefinitionByName(TestOptions.class, "trim_test_configuration");

    @Option(
        name = "test_timeout",
        defaultValue = "-1",
        converter = TestTimeout.TestTimeoutConverter.class,
        documentationCategory = OptionDocumentationCategory.TESTING,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "Override the default test timeout values for test timeouts (in secs). If a single "
                + "positive integer value is specified it will override all categories.  If 4 "
                + "comma-separated integers are specified, they will override the timeouts for "
                + "short, moderate, long and eternal (in that order). In either form, a value of "
                + "-1 tells blaze to use its default timeouts for that category.")
    public Map<TestTimeout, Duration> testTimeout;

    @Option(
      name = "test_filter",
      allowMultiple = false,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Specifies a filter to forward to the test framework.  Used to limit "
              + "the tests run. Note that this does not affect which targets are built."
    )
    public String testFilter;

    @Option(
        name = "test_runner_fail_fast",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "Forwards fail fast option to the test runner. The test runner should stop execution"
                + " upon first failure.")
    public boolean testRunnerFailFast;

    @Option(
      name = "cache_test_results",
      defaultValue = "auto",
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
        defaultValue = "-1", // No expiration by default.
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "This option is deprecated and has no effect.")
    public int testResultExpiration;

    @Option(
      name = "trim_test_configuration",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.BUILD_TIME_OPTIMIZATION,
      effectTags = {
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.LOSES_INCREMENTAL_STATE,
      },
      help = "When enabled, test-related options will be cleared below the top level of the build. "
          + "When this flag is active, tests cannot be built as dependencies of non-test rules, "
          + "but changes to test-related options will not cause non-test rules to be re-analyzed."
    )
    public boolean trimTestConfiguration;

    @Option(
        name = "test_arg",
        allowMultiple = true,
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "Specifies additional options and arguments that should be passed to the test "
                + "executable. Can be used multiple times to specify several arguments. "
                + "If multiple tests are executed, each of them will receive identical arguments. "
                + "Used only by the 'bazel test' command.")
    public List<String> testArguments;

    @Option(
        name = "test_sharding_strategy",
        defaultValue = "explicit",
        converter = TestActionBuilder.ShardingStrategyConverter.class,
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "Specify strategy for test sharding: "
                + "'explicit' to only use sharding if the 'shard_count' BUILD attribute is "
                + "present. 'disabled' to never use test sharding.")
    public TestActionBuilder.TestShardingStrategy testShardingStrategy;

    @Option(
        name = "experimental_persistent_test_runner",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "Allows running java_test targets locally within a persistent worker. "
                + "To enable the persistent test runner one must run bazel test with the flags:"
                + "--test_strategy=local --strategy=TestRunner=worker "
                + " --experimental_persistent_test_runner")
    public boolean persistentTestRunner;

    @Option(
      name = "runs_per_test",
      allowMultiple = true,
      defaultValue = "1",
      converter = RunsPerTestConverter.class,
      documentationCategory = OptionDocumentationCategory.TESTING,
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

    @Option(
        name = "runs_per_test_detects_flakes",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "If true, any shard in which at least one run/attempt passes and at least one "
                + "run/attempt fails gets a FLAKY status.")
    public boolean runsPerTestDetectsFlakes;

    @Option(
        name = "experimental_cancel_concurrent_tests",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
        help =
            "If true, then Blaze will cancel concurrently running tests on the first successful "
                + "run. This is only useful in combination with --runs_per_test_detects_flakes.")
    public boolean cancelConcurrentTests;

    @Option(
        name = "coverage_support",
        converter = LabelConverter.class,
        defaultValue = "@bazel_tools//tools/test:coverage_support",
        documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
        effectTags = {
            OptionEffectTag.CHANGES_INPUTS,
            OptionEffectTag.AFFECTS_OUTPUTS,
            OptionEffectTag.LOADING_AND_ANALYSIS
        },
        help =
            "Location of support files that are required on the inputs of every test action "
                + "that collects code coverage. Defaults to '//tools/test:coverage_support'."
    )
    public Label coverageSupport;

    @Option(
        name = "coverage_report_generator",
        converter = LabelConverter.class,
        defaultValue = "@bazel_tools//tools/test:coverage_report_generator",
        documentationCategory = OptionDocumentationCategory.TOOLCHAIN,
        effectTags = {
            OptionEffectTag.CHANGES_INPUTS,
            OptionEffectTag.AFFECTS_OUTPUTS,
            OptionEffectTag.LOADING_AND_ANALYSIS
        },
        help =
            "Location of the binary that is used to generate coverage reports. This must "
                + "currently be a filegroup that contains a single file, the binary. Defaults to "
                + "'//tools/test:coverage_report_generator'."
    )
    public Label coverageReportGenerator;

    @Option(
        name = "experimental_fetch_all_coverage_outputs",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.LOADING_AND_ANALYSIS},
        help =
            "If true, then Bazel fetches the entire coverage data directory for each test during a "
                + "coverage run.")
    public boolean fetchAllCoverageOutputs;

    @Override
    public FragmentOptions getHost() {
      TestOptions hostOptions = (TestOptions) getDefault();
      // These fields are used in late-bound attributes, which must not be null in the host
      // configuration.
      hostOptions.coverageSupport = this.coverageSupport;
      hostOptions.coverageReportGenerator = this.coverageReportGenerator;
      // trimTestConfiguration is a global analysis option and should be platform-agnostic
      hostOptions.trimTestConfiguration = this.trimTestConfiguration;
      return hostOptions;
    }
  }

  /** Configuration loader for test options */
  public static class Loader implements ConfigurationFragmentFactory {
    @Override
    public Fragment create(BuildOptions buildOptions)
        throws InvalidConfigurationException {
      if (!buildOptions.contains(TestOptions.class)) {
        return null;
      }
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
  private final ImmutableMap<TestTimeout, Duration> testTimeout;

  private TestConfiguration(TestOptions options) {
    this.options = options;
    this.testTimeout = ImmutableMap.copyOf(options.testTimeout);
  }

  /** Returns test timeout mapping as set by --test_timeout options. */
  public ImmutableMap<TestTimeout, Duration> getTestTimeout() {
    return testTimeout;
  }

  public String getTestFilter() {
    return options.testFilter;
  }

  public boolean getTestRunnerFailFast() {
    return options.testRunnerFailFast;
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

  public boolean isPersistentTestRunner() {
    return options.persistentTestRunner;
  }

  public Label getCoverageSupport(){
    return options.coverageSupport;
  }

  public Label getCoverageReportGenerator(){
    return options.coverageReportGenerator;
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

  public boolean runsPerTestDetectsFlakes() {
    return options.runsPerTestDetectsFlakes;
  }

  public boolean cancelConcurrentTests() {
    return options.cancelConcurrentTests;
  }

  public boolean fetchAllCoverageOutputs() {
    return options.fetchAllCoverageOutputs;
  }

  /**
   * Option converter that han handle two styles of value for "--runs_per_test":
   *
   * <ul>
   *   <li>--runs_per_test=NUMBER: Run each test NUMBER times.
   *   <li>--runs_per_test=test_regex@NUMBER: Run each test that matches test_regex NUMBER times.
   *       This form can be repeated with multiple regexes.
   * </ul>
   */
  public static class RunsPerTestConverter extends PerLabelOptions.PerLabelOptionsConverter {
    @Override
    public PerLabelOptions convert(String input) throws OptionsParsingException {
      try {
        return parseAsInteger(input);
      } catch (NumberFormatException ignored) {
        return parseAsRegex(input);
      }
    }

    private PerLabelOptions parseAsInteger(String input)
        throws NumberFormatException, OptionsParsingException {
      int numericValue = Integer.parseInt(input);
      if (numericValue <= 0) {
        throw new OptionsParsingException("'" + input + "' should be >= 1");
      } else {
        RegexFilter catchAll =
            new RegexFilter(Collections.singletonList(".*"), Collections.<String>emptyList());
        return new PerLabelOptions(catchAll, Collections.singletonList(input));
      }
    }

    private PerLabelOptions parseAsRegex(String input) throws OptionsParsingException {
      PerLabelOptions testRegexps = super.convert(input);
      if (testRegexps.getOptions().size() != 1) {
        throw new OptionsParsingException("'" + input + "' has multiple runs for a single pattern");
      }
      String runsPerTest = Iterables.getOnlyElement(testRegexps.getOptions());
      try {
        int numericRunsPerTest = Integer.parseInt(runsPerTest);
        if (numericRunsPerTest <= 0) {
          throw new OptionsParsingException("'" + input + "' has a value < 1");
        }
      } catch (NumberFormatException e) {
        throw new OptionsParsingException("'" + input + "' has a non-numeric value", e);
      }
      return testRegexps;
    }

    @Override
    public String getTypeDescription() {
      return "a positive integer or test_regex@runs. This flag may be passed more than once";
    }
  }
}
