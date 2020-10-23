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
package com.google.devtools.build.lib.exec;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext.ShowSubcommands;
import com.google.devtools.build.lib.actions.LocalHostCapacity;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.analysis.config.PerLabelOptions;
import com.google.devtools.build.lib.util.OptionsUtils;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.build.lib.util.ResourceConverter;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.BoolOrEnumConverter;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Converters.CommaSeparatedNonEmptyOptionListConverter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Options affecting the execution phase of a build.
 *
 * These options are interpreted by the BuildTool to choose an Executor to
 * be used for the build.
 *
 * Note: from the user's point of view, the characteristic function of this
 * set of options is indistinguishable from that of the BuildRequestOptions:
 * they are all per-request.  The difference is only apparent in the
 * implementation: these options are used only by the lib.exec machinery, which
 * affects how C++ and Java compilation occur.  (The BuildRequestOptions
 * contain a mixture of "semantic" options affecting the choice of targets to
 * build, and "non-semantic" options affecting the lib.actions machinery.)
 * Ideally, the user would be unaware of the difference.  For now, the usage
 * strings are identical modulo "part 1", "part 2".
 */
public class ExecutionOptions extends OptionsBase {

  public static final ExecutionOptions DEFAULTS = Options.getDefaults(ExecutionOptions.class);

  @Option(
      name = "spawn_strategy",
      defaultValue = "",
      converter = CommaSeparatedNonEmptyOptionListConverter.class,
      documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
      effectTags = {OptionEffectTag.EXECUTION},
      help =
          "Specify how spawn actions are executed by default. Accepts a comma-separated list of"
              + " strategies from highest to lowest priority. For each action Bazel picks the"
              + " strategy with the highest priority that can execute the action. The default"
              + " value is \"remote,worker,sandboxed,local\". See"
              + " https://blog.bazel.build/2019/06/19/list-strategy.html for details.")
  public List<String> spawnStrategy;

  @Option(
      name = "genrule_strategy",
      defaultValue = "",
      converter = CommaSeparatedNonEmptyOptionListConverter.class,
      documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
      effectTags = {OptionEffectTag.EXECUTION},
      help =
          "Specify how to execute genrules. This flag will be phased out. Instead, use "
              + "--spawn_strategy=<value> to control all actions or --strategy=Genrule=<value> "
              + "to control genrules only.")
  public List<String> genruleStrategy;

  @Option(
      name = "strategy",
      allowMultiple = true,
      converter = Converters.StringToStringListConverter.class,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
      effectTags = {OptionEffectTag.EXECUTION},
      help =
          "Specify how to distribute compilation of other spawn actions. Accepts a comma-separated"
              + " list of strategies from highest to lowest priority. For each action Bazel picks"
              + " the strategy with the highest priority that can execute the action. The default"
              + " value is \"remote,worker,sandboxed,local\". See"
              + " https://blog.bazel.build/2019/06/19/list-strategy.html for details.")
  public List<Map.Entry<String, List<String>>> strategy;

  @Option(
      name = "strategy_regexp",
      allowMultiple = true,
      converter = RegexFilterAssignmentConverter.class,
      documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
      effectTags = {OptionEffectTag.EXECUTION},
      defaultValue = "null",
      help =
          "Override which spawn strategy should be used to execute spawn actions that have "
              + "descriptions matching a certain regex_filter. See --per_file_copt for details on"
              + "regex_filter matching. "
              + "The first regex_filter that matches the description is used. "
              + "This option overrides other flags for specifying strategy. "
              + "Example: --strategy_regexp=//foo.*\\.cc,-//foo/bar=local means to run actions "
              + "using local strategy if their descriptions match //foo.*.cc but not //foo/bar. "
              + "Example: --strategy_regexp='Compiling.*/bar=local "
              + " --strategy_regexp=Compiling=sandboxed will run 'Compiling //foo/bar/baz' with "
              + "the 'local' strategy, but reversing the order would run it with 'sandboxed'. ")
  public List<Map.Entry<RegexFilter, List<String>>> strategyByRegexp;

  @Option(
      name = "materialize_param_files",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.EXECUTION},
      help =
          "Writes intermediate parameter files to output tree even when using "
              + "remote action execution. Useful when debugging actions. "
              + "This is implied by --subcommands and --verbose_failures.")
  public boolean materializeParamFiles;

  @Option(
      name = "experimental_materialize_param_files_directly",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.EXECUTION},
      help = "If materializing param files, do so with direct writes to disk.")
  public boolean materializeParamFilesDirectly;

  public boolean shouldMaterializeParamFiles() {
    // Implied by --subcommands and --verbose_failures
    return materializeParamFiles
        || showSubcommands != ActionExecutionContext.ShowSubcommands.FALSE
        || verboseFailures;
  }

  @Option(
      name = "verbose_failures",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help = "If a command fails, print out the full command line.")
  public boolean verboseFailures;

  @Option(
      name = "subcommands",
      abbrev = 's',
      defaultValue = "false",
      converter = ShowSubcommandsConverter.class,
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help = "Display the subcommands executed during a build.")
  public ShowSubcommands showSubcommands;

  @Option(
      name = "check_up_to_date",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
      effectTags = {OptionEffectTag.EXECUTION},
      help =
          "Don't perform the build, just check if it is up-to-date.  If all targets are "
              + "up-to-date, the build completes successfully.  If any step needs to be executed "
              + "an error is reported and the build fails.")
  public boolean checkUpToDate;

  @Option(
      name = "check_tests_up_to_date",
      defaultValue = "false",
      implicitRequirements = {"--check_up_to_date"},
      documentationCategory = OptionDocumentationCategory.TESTING,
      effectTags = {OptionEffectTag.EXECUTION},
      help =
          "Don't run tests, just check if they are up-to-date.  If all tests results are "
              + "up-to-date, the testing completes successfully.  If any test needs to be built or "
              + "executed, an error is reported and the testing fails.  This option implies "
              + "--check_up_to_date behavior.")
  public boolean testCheckUpToDate;

  @Option(
      name = "test_strategy",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.TESTING,
      effectTags = {OptionEffectTag.EXECUTION},
      help = "Specifies which strategy to use when running tests.")
  public String testStrategy;

  @Option(
      name = "test_keep_going",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.TESTING,
      effectTags = {OptionEffectTag.EXECUTION},
      help =
          "When disabled, any non-passing test will cause the entire build to stop. By default "
              + "all tests are run, even if some do not pass.")
  public boolean testKeepGoing;

  @Option(
      name = "flaky_test_attempts",
      allowMultiple = true,
      defaultValue = "default",
      converter = TestAttemptsConverter.class,
      documentationCategory = OptionDocumentationCategory.TESTING,
      effectTags = {OptionEffectTag.EXECUTION},
      help =
          "Each test will be retried up to the specified number of times in case of any test"
              + " failure. Tests that required more than one attempt to pass are marked as 'FLAKY'"
              + " in the test summary. Normally the value specified is just an integer or the"
              + " string 'default'. If an integer, then all tests will be run up to N times. If"
              + " 'default', then only a single test attempt will be made for regular tests and"
              + " three for tests marked explicitly as flaky by their rule (flaky=1 attribute)."
              + " Alternate syntax: regex_filter@flaky_test_attempts. Where flaky_test_attempts is"
              + " as above and regex_filter stands for a list of include and exclude regular"
              + " expression patterns (Also see --runs_per_test). Example:"
              + " --flaky_test_attempts=//foo/.*,-//foo/bar/.*@3 deflakes all tests in //foo/"
              + " except those under foo/bar three times. This option can be passed multiple"
              + " times. The most recently passed argument that matches takes precedence. If"
              + " nothing matches, behavior is as if 'default' above.")
  public List<PerLabelOptions> testAttempts;

  @Option(
      name = "test_tmpdir",
      defaultValue = "null",
      converter = OptionsUtils.PathFragmentConverter.class,
      documentationCategory = OptionDocumentationCategory.TESTING,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Specifies the base temporary directory for 'bazel test' to use.")
  public PathFragment testTmpDir;

  @Option(
      name = "test_output",
      defaultValue = "summary",
      converter = TestOutputFormat.Converter.class,
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {
        OptionEffectTag.TEST_RUNNER,
        OptionEffectTag.TERMINAL_OUTPUT,
        OptionEffectTag.EXECUTION
      },
      help =
          "Specifies desired output mode. Valid values are 'summary' to output only test status "
              + "summary, 'errors' to also print test logs for failed tests, 'all' to print logs "
              + "for all tests and 'streamed' to output logs for all tests in real time "
              + "(this will force tests to be executed locally one at a time regardless of "
              + "--test_strategy value).")
  public TestOutputFormat testOutput;

  @Option(
      name = "max_test_output_bytes",
      defaultValue = "-1",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {
        OptionEffectTag.TEST_RUNNER,
        OptionEffectTag.TERMINAL_OUTPUT,
        OptionEffectTag.EXECUTION
      },
      help =
          "Specifies maximum per-test-log size that can be emitted when --test_summary is 'errors' "
              + "or 'all'. Useful for avoiding overwhelming the output with excessively noisy test "
              + "output. The test header is included in the log size. Negative values imply no "
              + "limit. Output is all or nothing.")
  public int maxTestOutputBytes;

  @Option(
      name = "test_summary",
      defaultValue = "short",
      converter = TestSummaryFormat.Converter.class,
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help =
          "Specifies the desired format ot the test summary. Valid values are 'short' to print "
              + "information only about tests executed, 'terse', to print information only about "
              + "unsuccessful tests that were run, 'detailed' to print detailed information about "
              + "failed test cases, and 'none' to omit the summary.")
  public TestSummaryFormat testSummary;

  @Option(
      name = "resource_autosense",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "This flag has no effect, and is deprecated")
  public boolean useResourceAutoSense;

  @Option(
      name = "local_resources",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Deprecated by '--incompatible_remove_local_resources'. Please use "
              + "'--local_ram_resources' and '--local_cpu_resources'",
      deprecationWarning =
          "--local_resources is deprecated. Please use"
              + " --local_ram_resources and --local_cpu_resources instead.",
      converter = ResourceSet.ResourceSetConverter.class)
  public ResourceSet availableResources;

  @Option(
      name = "incompatible_remove_local_resources",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
      effectTags = {OptionEffectTag.EXECUTION},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "Deprecate local_resources in favor of --local_ram_resources and "
              + "--local_cpu_resources.")
  public boolean removeLocalResources;

  @Option(
      name = "local_cpu_resources",
      defaultValue = "HOST_CPUS",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Explicitly set the number of local CPU threads available to Bazel. Takes "
              + "an integer, or \"HOST_CPUS\", optionally followed by [-|*]<float> "
              + "(eg. HOST_CPUS*.5 to use half the available CPU cores)."
              + "By default, (\"HOST_CPUS\"), Bazel will query system configuration to estimate "
              + "number of CPU cores available for the locally executed build actions. "
              + "Note: This is a no-op if --local_resources is set.",
      converter = CpuResourceConverter.class)
  public float localCpuResources;

  @Option(
      name = "local_ram_resources",
      defaultValue = "HOST_RAM*.67",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Explicitly set the amount of local host RAM (in MB) available to Bazel. Takes "
              + "an integer, or \"HOST_RAM\", optionally followed by [-|*]<float> "
              + "(eg. HOST_RAM*.5 to use half the available RAM)."
              + "By default, (\"HOST_RAM*.67\"), Bazel will query system configuration to estimate "
              + "amount of RAM available for the locally executed build actions and will use 67% "
              + "of available RAM. "
              + "Note: This is a no-op if --local_resources is set.",
      converter = RamResourceConverter.class)
  public float localRamResources;

  @Option(
    name = "experimental_local_memory_estimate",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Estimate the actual memory available online. "
            + "By default, Blaze assumes most actions use a fixed amount of memory, and counts "
            + "that against the total available system memory, regardless of how much memory is "
            + "actually available.  This option enables online estimation of how much memory is "
            + "available at any given time, and thus does not require accurate estimation of how "
            + "much memory a given action will take."
  )
  public boolean localMemoryEstimate;

  @Option(
      name = "local_test_jobs",
      defaultValue = "auto",
      documentationCategory = OptionDocumentationCategory.TESTING,
      effectTags = {OptionEffectTag.EXECUTION},
      help =
          "The max number of local test jobs to run concurrently. "
              + "Takes "
              + ResourceConverter.FLAG_SYNTAX
              + ". 0 means local resources will limit the number of local test jobs to run "
              + "concurrently instead. Setting this greater than the value for --jobs "
              + "is ineffectual.",
      converter = LocalTestJobsConverter.class)
  public int localTestJobs;

  public boolean usingLocalTestJobs() {
    return localTestJobs != 0;
  }

  @Option(
    name = "debug_print_action_contexts",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Print the contents of the SpawnActionContext and ContextProviders maps."
  )
  public boolean debugPrintActionContexts;

  @Option(
      name = "cache_computed_file_digests",
      defaultValue = "50000",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "If greater than 0, configures Bazel to cache file digests in memory based on their "
              + "metadata instead of recomputing the digests from disk every time they are needed. "
              + "Setting this to 0 ensures correctness because not all file changes can be noted "
              + "from file metadata. When not 0, the number indicates the size of the cache as the "
              + "number of file digests to be cached.")
  public long cacheSizeForComputedFileDigests;

  @Option(
    name = "experimental_enable_critical_path_profiling",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "If set (the default), critical path profiling is enabled for the execution phase. "
            + "This has a slight overhead in RAM and CPU, and may prevent Bazel from making certain"
            + " aggressive RAM optimizations in some cases."
  )
  public boolean enableCriticalPathProfiling;

  @Option(
      name = "experimental_stats_summary",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      defaultValue = "false",
      help = "Enable a modernized summary of the build stats."
  )
  public boolean statsSummary;

  @Option(
      name = "experimental_execution_log_file",
      defaultValue = "null",
      category = "verbosity",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = OptionsUtils.PathFragmentConverter.class,
      help = "Log the executed spawns into this file as delimited Spawn protos.")
  public PathFragment executionLogFile;

  @Option(
      name = "execution_log_binary_file",
      defaultValue = "null",
      category = "verbosity",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = OptionsUtils.PathFragmentConverter.class,
      help = "Log the executed spawns into this file as delimited Spawn protos.")
  public PathFragment executionLogBinaryFile;

  @Option(
      name = "execution_log_json_file",
      defaultValue = "null",
      category = "verbosity",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      converter = OptionsUtils.PathFragmentConverter.class,
      help =
          "Log the executed spawns into this file as json representation of the delimited Spawn"
              + " protos.")
  public PathFragment executionLogJsonFile;

  @Option(
      name = "experimental_split_xml_generation",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
      effectTags = {OptionEffectTag.EXECUTION},
      help =
          "If this flag is set, and a test action does not generate a test.xml file, then "
              + "Bazel uses a separate action to generate a dummy test.xml file containing the "
              + "test log. Otherwise, Bazel generates a test.xml as part of the test action.")
  public boolean splitXmlGeneration;

  /** An enum for specifying different formats of test output. */
  public enum TestOutputFormat {
    SUMMARY, // Provide summary output only.
    ERRORS, // Print output from failed tests to the stderr after the test failure.
    ALL, // Print output from all tests to the stderr after the test completion.
    STREAMED; // Stream output for each test.

    /** Converts to {@link TestOutputFormat}. */
    public static class Converter extends EnumConverter<TestOutputFormat> {
      public Converter() {
        super(TestOutputFormat.class, "test output");
      }
    }
  }

  /** An enum for specifying different formatting styles of test summaries. */
  public enum TestSummaryFormat {
    SHORT, // Print information only about tests.
    TERSE, // Like "SHORT", but even shorter: Do not print PASSED and NO STATUS tests.
    DETAILED, // Print information only about failed test cases.
    NONE, // Do not print summary.
    TESTCASE; // Print summary in test case resolution, do not print detailed information about
    // failed test cases.

    /** Converts to {@link TestSummaryFormat}. */
    public static class Converter extends EnumConverter<TestSummaryFormat> {
      public Converter() {
        super(TestSummaryFormat.class, "test summary");
      }
    }
  }

  /** Converter for the --flaky_test_attempts option. */
  public static class TestAttemptsConverter extends PerLabelOptions.PerLabelOptionsConverter {
    private static final int MIN_VALUE = 1;
    private static final int MAX_VALUE = 10;

    private void validateInput(String input) throws OptionsParsingException {
      if ("default".equals(input)) {
        return;
      } else {
        Integer value = Integer.parseInt(input);
        if (value < MIN_VALUE) {
          throw new OptionsParsingException("'" + input + "' should be >= " + MIN_VALUE);
        } else if (value < MIN_VALUE || value > MAX_VALUE) {
          throw new OptionsParsingException("'" + input + "' should be <= " + MAX_VALUE);
        }
        return;
      }
    }

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
      validateInput(input);
      RegexFilter catchAll =
          new RegexFilter(Collections.singletonList(".*"), Collections.<String>emptyList());
      return new PerLabelOptions(catchAll, Collections.singletonList(input));
    }

    private PerLabelOptions parseAsRegex(String input) throws OptionsParsingException {
      PerLabelOptions testRegexps = super.convert(input);
      if (testRegexps.getOptions().size() != 1) {
        throw new OptionsParsingException("'" + input + "' has multiple runs for a single pattern");
      }
      String runsPerTest = Iterables.getOnlyElement(testRegexps.getOptions());
      try {
        // Run this in order to catch errors.
        validateInput(runsPerTest);
      } catch (NumberFormatException e) {
        throw new OptionsParsingException("'" + input + "' has a non-numeric value", e);
      }
      return testRegexps;
    }

    @Override
    public String getTypeDescription() {
      return "a positive integer, the string \"default\", or test_regex@attempts. "
          + "This flag may be passed more than once";
    }
  }

  /** Converter for --local_test_jobs, which takes {@value FLAG_SYNTAX} */
  public static class LocalTestJobsConverter extends ResourceConverter {
    public LocalTestJobsConverter() throws OptionsParsingException {
      super(/* autoSupplier= */ () -> 0, /* minValue= */ 0, /* maxValue= */ Integer.MAX_VALUE);
    }
  }

  /** Converter for --subcommands */
  public static class ShowSubcommandsConverter extends BoolOrEnumConverter<ShowSubcommands> {
    public ShowSubcommandsConverter() {
      super(
          ShowSubcommands.class, "subcommand option", ShowSubcommands.TRUE, ShowSubcommands.FALSE);
    }
  }

  /**
   * Converter for --local_cpu_resources, which takes an integer greater than or equal to 1, or
   * "HOST_CPUS", optionally followed by [-|*]<float>.
   */
  public static class CpuResourceConverter extends ResourceConverter {
    public CpuResourceConverter() {
      super(
          ImmutableMap.of(
              "HOST_CPUS",
              () -> (int) Math.ceil(LocalHostCapacity.getLocalHostCapacity().getCpuUsage())),
          1,
          Integer.MAX_VALUE);
    }

    @Override
    public String getTypeDescription() {
      return "an integer, or \"HOST_CPUS\", optionally followed by [-|*]<float>.";
    }
  }

  /**
   * Converter for --local_cpu_resources, which takes an integer greater than or equal to 1, or
   * "HOST_RAM", optionally followed by [-|*]<float>.
   */
  public static class RamResourceConverter extends ResourceConverter {
    public RamResourceConverter() {
      super(
          ImmutableMap.of(
              "HOST_RAM",
              () -> (int) Math.ceil(LocalHostCapacity.getLocalHostCapacity().getMemoryMb())),
          1,
          Integer.MAX_VALUE);
    }

    @Override
    public String getTypeDescription() {
      return "an integer, or \"HOST_RAM\", optionally followed by [-|*]<float>.";
    }
  }
}
