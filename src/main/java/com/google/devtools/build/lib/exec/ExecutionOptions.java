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

import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.exec.TestStrategy.TestOutputFormat;
import com.google.devtools.build.lib.exec.TestStrategy.TestSummaryFormat;
import com.google.devtools.build.lib.packages.TestTimeout;
import com.google.devtools.build.lib.util.OptionsUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsBase;
import java.time.Duration;
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
    name = "verbose_failures",
    defaultValue = "false",
    category = "verbosity",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "If a command fails, print out the full command line."
  )
  public boolean verboseFailures;

  @Option(
    name = "subcommands",
    abbrev = 's',
    defaultValue = "false",
    category = "verbosity",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Display the subcommands executed during a build."
  )
  public boolean showSubcommands;

  @Option(
    name = "check_up_to_date",
    defaultValue = "false",
    category = "what",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Don't perform the build, just check if it is up-to-date.  If all targets are "
            + "up-to-date, the build completes successfully.  If any step needs to be executed "
            + "an error is reported and the build fails."
  )
  public boolean checkUpToDate;

  @Option(
    name = "check_tests_up_to_date",
    defaultValue = "false",
    category = "testing",
    implicitRequirements = {"--check_up_to_date"},
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Don't run tests, just check if they are up-to-date.  If all tests results are "
            + "up-to-date, the testing completes successfully.  If any test needs to be built or "
            + "executed, an error is reported and the testing fails.  This option implies "
            + "--check_up_to_date behavior."
  )
  public boolean testCheckUpToDate;

  @Option(
    name = "test_strategy",
    defaultValue = "",
    category = "testing",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Specifies which strategy to use when running tests."
  )
  public String testStrategy;

  @Option(
    name = "test_keep_going",
    defaultValue = "true",
    category = "testing",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "When disabled, any non-passing test will cause the entire build to stop. By default "
            + "all tests are run, even if some do not pass."
  )
  public boolean testKeepGoing;

  @Option(
    name = "runs_per_test_detects_flakes",
    defaultValue = "false",
    category = "testing",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "If true, any shard in which at least one run/attempt passes and at least one "
            + "run/attempt fails gets a FLAKY status."
  )
  public boolean runsPerTestDetectsFlakes;

  @Option(
    name = "flaky_test_attempts",
    defaultValue = "default",
    category = "testing",
    converter = TestStrategy.TestAttemptsConverter.class,
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Each test will be retried up to the specified number of times in case of any test "
            + "failure. Tests that required more than one attempt to pass would be marked as "
            + "'FLAKY' in the test summary. If this option is set, it should specify an int N or "
            + "the string 'default'. If it's an int, then all tests will be run up to N times. "
            + "If it is not specified or its value is 'default', then only a single test attempt "
            + "will be made for regular tests and three for tests marked explicitly as flaky by "
            + "their rule (flaky=1 attribute)."
  )
  public int testAttempts;

  @Option(
    name = "test_tmpdir",
    defaultValue = "null",
    category = "testing",
    converter = OptionsUtils.PathFragmentConverter.class,
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Specifies the base temporary directory for 'blaze test' to use."
  )
  public PathFragment testTmpDir;

  @Option(
    name = "test_output",
    defaultValue = "summary",
    category = "testing",
    converter = TestStrategy.TestOutputFormat.Converter.class,
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Specifies desired output mode. Valid values are 'summary' to output only test status "
            + "summary, 'errors' to also print test logs for failed tests, 'all' to print logs "
            + "for all tests and 'streamed' to output logs for all tests in real time "
            + "(this will force tests to be executed locally one at a time regardless of "
            + "--test_strategy value)."
  )
  public TestOutputFormat testOutput;

  @Option(
    name = "test_summary",
    defaultValue = "short",
    category = "testing",
    converter = TestStrategy.TestSummaryFormat.Converter.class,
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Specifies the desired format ot the test summary. Valid values are 'short' to print "
            + "information only about tests executed, 'terse', to print information only about "
            + "unsuccessful tests, 'detailed' to print detailed information about failed "
            + "test cases, and 'none' to omit the summary."
  )
  public TestSummaryFormat testSummary;

  @Option(
    name = "test_timeout",
    defaultValue = "-1",
    category = "testing",
    converter = TestTimeout.TestTimeoutConverter.class,
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Override the default test timeout values for test timeouts (in secs). If a single "
            + "positive integer value is specified it will override all categories.  If 4 "
            + "comma-separated integers are specified, they will override the timeouts for short, "
            + "moderate, long and eternal (in that order). In either form, a value of -1 tells "
            + "blaze to use its default timeouts for that category."
  )
  public Map<TestTimeout, Duration> testTimeout;

  @Option(
    name = "resource_autosense",
    defaultValue = "false",
    category = "strategy",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "This flag has no effect, and is deprecated"
  )
  public boolean useResourceAutoSense;

  @Option(
    name = "ram_utilization_factor",
    defaultValue = "67",
    category = "strategy",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Specify what percentage of the system's RAM Blaze should try to use for its subprocesses. "
            + "This option affects how many processes Blaze will try to run in parallel. "
            + "If you run several Blaze builds in parallel, using a lower value for "
            + "this option may avoid thrashing and thus improve overall throughput. "
            + "Using a value higher than the default is NOT recommended. "
            + "Note that Blaze's estimates are very coarse, so the actual RAM usage may be much "
            + "higher or much lower than specified. "
            + "Note also that this option does not affect the amount of memory that the Blaze "
            + "server itself will use. "
  )
  public int ramUtilizationPercentage;

  @Option(
    name = "local_resources",
    defaultValue = "null",
    category = "strategy",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Explicitly set amount of local resources available to Blaze. "
            + "By default, Blaze will query system configuration to estimate amount of RAM (in MB) "
            + "and number of CPU cores available for the locally executed build actions. It would "
            + "also assume default I/O capabilities of the local workstation (1.0). This options "
            + "allows to explicitly set all 3 values. Note, that if this option is used, Blaze "
            + "will ignore --ram_utilization_factor.",
    converter = ResourceSet.ResourceSetConverter.class
  )
  public ResourceSet availableResources;

  @Option(
    name = "local_test_jobs",
    defaultValue = "0",
    category = "testing",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "The max number of local test jobs to run concurrently. "
            + "0 means local resources will limit the number of local test jobs to run "
            + "concurrently instead. Setting this greater than the value for --jobs is ineffectual."
  )
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
        "If greater than 0, configures Blaze to cache file digests in memory based on their "
            + "metadata instead of recomputing the digests from disk every time they are needed. "
            + "Setting this to 0 ensures correctness because not all file changes can be noted "
            + "from file metadata. When not 0, the number indicates the size of the cache as the "
            + "number of file digests to be cached."
  )
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
}
