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

package com.google.devtools.build.lib.exec;

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.io.ByteStreams;
import com.google.common.io.CharStreams;
import com.google.common.io.Closeables;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.TestExecException;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.ShellEscaper;
import com.google.devtools.build.lib.util.UserUtils;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.util.io.FileWatcher;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.lib.view.test.TestActionContext;
import com.google.devtools.build.lib.view.test.TestResult;
import com.google.devtools.build.lib.view.test.TestRunnerAction;
import com.google.devtools.build.lib.view.test.TestTargetProperties;
import com.google.devtools.common.options.Converters.RangeConverter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.OptionsClassProvider;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.testing.proto.CoverageInstrumentation;
import com.google.testing.proto.HierarchicalTestResult;
import com.google.testing.proto.InfrastructureFailureInfo;
import com.google.testing.proto.Property;
import com.google.testing.proto.TargetCoverage;
import com.google.testing.proto.TargetCoverage.Builder;
import com.google.testing.proto.TestSize;
import com.google.testing.proto.TestStatus;
import com.google.testing.proto.TestTargetResult;
import com.google.testing.proto.TestWarning;
import com.google.testing.proto.TextFile;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A strategy for executing a {@link TestRunnerAction}.
 */
public abstract class TestStrategy implements TestActionContext {
  /**
   * Converter for the --flaky_test_attempts option.
   */
  public static class TestAttemptsConverter extends RangeConverter {
    public TestAttemptsConverter() {
      super(1, 10);
    }

    @Override
    public Integer convert(String input) throws OptionsParsingException {
      if ("default".equals(input)) {
        return -1;
      } else {
        return super.convert(input);
      }
    }

    @Override
    public String getTypeDescription() {
      return super.getTypeDescription() + " or the string \"default\"";
    }
  }

  public enum TestOutputFormat {
    SUMMARY,  // Provide summary output only.
    ERRORS,   // Print output from failed tests to the stderr after the test failure.
    ALL,      // Print output from all tests to the stderr after the test completion.
    STREAMED; // Stream output for each test.

    /**
     * Converts to {@link TestOutputFormat}.
     */
    public static class Converter extends EnumConverter<TestOutputFormat> {
      public Converter() {
        super(TestOutputFormat.class, "test output");
      }
    }
  }

  public enum TestSummaryFormat {
    SHORT,      // Print information only about tests.
    TERSE,      // Like "SHORT", but even shorter: Do not print PASSED tests.
    DETAILED,   // Print information only about failed test cases.
    NONE;       // Do not print summary.

    /**
     * Converts to {@link TestSummaryFormat}.
     */
    public static class Converter extends EnumConverter<TestSummaryFormat> {
      public Converter() {
        super(TestSummaryFormat.class, "test summary");
      }
    }
  }

  public static final PathFragment COVERAGE_TMP_ROOT = new PathFragment("_coverage");
  public static final PathFragment TEST_TMP_ROOT = new PathFragment("_tmp");

  private static final String TEST_BRIDGE_TEST_FILTER_ENV = "TESTBRIDGE_TEST_ONLY";

  protected final ExecutionOptions executionOptions;

  private final OutErr outErr;

  public TestStrategy(OptionsClassProvider options, OutErr outErr) {
    this.executionOptions = options.getOptions(ExecutionOptions.class);
    this.outErr = outErr;
  }

  /** The strategy name, preferably suitable for passing to --test_strategy. */
  public abstract String testStrategyName();

  /** Returns a TextFile message for a specified path. This is an abstract method because
   * test strategy implementations may want to put more information into the message than simply
   * its path.
   */
  protected abstract TextFile.Builder textFileFor(Executor executor, Path path);

  /**
   * Run the test and interpret test results.
   */
  protected abstract TestTargetResult.Builder runTestProcess(
      TestRunnerAction action, Spawn spawn,
      TestTargetResult.Builder resultBulder, ActionExecutionContext actionExecutionContext)
      throws InterruptedException, IOException;

  @Override
  public abstract void exec(TestRunnerAction action,
      ActionExecutionContext actionExecutionContext) throws ExecException;

  @Override
  public abstract String strategyLocality(TestRunnerAction action);

  /**
   * Callback for determining the strategy locality.
   *
   * @param action the test action
   * @param localRun whether to run it locally
   */
  protected String strategyLocality(TestRunnerAction action, boolean localRun) {
    return strategyLocality(action);
  }

  /**
   * Called when the test invocation (including retries and fallback) are
   * completed. This method broadcasts the test result to listeners, and writes
   * the output if necessary.
   */
  protected final void finalizeTest(Executor executor,
      TestRunnerAction testAction, TestTargetResult result, FileOutErr outErr)
          throws IOException, ExecException {
    postTestResult(executor, testAction, result);
    processTestOutput(executor, outErr, result, testAction.getTestName());

    if (!executionOptions.testKeepGoing && result.getStatus() != TestStatus.PASSED) {
      throw new TestExecException("Test failed: aborting");
    }
  }

  /**
   * Returns mutable map of default testing shell environment. By itself it is
   * incomplete and is modified further by the specific test strategy
   * implementations (mostly due to the fact that environments used locally and
   * remotely are different).
   */
  protected final Map<String, String> getDefaultTestEnvironment(TestRunnerAction action) {
    Map<String, String> env = new HashMap<>();

    env.putAll(action.getConfiguration().getDefaultShellEnvironment());
    env.remove("LANG");
    env.put("TZ", "UTC");
    env.put("TEST_SIZE", action.getTestProperties().getSize().toString());

    if (action.isSharded()) {
      env.put("TEST_SHARD_INDEX", Integer.toString(action.getShardNum()));
      env.put("TEST_TOTAL_SHARDS",
          Integer.toString(action.getExecutionSettings().getTotalShards()));
    }

    // When we run test multiple times, set different TEST_RANDOM_SEED values for each run.
    if (action.getConfiguration().getRunsPerTestForLabel(action.getOwner().getLabel()) > 1) {
      env.put("TEST_RANDOM_SEED", Integer.toString(action.getRunNumber() + 1));
    }

    if (isCoverageMode(action)) {
      env.put("COVERAGE_MANIFEST",
          action.getExecutionSettings().getInstrumentedFileManifest().getExecPathString());
      // Instruct test-setup.sh not to cd into the runfiles directory.
      env.put("RUNTEST_PRESERVE_CWD", "1");
      env.put("MICROCOVERAGE_REQUESTED", isMicroCoverageMode(action) ? "true" : "false");

      env.putAll(action.getConfiguration().getCoverageEnvironment());
    }

    String testFilter = action.getExecutionSettings().getTestFilter();
    if (testFilter != null) {
      env.put(TEST_BRIDGE_TEST_FILTER_ENV, testFilter);
    }

    return env;
  }

  /**
   * Returns true if coverage data should be gathered.
   */
  protected static boolean isCoverageMode(TestRunnerAction action) {
    return action.getCoverageData() != null;
  }

  /**
   * Returns true if coverage data should be gathered.
   */
  protected static boolean isMicroCoverageMode(TestRunnerAction action) {
    return action.getMicroCoverageData() != null;
  }

  /**
   * Returns directory to store coverage results for the given action relative
   * to the execution root. This directory is used to store all coverage results
   * related to the test execution with exception of the locally generated
   * *.gcda files. Those are stored separately using google3 relative path
   * within coverage directory.
   *
   * Coverage directory name for the given test runner action is constructed as:
   *   $(blaze info execution_root)/_coverage/<google3_target_path>/<test_log_name>
   * where <test_log_name> is usually a target name but potentially can include
   * extra suffix, such as a shard number (if test execution was sharded).
   */
  protected static PathFragment getCoverageDirectory(TestRunnerAction action) {
    return COVERAGE_TMP_ROOT.getRelative(FileSystemUtils.removeExtension(
        action.getTestLog().getRootRelativePath()));
  }

  /**
   * Returns the number of attempts specific test action can be retried.
   *
   * For rules with "flaky = 1" attribute, this method will return 3 unless
   * --flaky_test_attempts option is given and specifies another value.
   */
  @VisibleForTesting
  /* protected */ public int getTestAttempts(TestRunnerAction action) {
    if (executionOptions.testAttempts == -1) {
      return action.getTestProperties().isFlaky() ? 3 : 1;
    } else {
      return executionOptions.testAttempts;
    }
  }

  /**
   * Renames current test action outputs in order to preserve logs and other information
   * (e.g. files downloaded from remote execution) for previous test retries.
   */
  private void renameOutputs(Executor executor, TestTargetResult.Builder resultBuilder,
      TestRunnerAction action, int attemptId, boolean localRun) throws IOException {
    String namePrefix =
        FileSystemUtils.removeExtension(action.getTestLog().getExecPath().getBaseName());
    Path attemptsDir =
        action.getTestLog().getPath().getParentDirectory().getChild(namePrefix + "_attempts");
    attemptsDir.createDirectory();
    String attemptPrefix = strategyLocality(action, localRun) + "_" + attemptId;
    if (action.getTestLog().getPath().exists()) {
      action.getTestLog().getPath().renameTo(attemptsDir.getChild(attemptPrefix + ".log"));
    }
    resultBuilder.setCombinedOut(textFileFor(
        executor, attemptsDir.getChild(attemptPrefix + ".log")));
    if (action.getXmlOutputPath().exists()) {
      Path destinationPath = attemptsDir.getChild(attemptPrefix + ".xml");
      action.getXmlOutputPath().renameTo(destinationPath);
      Preconditions.checkState(resultBuilder.hasXml());
      resultBuilder.setXml(textFileFor(executor, destinationPath));
      HierarchicalTestResult testResultParsed = parseTestResult(destinationPath);
      if (testResultParsed != null) {
        resultBuilder.setHierarchicalTestResult(testResultParsed);
      }
    }
    if (action.getSplitLogsPath().exists()) {
      // For logsplitter output log.
      Path destinationDir = attemptsDir.getChild(attemptPrefix + ".raw_splitlogs");
      action.getSplitLogsDir().renameTo(destinationDir);

      // The split_logs file (containing concatenated split logs) must be
      // called test.splitlogs for backwards compatibility.
      // @see com.google.devtools.build.lib.view.test.TestRunnerAction#getSplitLogsPath
      Path destinationPath = destinationDir.getChild(action.getSplitLogsPath().getBaseName());
      resultBuilder.setSplitlogs(textFileFor(executor, destinationPath));
    }
    if (action.getUndeclaredOutputsDir().exists()) {
      Path destinationDir = attemptsDir.getChild(attemptPrefix + ".outputs");
      action.getUndeclaredOutputsDir().renameTo(destinationDir);
      Path destinationPath = destinationDir.getChild("outputs.zip");
      resultBuilder.setUndeclaredOutputsZip(textFileFor(executor, destinationPath));
    }
    if (action.getUndeclaredOutputsAnnotationsDir().exists()) {
      Path destinationDir = attemptsDir.getChild(attemptPrefix + ".outputs_manifest");
      action.getUndeclaredOutputsAnnotationsDir().renameTo(destinationDir);
      Path destinationPath = destinationDir.getChild("MANIFEST");
      resultBuilder.setUndeclaredOutputsManifest(textFileFor(executor, destinationPath));
      destinationPath = destinationDir.getChild("ANNOTATIONS");
      resultBuilder.setUndeclaredOutputsAnnotations(textFileFor(executor, destinationPath));
    }
    if (action.getTestDiagnostics().exists()) {
      // Extra test diagnostics.
      Path destinationDir = attemptsDir.getChild(attemptPrefix + ".test_diagnostics");
      action.getTestDiagnosticsDir().renameTo(destinationDir);
      Path destinationPath = destinationDir.getChild(action.getTestDiagnostics().getBaseName());
      resultBuilder.setTestDiagnosticsData(textFileFor(executor, destinationPath));
    }

    // Don't allow left-over sentinel files between test attempts.
    if (action.getTestShard() != null) {
      action.getTestShard().delete();
    }
    action.getExitSafeFile().delete();
    action.getInfrastructureFailureFile().delete();
  }

  /**
   * Finalizes failed test attempt - renames logs, process test output, etc.
   *
   * @return TestTargetResult protobuffer
   */
  protected final TestTargetResult finalizeFailedTestAttempt(Executor executor,
      TestTargetResult.Builder resultBuilder, TestRunnerAction action, int attemptId,
      FileOutErr outErr, boolean localRun) throws IOException {
    renameOutputs(executor, resultBuilder, action, attemptId, localRun);
    TestTargetResult result = resultBuilder.build();
    processTestOutput(executor, outErr, result, action.getTestName());
    return result;
  }

  private Path pathForExecPath(Executor executor, PathFragment execPath) {
    return executor.getExecRoot().getRelative(execPath);
  }

  /**
   * Performs coverage data post-processing if necessary. Post-processing will
   * unzip data generated by remote executor and then merge all coverage data
   * into the single lcov-based file.
   */
  private void processCoverageData(Executor executor, TestRunnerAction action,
      TestTargetResult.Builder resultBuilder) {
    if (isCoverageMode(action) && pathForExecPath(executor, action.getCoverageData()).exists()) {
      Builder builder = TargetCoverage.newBuilder()
          .setInstrumentation(CoverageInstrumentation.UNSPECIFIED_INSTRUMENTATION)
          .setLcov(textFileFor(executor, pathForExecPath(executor, action.getCoverageData())));
      if (isMicroCoverageMode(action) &&
          pathForExecPath(executor, action.getMicroCoverageData()).exists()) {
        builder.setMicroCoverage(textFileFor(executor, pathForExecPath(
            executor, action.getMicroCoverageData())));
      }
      resultBuilder.setCoverage(builder);
    }
  }

  /**
   * Returns timeout value in seconds that should be used for the given test
   * action.  We always use the "categorical timeouts" which are based on the
   * --test_timeout flag.  A rule picks its timeout but ends up with the same
   * effective value as all other rules in that bucket.
   */
  protected final int getTimeout(TestRunnerAction testAction) {
    return executionOptions.testTimeout.get(testAction.getTestProperties().getTimeout());
  }

  /**
   * Adds the given warning to the "test warnings" with the proper
   * suffix added on the end.
   */
  protected static void addTestWarning(TestTargetResult.Builder builder, TestRunnerAction action,
                                       String warning) {
    String suffix = action.getTestSuffix();
    if (!suffix.isEmpty()) {
      warning = warning + " " + suffix;
    }
    builder.addWarning(TestWarning.newBuilder().setWarningMessage(warning).build());
  }

  /**
   * Transfers the test warnings to the TestTargetResult and then deletes
   * the warnings file.
   *
   * @throws IOException If the warning file could not be read or deleted.
   */
  private static void processWarningsFile(Path path, TestRunnerAction action,
                                   TestTargetResult.Builder builder)
      throws IOException {
    if (!path.exists()) {
      return;
    }

    for (String warning : FileSystemUtils.iterateLinesAsLatin1(path)) {
      if (!warning.isEmpty()) {
        addTestWarning(builder, action, warning);
      }
    }
    path.delete();
  }

  /**
   * Returns true if Blaze needs to parse test.xml files. We avoid parsing it
   * unnecessarily, since test results can potentially consume a large amount
   * of memory.
   */
  private boolean testResultParsingNeeded() {
    return executionOptions.testSummary == TestSummaryFormat.DETAILED;
  }

  /**
   * Parse a test result XML file into a HierarchicalTestResult.
   */
  private HierarchicalTestResult parseTestResult(Path resultFile) {
    InputStream fileStream = null;

    if (!testResultParsingNeeded()) {
      return null;
    }

    try {
      fileStream = resultFile.getInputStream();
      return new TestXmlOutputParser().parseXmlIntoTestResult(fileStream);
    } catch (IOException | TestXmlOutputParserException e) {
      return null;
    } finally {
      if (fileStream != null) {
        try {
          fileStream.close();
       } catch (IOException e) {}
      }
    }
  }

  private static InfrastructureFailureInfo parseInfrastructureFailureFile(Path failureFile)
      throws IOException {
    try (InputStream fis = failureFile.getInputStream()) {
      // We typically avoid reading in remote files, but reading this file shouldn't
      // be too bad: The file is typically small and is only read in exceptional situations.
      List<String> fileContents = CharStreams.readLines(new InputStreamReader(fis));
      if (fileContents.size() < 2) {
        // Not enough lines in the file - return a null InfrastructureFailureInfo, which
        //  will cause a test warning to be added.
        return null;
      }
      String origin = fileContents.get(0);
      String failureCause = fileContents.get(1);

      InfrastructureFailureInfo.Builder failureInfoBuilder =
          InfrastructureFailureInfo.newBuilder();
      failureInfoBuilder.setOrigin(origin);
      failureInfoBuilder.setCause(failureCause);
      return failureInfoBuilder.build();
    }
  }

  /**
   * Executes test using provided spawn action, generates test result and test
   * status artifacts and returns TestTargetResult builder with information about
   * test run. If streamed test output is requested, will also redirect test
   * stdout/stderr to the Blaze stdout.
   */
  protected final TestTargetResult.Builder executeTest(
      TestRunnerAction action, Spawn spawn,
      ActionExecutionContext actionExecutionContext)
      throws IOException, InterruptedException {
    Path testLogPath = action.getTestLog().getPath();
    FileSystemUtils.createDirectoryAndParents(testLogPath.getParentDirectory());
    FileSystemUtils.createDirectoryAndParents(action.getUndeclaredOutputsDir());
    FileSystemUtils.createDirectoryAndParents(action.getUndeclaredOutputsAnnotationsDir());
    FileSystemUtils.createDirectoryAndParents(action.getSplitLogsDir());
    FileSystemUtils.createDirectoryAndParents(action.getTestDiagnosticsDir());
    FileWatcher watcher = null;
    TestLogHelper.FilterTestHeaderOutputStream headerFilter = null;

    TestTargetResult.Builder resultBuilder;
    try {
      if (executionOptions.testOutput.equals(TestOutputFormat.STREAMED) && !spawn.isRemotable()) {
        // Note: code below relies on headerFilter and watcher being non-null
        // simultaneously.
        headerFilter = TestLogHelper.getHeaderFilteringOutputStream(outErr.getOutputStream());
        watcher = new FileWatcher(testLogPath, OutErr.create(headerFilter, headerFilter), false);
        watcher.start();
      }

      // Redirect stdout to the test log file instead of default location. We expect that all test
      // output will be redirected to the stdout by test runner scripts.
      FileOutErr fileOutErr = new FileOutErr(testLogPath, action.getTestStderr());
      resultBuilder = initResultBuilder(action, spawn);
      resultBuilder = runTestProcess(action, spawn, resultBuilder,
          actionExecutionContext.withFileOutErr(fileOutErr));
      appendStderr(fileOutErr.getOutputFile(), fileOutErr.getErrorFile());

      // Collect the information from the infrastructure failure file regardless of the test status.
      // This will allow us to catch tests that re-try on behalf of the infrastructure. Once
      // such tests stop re-trying themselves, the status could be changed to indicate failure.
      if (action.getInfrastructureFailureFile().exists()) {
        InfrastructureFailureInfo failureInfo =
            parseInfrastructureFailureFile(action.getInfrastructureFailureFile());
        if (failureInfo != null) {
          resultBuilder.setStatusDetails(
              String.format("%s failure. Reason: %s",
                            failureInfo.getOrigin(), failureInfo.getCause()))
              .setInfrastructureFailureInfo(failureInfo);
        } else {
          addTestWarning(resultBuilder, action, "Short TEST_INFRASTRUCTURE_FAILURE_FILE. " +
                         "See the Build Encyclopedia for the format of this file.");
        }
      }

      if (!testLogPath.exists()) {
        // Print detailed status message to the test log if log does not exist
        // in order to help user to understand reason behind missing test log.
        FileSystemUtils.writeContent(testLogPath, ISO_8859_1,
            "Blaze: " + (resultBuilder.hasStatusDetails()
                ? resultBuilder.getStatusDetails()
                : ("test exited with error code " + resultBuilder.getExitCode()))
            + "\n");
      }
    } finally {
      if (watcher != null) {
        watcher.stopPumping();
        // The watcher thread might leak if the following call is interrupted.
        // This is a relatively minor issue since the worst it could do is
        // write one additional line from the test.log to the console later on
        // in the build.
        watcher.join();
        if (!headerFilter.foundHeader()) {
          InputStream input = testLogPath.getInputStream();
          try {
            ByteStreams.copy(input, outErr.getOutputStream());
          } finally {
            input.close();
          }
        }
      }
    }

    processWarningsFile(action.getTestWarningsPath(), action, resultBuilder);
    Executor executor = actionExecutionContext.getExecutor();
    resultBuilder.setCombinedOut(textFileFor(executor, action.getTestLog().getPath()));
    if (action.getXmlOutputPath().exists()) {
      resultBuilder.setXml(textFileFor(executor, action.getXmlOutputPath()));
      HierarchicalTestResult testResultParsed = parseTestResult(action.getXmlOutputPath());

      if (testResultParsed != null) {
        resultBuilder.setHierarchicalTestResult(testResultParsed);
      }
    }
    if (action.getSplitLogsPath().exists()) {
      // For logsplitter output log. (http://go/logsplitter)
      resultBuilder.setSplitlogs(textFileFor(executor, action.getSplitLogsPath()));
    }
    if (action.getUndeclaredOutputsZipPath().exists()) {
      resultBuilder.setUndeclaredOutputsZip(textFileFor(
          executor, action.getUndeclaredOutputsZipPath()));
    }
    if (action.getUndeclaredOutputsManifestPath().exists()) {
      resultBuilder.setUndeclaredOutputsManifest(textFileFor(
          executor, action.getUndeclaredOutputsManifestPath()));
    }
    if (action.getUndeclaredOutputsAnnotationsPath().exists()) {
      resultBuilder.setUndeclaredOutputsAnnotations(textFileFor(
          executor, action.getUndeclaredOutputsAnnotationsPath()));
    }
    if (action.getTestDiagnostics().exists()) {
      resultBuilder.setTestDiagnosticsData(textFileFor(
          executor, action.getTestDiagnostics()));
    }

    if (resultBuilder.getStatus() == TestStatus.PASSED) {
      processCoverageData(executor, action, resultBuilder);
    }

    resultBuilder.addAllTag(action.getTestProperties().getTags());
    return resultBuilder;
  }

  protected TestTargetResult.Builder initResultBuilder(TestRunnerAction action, Spawn spawn) {
    TestTargetProperties testProperties = action.getTestProperties();
    TestTargetResult.Builder resultBuilder = TestTargetResult.newBuilder()
        .setName(Label.print(action.getOwner().getLabel())) // maybe action.getTestName()?
        .setLanguage(testProperties.getLanguage())
        .setCommandLine(ShellEscaper.escapeJoinAll(spawn.getArguments()))
        .setSize(TestSize.valueOf(testProperties.getSize().name()))
        .setTimeoutSeconds(getTimeout(action))
        .setUser(UserUtils.getUserName());

    resultBuilder.setStrategy(spawn.isRemotable()
        ? com.google.testing.proto.TestStrategy.REMOTE : (action.isExclusive()
        ? com.google.testing.proto.TestStrategy.LOCAL_SEQUENTIAL
        : com.google.testing.proto.TestStrategy.LOCAL_PARALLEL));
    for (Map.Entry<String, String> var : spawn.getEnvironment().entrySet()) {
      resultBuilder.addEnvironmentVariable(
          Property.newBuilder().setKey(var.getKey()).setValue(var.getValue()));
    }
    if (action.isSharded()) {
      resultBuilder.setShardNumber(action.getShardNum())
          .setTotalShards(action.getExecutionSettings().getTotalShards());
    }
    if (action.getConfiguration().getRunsPerTestForLabel(action.getOwner().getLabel()) > 1) {
      resultBuilder.setRunNumber(action.getRunNumber() + 1);
    }
    return resultBuilder;
  }

  /**
   * In rare cases, the distributor might write something to stderr.
   * Append it to the real test.log.
   */
  private static void appendStderr(Path stdOut, Path stdErr) throws IOException {
    FileStatus stat = stdErr.statNullable();
    OutputStream out = null;
    InputStream in = null;
    if (stat != null) {
      try {
        if (stat.getSize() > 0) {
          if (stdOut.exists()) {
            stdOut.setWritable(true);
          }
          out = stdOut.getOutputStream(true);
          in = stdErr.getInputStream();
          ByteStreams.copy(in, out);
        }
      } finally {
        Closeables.close(out, true);
        Closeables.close(in, true);
        stdErr.delete();
      }
    }
  }

  /**
   * Persists test result protobuffer and posts TestResult message containing
   * final test execution results.
   */
  private void postTestResult(Executor executor, TestRunnerAction action, TestTargetResult result)
      throws IOException {
    OutputStream out = action.getTestStatus().getPath().getOutputStream();
    try {
      result.writeTo(out);
    } finally {
      out.close();
    }
    executor.getEventBus().post(TestResult.createNew(action, result));
  }

  /**
   * Outputs test result to the stdout after test has finished (e.g. for
   * --test_output=all or --test_output=errors). Will also try to group
   * output lines together (up to 10000 lines) so parallel test outputs
   * will not get interleaved.
   */
  private void processTestOutput(Executor executor, FileOutErr outErr, TestTargetResult result,
      String testName) throws IOException {
    Path testOutput =
        executor.getExecRoot().getRelative(result.getCombinedOut().getPathname());
    boolean isPassed = (result.getStatus() == TestStatus.PASSED);
    try {
      if (TestLogHelper.shouldOutputTestLog(executionOptions.testOutput, isPassed)) {
        TestLogHelper.writeTestLog(testOutput, testName, outErr.getOutputStream());
      }
    } finally {
      if (isPassed) {
        executor.getReporter().report(EventKind.PASS, null, testName);
      } else {
        if (result.hasStatusDetails()) {
          executor.getReporter().error(null, testName + ": " + result.getStatusDetails());
        }
        executor.getReporter().report(EventKind.FAIL, null, testName +
            " (see " + testOutput + ")");
      }
    }
  }

  /**
   * Returns a subset of the environment from the current shell.
   *
   * Warning: Since these variables are not part of the
   * configuration's fingerprint, they MUST NOT be used by any rule or
   * action in such a way as to affect the semantics of that build
   * step.
   */
  public Map<String, String> getAdmissibleShellEnvironment(BuildConfiguration config,
      Iterable<String> variables) {
    return getMapping(variables, config.getClientEnv());
  }

  /**
   * For an given environment, returns a subset containing all variables in the given list if they
   * are defined in the given environment.
   */
  @VisibleForTesting
  static Map<String, String> getMapping(Iterable<String> variables,
                                        Map<String, String> environment) {
    Map<String, String> result = new HashMap<>();
    for (String var : variables) {
      if (environment.containsKey(var)) {
        result.put(var, environment.get(var));
      }
    }
    return result;
  }
}
