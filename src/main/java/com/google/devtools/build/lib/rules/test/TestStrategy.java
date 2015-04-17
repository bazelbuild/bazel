// Copyright 2014 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.rules.test;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.ByteStreams;
import com.google.common.io.Closeables;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.analysis.config.BinTools;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.SymlinkTreeHelper;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.runtime.BlazeServerStartupOptions;
import com.google.devtools.build.lib.util.io.FileWatcher;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.test.TestStatus.TestCase;
import com.google.devtools.common.options.Converters.RangeConverter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.OptionsClassProvider;
import com.google.devtools.common.options.OptionsParsingException;

import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import javax.annotation.Nullable;

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
    SUMMARY, // Provide summary output only.
    ERRORS, // Print output from failed tests to the stderr after the test failure.
    ALL, // Print output from all tests to the stderr after the test completion.
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
    SHORT, // Print information only about tests.
    TERSE, // Like "SHORT", but even shorter: Do not print PASSED tests.
    DETAILED, // Print information only about failed test cases.
    NONE; // Do not print summary.

    /**
     * Converts to {@link TestSummaryFormat}.
     */
    public static class Converter extends EnumConverter<TestSummaryFormat> {
      public Converter() {
        super(TestSummaryFormat.class, "test summary");
      }
    }
  }

  public static final PathFragment TEST_TMP_ROOT = new PathFragment("_tmp");

  // Used for selecting subset of testcase / testmethods.
  private static final String TEST_BRIDGE_TEST_FILTER_ENV = "TESTBRIDGE_TEST_ONLY";

  private final boolean statusServerRunning;
  protected final ImmutableMap<String, String> clientEnv;
  protected final ExecutionOptions executionOptions;
  protected final BinTools binTools;

  public TestStrategy(OptionsClassProvider requestOptionsProvider,
      OptionsClassProvider startupOptionsProvider, BinTools binTools,
      Map<String, String> clientEnv) {
    this.executionOptions = requestOptionsProvider.getOptions(ExecutionOptions.class);
    this.binTools = binTools;
    this.clientEnv = ImmutableMap.copyOf(clientEnv);
    BlazeServerStartupOptions startupOptions =
        startupOptionsProvider.getOptions(BlazeServerStartupOptions.class);
    statusServerRunning = startupOptions != null && startupOptions.useWebStatusServer > 0;
  }

  @Override
  public abstract void exec(TestRunnerAction action, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException;

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
   * Returns mutable map of default testing shell environment. By itself it is incomplete and is
   * modified further by the specific test strategy implementations (mostly due to the fact that
   * environments used locally and remotely are different).
   */
  protected Map<String, String> getDefaultTestEnvironment(TestRunnerAction action) {
    Map<String, String> env = new HashMap<>();

    env.putAll(action.getConfiguration().getDefaultShellEnvironment());
    env.remove("LANG");
    env.put("TZ", "UTC");
    env.put("TEST_SIZE", action.getTestProperties().getSize().toString());
    env.put("TEST_TIMEOUT", Integer.toString(getTimeout(action)));

    if (action.isSharded()) {
      env.put("TEST_SHARD_INDEX", Integer.toString(action.getShardNum()));
      env.put("TEST_TOTAL_SHARDS",
          Integer.toString(action.getExecutionSettings().getTotalShards()));
    }

    // When we run test multiple times, set different TEST_RANDOM_SEED values for each run.
    if (action.getConfiguration().getRunsPerTestForLabel(action.getOwner().getLabel()) > 1) {
      env.put("TEST_RANDOM_SEED", Integer.toString(action.getRunNumber() + 1));
    }

    String testFilter = action.getExecutionSettings().getTestFilter();
    if (testFilter != null) {
      env.put(TEST_BRIDGE_TEST_FILTER_ENV, testFilter);
    }

    return env;
  }

  /**
   * Returns the number of attempts specific test action can be retried.
   *
   * <p>For rules with "flaky = 1" attribute, this method will return 3 unless --flaky_test_attempts
   * option is given and specifies another value.
   */
  @VisibleForTesting /* protected */
  public int getTestAttempts(TestRunnerAction action) {
    if (executionOptions.testAttempts == -1) {
      return action.getTestProperties().isFlaky() ? 3 : 1;
    } else {
      return executionOptions.testAttempts;
    }
  }

  /**
   * Returns timeout value in seconds that should be used for the given test action. We always use
   * the "categorical timeouts" which are based on the --test_timeout flag. A rule picks its timeout
   * but ends up with the same effective value as all other rules in that bucket.
   */
  protected final int getTimeout(TestRunnerAction testAction) {
    return executionOptions.testTimeout.get(testAction.getTestProperties().getTimeout());
  }

  /**
   * Returns a subset of the environment from the current shell.
   *
   * <p>Warning: Since these variables are not part of the configuration's fingerprint, they
   * MUST NOT be used by any rule or action in such a way as to affect the semantics of that
   * build step.
   */
  public Map<String, String> getAdmissibleShellEnvironment(BuildConfiguration config,
      Iterable<String> variables) {
    return getMapping(variables, clientEnv);
  }

  /*
   * Finalize test run: persist the result, and post on the event bus.
   */
  protected void postTestResult(Executor executor, TestResult result) throws IOException {
    result.getTestAction().saveCacheStatus(result.getData());
    executor.getEventBus().post(result);
  }

  /**
   * Parse a test result XML file into a {@link TestCase}.
   */
  @Nullable
  protected TestCase parseTestResult(Path resultFile) {
    /* xml files. We avoid parsing it unnecessarily, since test results can potentially consume
       a large amount of memory. */
    if (executionOptions.testSummary != TestSummaryFormat.DETAILED && !statusServerRunning) {
      return null;
    }

    try (InputStream fileStream = resultFile.getInputStream()) {
      return new TestXmlOutputParser().parseXmlIntoTestResult(fileStream);
    } catch (IOException | TestXmlOutputParserException e) {
      return null;
    }
  }

  /**
   * For an given environment, returns a subset containing all variables in the given list if they
   * are defined in the given environment.
   */
  @VisibleForTesting
  public static Map<String, String> getMapping(Iterable<String> variables,
      Map<String, String> environment) {
    Map<String, String> result = new HashMap<>();
    for (String var : variables) {
      if (environment.containsKey(var)) {
        result.put(var, environment.get(var));
      }
    }
    return result;
  }

  /**
   * Returns the runfiles directory associated with the test executable,
   * creating/updating it if necessary and --build_runfile_links is specified.
   */
  protected static Path getLocalRunfilesDirectory(TestRunnerAction testAction,
      ActionExecutionContext actionExecutionContext, BinTools binTools) throws ExecException,
      InterruptedException {
    TestTargetExecutionSettings execSettings = testAction.getExecutionSettings();

    // --nobuild_runfile_links disables runfiles generation only for C++ rules.
    // In that case, getManifest returns the .runfiles_manifest (input) file,
    // not the MANIFEST output file of the build-runfiles action. So the
    // extension ".runfiles_manifest" indicates no runfiles tree.
    if (!execSettings.getManifest().equals(execSettings.getInputManifest())) {
      return execSettings.getManifest().getPath().getParentDirectory();
    }

    // We might need to build runfiles tree now, since it was not created yet
    // local testing is needed.
    Path program = execSettings.getExecutable().getPath();
    Path runfilesDir = program.getParentDirectory().getChild(program.getBaseName() + ".runfiles");

    // Synchronize runfiles tree generation on the runfiles manifest artifact.
    // This is necessary, because we might end up with multiple test runner actions
    // trying to generate same runfiles tree in case of --runs_per_test > 1 or
    // local test sharding.
    long startTime = Profiler.nanoTimeMaybe();
    synchronized (execSettings.getManifest()) {
      Profiler.instance().logSimpleTask(startTime, ProfilerTask.WAIT, testAction);
      updateLocalRunfilesDirectory(testAction, runfilesDir, actionExecutionContext, binTools);
    }

    return runfilesDir;
  }

  /**
   * Ensure the runfiles tree exists and is consistent with the TestAction's manifest
   * ($0.runfiles_manifest), bringing it into consistency if not. The contents of the output file
   * $0.runfiles/MANIFEST, if it exists, are used a proxy for the set of existing symlinks, to avoid
   * the need for recursion.
   */
  private static void updateLocalRunfilesDirectory(TestRunnerAction testAction, Path runfilesDir,
      ActionExecutionContext actionExecutionContext, BinTools binTools) throws ExecException,
      InterruptedException {
    Executor executor = actionExecutionContext.getExecutor();

    TestTargetExecutionSettings execSettings = testAction.getExecutionSettings();
    try {
      if (Arrays.equals(runfilesDir.getRelative("MANIFEST").getMD5Digest(),
          execSettings.getManifest().getPath().getMD5Digest())) {
        return;
      }
    } catch (IOException e1) {
      // Ignore it - we will just try to create runfiles directory.
    }

    executor.getEventHandler().handle(Event.progress(
        "Building runfiles directory for '" + execSettings.getExecutable().prettyPrint() + "'."));

    new SymlinkTreeHelper(execSettings.getManifest().getExecPath(),
        runfilesDir.relativeTo(executor.getExecRoot()), /* filesetTree= */ false)
        .createSymlinks(testAction, actionExecutionContext, binTools);

    executor.getEventHandler().handle(Event.progress(testAction.getProgressMessage()));
  }

  /**
   * In rare cases, we might write something to stderr. Append it to the real test.log.
   */
  protected static void appendStderr(Path stdOut, Path stdErr) throws IOException {
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
   * Implements the --test_output=streamed option.
   */
  protected static class StreamedTestOutput implements Closeable {
    private final TestLogHelper.FilterTestHeaderOutputStream headerFilter;
    private final FileWatcher watcher;
    private final Path testLogPath;
    private final OutErr outErr;

    public StreamedTestOutput(OutErr outErr, Path testLogPath) throws IOException {
      this.testLogPath = testLogPath;
      this.outErr = outErr;
      this.headerFilter = TestLogHelper.getHeaderFilteringOutputStream(outErr.getOutputStream());
      this.watcher = new FileWatcher(testLogPath, OutErr.create(headerFilter, headerFilter), false);
      watcher.start();
    }

    @Override
    public void close() throws IOException {
      watcher.stopPumping();
      try {
        // The watcher thread might leak if the following call is interrupted.
        // This is a relatively minor issue since the worst it could do is
        // write one additional line from the test.log to the console later on
        // in the build.
        watcher.join();
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }
      if (!headerFilter.foundHeader()) {
        try (InputStream input = testLogPath.getInputStream()) {
          ByteStreams.copy(input, outErr.getOutputStream());
        }
      }
    }
  }

}
