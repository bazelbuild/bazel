// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.exec;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.io.ByteStreams;
import com.google.common.io.Closeables;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.config.BinTools;
import com.google.devtools.build.lib.analysis.test.TestActionContext;
import com.google.devtools.build.lib.analysis.test.TestResult;
import com.google.devtools.build.lib.analysis.test.TestRunnerAction;
import com.google.devtools.build.lib.analysis.test.TestTargetExecutionSettings;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.io.FileWatcher;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.test.TestStatus.TestCase;
import com.google.devtools.common.options.Converters.RangeConverter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.OptionsParsingException;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.time.Duration;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/** A strategy for executing a {@link TestRunnerAction}. */
public abstract class TestStrategy implements TestActionContext {
  public static final String TEST_SETUP_BASENAME = "test-setup.sh";

  /**
   * Ensures that all directories used to run test are in the correct state and their content will
   * not result in stale files.
   */
  protected void prepareFileSystem(
      TestRunnerAction testAction, Path tmpDir, Path coverageDir, Path workingDirectory)
      throws IOException {
    if (testAction.isCoverageMode()) {
      recreateDirectory(coverageDir);
    }
    recreateDirectory(tmpDir);
    FileSystemUtils.createDirectoryAndParents(workingDirectory);
  }

  /** Removes directory if it exists and recreates it. */
  protected void recreateDirectory(Path directory) throws IOException {
    FileSystemUtils.deleteTree(directory);
    FileSystemUtils.createDirectoryAndParents(directory);
  }

  /** Converter for the --flaky_test_attempts option. */
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
    NONE; // Do not print summary.

    /** Converts to {@link TestSummaryFormat}. */
    public static class Converter extends EnumConverter<TestSummaryFormat> {
      public Converter() {
        super(TestSummaryFormat.class, "test summary");
      }
    }
  }

  public static final PathFragment TEST_TMP_ROOT = PathFragment.create("_tmp");

  // Used for generating unique temporary directory names. Contains the next numeric index for every
  // executable base name.
  private final Map<String, Integer> tmpIndex = new HashMap<>();
  protected final ExecutionOptions executionOptions;
  protected final BinTools binTools;

  public TestStrategy(ExecutionOptions executionOptions, BinTools binTools) {
    this.executionOptions = executionOptions;
    this.binTools = binTools;
  }

  @Override
  public abstract List<SpawnResult> exec(
      TestRunnerAction action, ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException;

  /**
   * Generates a command line to run for the test action, taking into account coverage and {@code
   * --run_under} settings.
   *
   * @param coverageScript a script interjected between setup script and rest of command line to
   *     collect coverage data. If this is an empty string, it is ignored.
   * @param testAction The test action.
   * @return the command line as string list.
   * @throws ExecException 
   */
  protected ImmutableList<String> getArgs(String coverageScript, TestRunnerAction testAction)
      throws ExecException {
    List<String> args = Lists.newArrayList();
    // TODO(ulfjack): This is incorrect for remote execution, where we need to consider the target
    // configuration, not the machine Bazel happens to run on. Change this to something like:
    // testAction.getConfiguration().getTargetOS() == OS.WINDOWS
    if (OS.getCurrent() == OS.WINDOWS) {
      args.add(testAction.getShExecutable().getPathString());
      args.add("-c");
      args.add("$0 $*");
    }

    Artifact testSetup = testAction.getRuntimeArtifact(TEST_SETUP_BASENAME);
    args.add(testSetup.getExecPath().getCallablePathString());

    if (testAction.isCoverageMode()) {
      args.add(coverageScript);
    }

    TestTargetExecutionSettings execSettings = testAction.getExecutionSettings();

    // Insert the command prefix specified by the "--run_under=<command-prefix>" option, if any.
    if (execSettings.getRunUnder() != null) {
      addRunUnderArgs(testAction, args);
    }

    // Execute the test using the alias in the runfiles tree, as mandated by the Test Encyclopedia.
    args.add(execSettings.getExecutable().getRootRelativePath().getCallablePathString());
    try {
      Iterables.addAll(args, execSettings.getArgs().arguments());
    } catch (CommandLineExpansionException e) {
      throw new UserExecException(e);
    }
    return ImmutableList.copyOf(args);
  }

  private static void addRunUnderArgs(TestRunnerAction testAction, List<String> args) {
    TestTargetExecutionSettings execSettings = testAction.getExecutionSettings();
    if (execSettings.getRunUnderExecutable() != null) {
      args.add(execSettings.getRunUnderExecutable().getRootRelativePath().getCallablePathString());
    } else {
      String command = execSettings.getRunUnder().getCommand();
      // --run_under commands that do not contain '/' are either shell built-ins or need to be
      // located on the PATH env, so we wrap them in a shell invocation. Note that we shell tokenize
      // the --run_under parameter and getCommand only returns the first such token.
      boolean needsShell = !command.contains("/");
      if (needsShell) {
        args.add(testAction.getConfiguration().getShellExecutable().getPathString());
        args.add("-c");
        args.add("\"$@\"");
        args.add("/bin/sh"); // Sets $0.
      }
      args.add(command);
    }
    args.addAll(testAction.getExecutionSettings().getRunUnder().getOptions());
  }

  /**
   * Returns the number of attempts specific test action can be retried.
   *
   * <p>For rules with "flaky = 1" attribute, this method will return 3 unless --flaky_test_attempts
   * option is given and specifies another value.
   */
  @VisibleForTesting /* protected */
  public int getTestAttempts(TestRunnerAction action) {
    return action.getTestProperties().isFlaky()
        ? getTestAttemptsForFlakyTest()
        : getTestAttempts(/*defaultTestAttempts=*/ 1);
  }

  public int getTestAttemptsForFlakyTest() {
    return getTestAttempts(/*defaultTestAttempts=*/ 3);
  }

  private int getTestAttempts(int defaultTestAttempts) {
    return executionOptions.testAttempts == -1
        ? defaultTestAttempts
        : executionOptions.testAttempts;
  }

  /**
   * Returns timeout value in seconds that should be used for the given test action. We always use
   * the "categorical timeouts" which are based on the --test_timeout flag. A rule picks its timeout
   * but ends up with the same effective value as all other rules in that bucket.
   */
  protected final Duration getTimeout(TestRunnerAction testAction) {
    return executionOptions.testTimeout.get(testAction.getTestProperties().getTimeout());
  }

  /*
   * Finalize test run: persist the result, and post on the event bus.
   */
  protected void postTestResult(ActionExecutionContext actionExecutionContext, TestResult result)
      throws IOException {
    result.getTestAction().saveCacheStatus(result.getData());
    actionExecutionContext.getEventBus().post(result);
  }

  /**
   * Returns a unique name for a temporary directory a test could use.
   *
   * <p>Since each test within single Blaze run must have a unique TEST_TMPDIR, we will use rule
   * name and a unique (within single Blaze request) number to generate directory name.
   *
   * <p>This does not create the directory.
   */
  protected String getTmpDirName(PathFragment execPath) {
    String basename = execPath.getBaseName();

    synchronized (tmpIndex) {
      int index = tmpIndex.containsKey(basename) ? tmpIndex.get(basename) : 1;
      tmpIndex.put(basename, index + 1);
      return basename + "_" + index;
    }
  }

  protected String getTmpDirName(PathFragment execPath, int shard, int run) {
    Fingerprint digest = new Fingerprint();
    digest.addPath(execPath);
    digest.addInt(shard);
    digest.addInt(run);
    return digest.hexDigestAndReset();
  }

  /** Parse a test result XML file into a {@link TestCase}. */
  @Nullable
  protected TestCase parseTestResult(Path resultFile) {
    /* xml files. We avoid parsing it unnecessarily, since test results can potentially consume
    a large amount of memory. */
    if (executionOptions.testSummary != TestSummaryFormat.DETAILED) {
      return null;
    }

    try (InputStream fileStream = resultFile.getInputStream()) {
      return new TestXmlOutputParser().parseXmlIntoTestResult(fileStream);
    } catch (IOException | TestXmlOutputParserException e) {
      return null;
    }
  }

  /**
   * Returns a temporary directory for all tests in a workspace to use. Individual tests should
   * create child directories to actually use.
   *
   * <p>This either dynamically generates a directory name or uses the directory specified by
   * --test_tmpdir. This does not create the directory.
   */
  public static Path getTmpRoot(Path workspace, Path execRoot, ExecutionOptions executionOptions) {
    return executionOptions.testTmpDir != null
        ? workspace.getRelative(executionOptions.testTmpDir).getRelative(TEST_TMP_ROOT)
        : execRoot.getRelative(TEST_TMP_ROOT);
  }

  /**
   * For an given environment, returns a subset containing all variables in the given list if they
   * are defined in the given environment.
   */
  @VisibleForTesting
  public static Map<String, String> getMapping(
      Iterable<String> variables, Map<String, String> environment) {
    Map<String, String> result = new HashMap<>();
    for (String var : variables) {
      if (environment.containsKey(var)) {
        result.put(var, environment.get(var));
      }
    }
    return result;
  }

  /**
   * Returns the runfiles directory associated with the test executable, creating/updating it if
   * necessary and --build_runfile_links is specified.
   */
  protected static Path getLocalRunfilesDirectory(
      TestRunnerAction testAction,
      ActionExecutionContext actionExecutionContext,
      BinTools binTools,
      ImmutableMap<String, String> shellEnvironment,
      boolean enableRunfiles)
      throws ExecException, InterruptedException {
    TestTargetExecutionSettings execSettings = testAction.getExecutionSettings();
    Path runfilesDir = execSettings.getRunfilesDir();

    // If the symlink farm is already created then return the existing directory. If not we
    // need to explicitly build it. This can happen when --nobuild_runfile_links is supplied
    // as a flag to the build.
    if (execSettings.getRunfilesSymlinksCreated()) {
      return runfilesDir;
    }

    // Synchronize runfiles tree generation on the runfiles manifest artifact.
    // This is necessary, because we might end up with multiple test runner actions
    // trying to generate same runfiles tree in case of --runs_per_test > 1 or
    // local test sharding.
    long startTime = Profiler.nanoTimeMaybe();
    synchronized (execSettings.getInputManifest()) {
      Profiler.instance().logSimpleTask(startTime, ProfilerTask.WAIT, testAction);
      updateLocalRunfilesDirectory(
          testAction,
          runfilesDir,
          actionExecutionContext,
          binTools,
          shellEnvironment,
          enableRunfiles);
    }

    return runfilesDir;
  }

  /**
   * Ensure the runfiles tree exists and is consistent with the TestAction's manifest
   * ($0.runfiles_manifest), bringing it into consistency if not. The contents of the output file
   * $0.runfiles/MANIFEST, if it exists, are used a proxy for the set of existing symlinks, to avoid
   * the need for recursion.
   */
  private static void updateLocalRunfilesDirectory(
      TestRunnerAction testAction,
      Path runfilesDir,
      ActionExecutionContext actionExecutionContext,
      BinTools binTools,
      ImmutableMap<String, String> shellEnvironment,
      boolean enableRunfiles)
      throws ExecException, InterruptedException {
    TestTargetExecutionSettings execSettings = testAction.getExecutionSettings();
    Path outputManifest = runfilesDir.getRelative("MANIFEST");
    try {
      // Avoid rebuilding the runfiles directory if the manifest in it matches the input manifest,
      // implying the symlinks exist and are already up to date. If the output manifest is a
      // symbolic link, it is likely a symbolic link to the input manifest, so we cannot trust it as
      // an up-to-date check.
      if (!outputManifest.isSymbolicLink()
          && Arrays.equals(
              outputManifest.getDigest(), execSettings.getInputManifest().getPath().getDigest())) {
        return;
      }
    } catch (IOException e1) {
      // Ignore it - we will just try to create runfiles directory.
    }

    actionExecutionContext
        .getEventHandler()
        .handle(
            Event.progress(
                "Building runfiles directory for '"
                    + execSettings.getExecutable().prettyPrint()
                    + "'."));

    new SymlinkTreeHelper(execSettings.getInputManifest().getPath(), runfilesDir, false)
        .createSymlinks(
            testAction,
            actionExecutionContext,
            binTools,
            shellEnvironment,
            execSettings.getInputManifest(),
            enableRunfiles);

    actionExecutionContext.getEventHandler()
        .handle(Event.progress(testAction.getProgressMessage()));
  }

  /** In rare cases, we might write something to stderr. Append it to the real test.log. */
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

  /** Implements the --test_output=streamed option. */
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
