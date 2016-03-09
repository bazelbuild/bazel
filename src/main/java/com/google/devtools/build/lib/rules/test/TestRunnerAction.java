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

package com.google.devtools.build.lib.rules.test;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.NotifyOnActionCacheHit;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.RunfilesSupplierImpl;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.RunUnder;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.test.TestStatus.TestResultData;
import com.google.devtools.common.options.TriState;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;

import javax.annotation.Nullable;

/**
 * An Action representing a test with the associated environment (runfiles,
 * environment variables, test result, etc). It consumes test executable and
 * runfiles artifacts and produces test result and test status artifacts.
 */
// Not final so that we can mock it in tests.
public class TestRunnerAction extends AbstractAction implements NotifyOnActionCacheHit {

  private static final String GUID = "94857c93-f11c-4cbc-8c1b-e0a281633f9e";

  private final NestedSet<Artifact> runtime;
  private final BuildConfiguration configuration;
  private final Artifact testLog;
  private final Artifact cacheStatus;
  private final PathFragment testWarningsPath;
  private final PathFragment unusedRunfilesLogPath;
  private final PathFragment splitLogsPath;
  private final PathFragment splitLogsDir;
  private final PathFragment undeclaredOutputsDir;
  private final PathFragment undeclaredOutputsZipPath;
  private final PathFragment undeclaredOutputsAnnotationsDir;
  private final PathFragment undeclaredOutputsManifestPath;
  private final PathFragment undeclaredOutputsAnnotationsPath;
  private final PathFragment xmlOutputPath;
  @Nullable
  private final PathFragment testShard;
  private final PathFragment testExitSafe;
  private final PathFragment testStderr;
  private final PathFragment testInfrastructureFailure;
  private final PathFragment baseDir;
  private final String namePrefix;
  private final Artifact coverageData;
  private final Artifact microCoverageData;
  private final TestTargetProperties testProperties;
  private final TestTargetExecutionSettings executionSettings;
  private final int shardNum;
  private final int runNumber;
  private final String workspaceName;
  private final PathFragment shExecutable;

  // Mutable state related to test caching.
  private boolean checkedCaching = false;
  private boolean unconditionalExecution = false;

  private ImmutableMap<String, String> testEnv;

  private static ImmutableList<Artifact> list(Artifact... artifacts) {
    ImmutableList.Builder<Artifact> builder = ImmutableList.builder();
    for (Artifact artifact : artifacts) {
      if (artifact != null) {
        builder.add(artifact);
      }
    }
    return builder.build();
  }

  /**
   * Create new TestRunnerAction instance. Should not be called directly.
   * Use {@link TestActionBuilder} instead.
   *
   * @param shardNum The shard number. Must be 0 if totalShards == 0
   *     (no sharding). Otherwise, must be >= 0 and < totalShards.
   * @param runNumber test run number
   */
  TestRunnerAction(ActionOwner owner,
      Iterable<Artifact> inputs,
      NestedSet<Artifact> runtime,   // Must be a subset of inputs
      Artifact testLog,
      Artifact cacheStatus,
      Artifact coverageArtifact,
      Artifact microCoverageArtifact,
      TestTargetProperties testProperties,
      Map<String, String> extraTestEnv,
      TestTargetExecutionSettings executionSettings,
      int shardNum,
      int runNumber,
      BuildConfiguration configuration,
      String workspaceName) {
    super(owner, inputs,
        // Note that this action only cares about the runfiles, not the mapping.
        new RunfilesSupplierImpl(new PathFragment("runfiles"), executionSettings.getRunfiles()),
        list(testLog, cacheStatus, coverageArtifact, microCoverageArtifact));
    this.runtime = runtime;
    this.configuration = Preconditions.checkNotNull(configuration);
    this.testLog = testLog;
    this.cacheStatus = cacheStatus;
    this.coverageData = coverageArtifact;
    this.microCoverageData = microCoverageArtifact;
    this.shardNum = shardNum;
    this.runNumber = runNumber;
    this.testProperties = Preconditions.checkNotNull(testProperties);
    this.executionSettings = Preconditions.checkNotNull(executionSettings);

    this.baseDir = cacheStatus.getExecPath().getParentDirectory();
    this.namePrefix = FileSystemUtils.removeExtension(cacheStatus.getExecPath().getBaseName());

    int totalShards = executionSettings.getTotalShards();
    Preconditions.checkState((totalShards == 0 && shardNum == 0) ||
                                (totalShards > 0 && 0 <= shardNum && shardNum < totalShards));
    this.testExitSafe = baseDir.getChild(namePrefix + ".exited_prematurely");
    // testShard Path should be set only if sharding is enabled.
    this.testShard = totalShards > 1
        ? baseDir.getChild(namePrefix + ".shard")
        : null;
    this.xmlOutputPath = baseDir.getChild(namePrefix + ".xml");
    this.testWarningsPath = baseDir.getChild(namePrefix + ".warnings");
    this.unusedRunfilesLogPath = baseDir.getChild(namePrefix + ".unused_runfiles_log");
    this.testStderr = baseDir.getChild(namePrefix + ".err");
    this.splitLogsDir = baseDir.getChild(namePrefix + ".raw_splitlogs");
    // See note in {@link #getSplitLogsPath} on the choice of file name.
    this.splitLogsPath = splitLogsDir.getChild("test.splitlogs");
    this.undeclaredOutputsDir = baseDir.getChild(namePrefix + ".outputs");
    this.undeclaredOutputsZipPath = undeclaredOutputsDir.getChild("outputs.zip");
    this.undeclaredOutputsAnnotationsDir = baseDir.getChild(namePrefix + ".outputs_manifest");
    this.undeclaredOutputsManifestPath = undeclaredOutputsAnnotationsDir.getChild("MANIFEST");
    this.undeclaredOutputsAnnotationsPath = undeclaredOutputsAnnotationsDir.getChild("ANNOTATIONS");
    this.testInfrastructureFailure = baseDir.getChild(namePrefix + ".infrastructure_failure");
    this.workspaceName = workspaceName;
    this.shExecutable = configuration.getShExecutable();

    Map<String, String> mergedTestEnv = new HashMap<>(configuration.getTestEnv());
    mergedTestEnv.putAll(extraTestEnv);
    this.testEnv = ImmutableMap.copyOf(mergedTestEnv);
  }

  public BuildConfiguration getConfiguration() {
    return configuration;
  }

  public final PathFragment getBaseDir() {
    return baseDir;
  }

  public final String getNamePrefix() {
    return namePrefix;
  }

  @Override
  public boolean showsOutputUnconditionally() {
    return true;
  }

  @Override
  protected String computeKey() {
    Fingerprint f = new Fingerprint();
    f.addString(GUID);
    f.addStrings(executionSettings.getArgs());
    f.addString(executionSettings.getTestFilter() == null ? "" : executionSettings.getTestFilter());
    RunUnder runUnder = executionSettings.getRunUnder();
    f.addString(runUnder == null ? "" : runUnder.getValue());
    f.addStringMap(getTestEnv());
    f.addString(testProperties.getSize().toString());
    f.addString(testProperties.getTimeout().toString());
    f.addStrings(testProperties.getTags());
    f.addInt(testProperties.isLocal() ? 1 : 0);
    f.addInt(shardNum);
    f.addInt(executionSettings.getTotalShards());
    f.addInt(runNumber);
    f.addInt(configuration.getRunsPerTestForLabel(getOwner().getLabel()));
    f.addInt(configuration.isCodeCoverageEnabled() ? 1 : 0);
    return f.hexDigestAndReset();
  }

  @Override
  public boolean executeUnconditionally() {
    // Note: isVolatile must return true if executeUnconditionally can ever return true
    // for this instance.
    unconditionalExecution = updateExecuteUnconditionallyFromTestStatus();
    checkedCaching = true;
    return unconditionalExecution;
  }

  @Override
  public boolean isVolatile() {
    return true;
  }

  /**
   * Saves cache status to disk.
   */
  public void saveCacheStatus(TestResultData data) throws IOException {
    try (OutputStream out = cacheStatus.getPath().getOutputStream()) {
      data.writeTo(out);
    }
  }

  /**
   * Returns the cache from disk, or null if there is an error.
   */
  @Nullable
  private TestResultData readCacheStatus() {
    try (InputStream in = cacheStatus.getPath().getInputStream()) {
      return TestResultData.parseFrom(in);
    } catch (IOException expected) {

    }
    return null;
  }

  private boolean updateExecuteUnconditionallyFromTestStatus() {
    if (configuration.cacheTestResults() == TriState.NO || testProperties.isExternal()
        || (configuration.cacheTestResults() == TriState.AUTO
            && configuration.getRunsPerTestForLabel(getOwner().getLabel()) > 1)) {
      return true;
    }

    // Test will not be executed unconditionally - check whether test result exists and is
    // valid. If it is, method will return false and we will rely on the dependency checker
    // to make a decision about test execution.
    TestResultData status = readCacheStatus();
    if (status != null) {
      if (!status.getCachable()) {
        return true;
      }

      return (configuration.cacheTestResults() == TriState.AUTO
          && !status.getTestPassed());
    }

    // CacheStatus is an artifact, so if it does not exist, the dependency checker will rebuild
    // it. We can't return "true" here, as it also signals to not accept cached remote results.
    return false;
  }

  /**
   * May only be called after the dependency checked called executeUnconditionally().
   * Returns whether caching has been deemed safe by looking at the previous test run
   * (for local caching). If the previous run is not present, return "true" here, as
   * remote execution caching should be safe.
   */
  public boolean shouldCacheResult() {
    Preconditions.checkState(checkedCaching);
    return !unconditionalExecution;
  }

  @Override
  public void actionCacheHit(Executor executor) {
    checkedCaching = false;
    try {
      executor.getEventBus().post(
          executor.getContext(TestActionContext.class).newCachedTestResult(
              executor.getExecRoot(), this, readCacheStatus()));
    } catch (IOException e) {
      LoggingUtil.logToRemote(Level.WARNING, "Failed creating cached protocol buffer", e);
    }
  }

  @Override
  public ResourceSet estimateResourceConsumption(Executor executor) {
    return ResourceSet.ZERO;
  }

  @Override
  protected String getRawProgressMessage() {
    return "Testing " + getTestName();
  }

  /**
   * Deletes <b>all</b> possible test outputs.
   *
   * TestRunnerAction potentially can create many more non-declared outputs - xml output,
   * coverage data file and logs for failed attempts. All those outputs are uniquely
   * identified by the test log base name with arbitrary prefix and extension.
   */
  @Override
  protected void deleteOutputs(Path execRoot) throws IOException {
    super.deleteOutputs(execRoot);

    // We do not rely on globs, as it causes quadratic behavior in --runs_per_test and test
    // shard count.

    // We also need to remove *.(xml|data|shard|warnings|zip) files if they are present.
    execRoot.getRelative(xmlOutputPath).delete();
    execRoot.getRelative(testWarningsPath).delete();
    execRoot.getRelative(unusedRunfilesLogPath).delete();
    // Note that splitLogsPath points to a file inside the splitLogsDir so
    // it's not necessary to delete it explicitly.
    FileSystemUtils.deleteTree(execRoot.getRelative(splitLogsDir));
    FileSystemUtils.deleteTree(execRoot.getRelative(undeclaredOutputsDir));
    FileSystemUtils.deleteTree(execRoot.getRelative(undeclaredOutputsAnnotationsDir));
    execRoot.getRelative(testStderr).delete();
    execRoot.getRelative(testExitSafe).delete();
    if (testShard != null) {
      execRoot.getRelative(testShard).delete();
    }
    execRoot.getRelative(testInfrastructureFailure).delete();

    // Coverage files use "coverage" instead of "test".
    String coveragePrefix = "coverage" + namePrefix.substring(4);

    // We cannot use coverageData artifact since it may be null. Generate coverage name instead.
    execRoot.getRelative(baseDir.getChild(coveragePrefix + ".dat")).delete();
    // We cannot use microcoverageData artifact since it may be null. Generate filename instead.
    execRoot.getRelative(baseDir.getChild(coveragePrefix + ".micro.dat")).delete();

    // Delete files fetched from remote execution.
    execRoot.getRelative(baseDir.getChild(namePrefix + ".zip")).delete();
    deleteTestAttemptsDirMaybe(execRoot.getRelative(baseDir), namePrefix);
  }

  private void deleteTestAttemptsDirMaybe(Path outputDir, String namePrefix) throws IOException {
    Path testAttemptsDir = outputDir.getChild(namePrefix + "_attempts");
    if (testAttemptsDir.exists()) {
      // Normally we should have used deleteTree(testAttemptsDir). However, if test output is
      // in a FUSE filesystem implemented with the high-level API, there may be .fuse???????
      // entries, which prevent removing the directory.  As a workaround, code below will throw
      // IOException if it will fail to remove something inside testAttemptsDir, but will
      // silently suppress any exceptions when deleting testAttemptsDir itself.
      FileSystemUtils.deleteTreesBelow(testAttemptsDir);
      try {
        testAttemptsDir.delete();
      } catch (IOException e) {
        // Do nothing.
      }
    }
  }

  /**
   * Gets the test name in a user-friendly format.
   * Will generally include the target name and run/shard numbers, if applicable.
   */
  public String getTestName() {
    String suffix = getTestSuffix();
    String label = Label.print(getOwner().getLabel());
    return suffix.isEmpty() ?  label : label + " " + suffix;
  }

  /**
   * Gets the test suffix in a user-friendly format, eg "(shard 1 of 7)".
   * Will include the target name and run/shard numbers, if applicable.
   */
  public String getTestSuffix() {
    int totalShards = executionSettings.getTotalShards();
    // Use a 1-based index for user friendliness.
    int runsPerTest = configuration.getRunsPerTestForLabel(getOwner().getLabel());
    if (totalShards > 1 && runsPerTest > 1) {
      return String.format("(shard %d of %d, run %d of %d)", shardNum + 1, totalShards,
          runNumber + 1, runsPerTest);
    } else if (totalShards > 1) {
      return String.format("(shard %d of %d)", shardNum + 1, totalShards);
    } else if (runsPerTest > 1) {
      return String.format("(run %d of %d)", runNumber + 1, runsPerTest);
    } else {
      return "";
    }
  }

  public Artifact getTestLog() {
    return testLog;
  }

  /**
   * Returns all environment variables which must be set in order to run this test.
   */
  public Map<String, String> getTestEnv() {
    return testEnv;
  }

  public ResolvedPaths resolve(Path execRoot) {
    return new ResolvedPaths(execRoot);
  }

  public Artifact getCacheStatusArtifact() {
    return cacheStatus;
  }

  public PathFragment getTestWarningsPath() {
    return testWarningsPath;
  }

  public PathFragment getUnusedRunfilesLogPath() {
    return unusedRunfilesLogPath;
  }

  public PathFragment getSplitLogsPath() {
    return splitLogsPath;
  }

  /**
   * @return path to the optional zip file of undeclared test outputs.
   */
  public PathFragment getUndeclaredOutputsZipPath() {
    return undeclaredOutputsZipPath;
  }

  /**
   * @return path to the undeclared output manifest file.
   */
  public PathFragment getUndeclaredOutputsManifestPath() {
    return undeclaredOutputsManifestPath;
  }

  /**
   * @return path to the undeclared output annotations file.
   */
  public PathFragment getUndeclaredOutputsAnnotationsPath() {
    return undeclaredOutputsAnnotationsPath;
  }

  public PathFragment getTestShard() {
    return testShard;
  }

  public PathFragment getExitSafeFile() {
    return testExitSafe;
  }

  public PathFragment getInfrastructureFailureFile() {
    return testInfrastructureFailure;
  }

  /**
   * @return path to the optionally created XML output file created by the test.
   */
  public PathFragment getXmlOutputPath() {
    return xmlOutputPath;
  }

  /**
   * @return coverage data artifact or null if code coverage was not requested.
   */
  @Nullable public Artifact getCoverageData() {
    return coverageData;
  }

  /**
   * @return microcoverage data artifact or null if code coverage was not requested.
   */
  @Nullable public Artifact getMicroCoverageData() {
    return microCoverageData;
  }

  public TestTargetProperties getTestProperties() {
    return testProperties;
  }

  public TestTargetExecutionSettings getExecutionSettings() {
    return executionSettings;
  }

  public boolean isSharded() {
    return testShard != null;
  }

  /**
   * @return the shard number for this action.
   *     If getTotalShards() > 0, must be >= 0 and < getTotalShards().
   *     Otherwise, must be 0.
   */
  public int getShardNum() {
    return shardNum;
  }

  /**
   * @return run number.
   */
  public int getRunNumber() {
    return runNumber;
  }

  /**
   * @return the workspace name.
   */
  public String getRunfilesPrefix() {
    return workspaceName;
  }

  @Override
  public Artifact getPrimaryOutput() {
    return testLog;
  }

  @Override
  public void execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    TestActionContext context =
        actionExecutionContext.getExecutor().getContext(TestActionContext.class);
    try {
      context.exec(this, actionExecutionContext);
    } catch (ExecException e) {
      throw e.toActionExecutionException(this);
    } finally {
      checkedCaching = false;
    }
  }

  @Override
  public String getMnemonic() {
    return "TestRunner";
  }

  @Override
  public ImmutableSet<Artifact> getMandatoryOutputs() {
    return getOutputs();
  }

  public Artifact getRuntimeArtifact(String basename) throws ExecException {
    for (Artifact runtimeArtifact : runtime) {
      if (runtimeArtifact.getExecPath().getBaseName().equals(basename)) {
        return runtimeArtifact;
      }
    }

    throw new UserExecException("'" + basename + "' not found in test runtime");
  }

  public PathFragment getShExecutable() {
    return shExecutable;
  }

  /**
   * The same set of paths as the parent test action, resolved against a given exec root.
   */
  public final class ResolvedPaths {
    private final Path execRoot;

    ResolvedPaths(Path execRoot) {
      this.execRoot = Preconditions.checkNotNull(execRoot);
    }

    private Path getPath(PathFragment relativePath) {
      return execRoot.getRelative(relativePath);
    }

    public final Path getBaseDir() {
      return getPath(baseDir);
    }

    /**
     * In rare cases, error messages will be printed to stderr instead of stdout. The test action is
     * responsible for appending anything in the stderr file to the real test.log.
     */
    public Path getTestStderr() {
      return getPath(testStderr);
    }

    public Path getTestWarningsPath() {
      return getPath(testWarningsPath);
    }

    public Path getSplitLogsPath() {
      return getPath(splitLogsPath);
    }

    public Path getUnusedRunfilesLogPath() {
      return getPath(unusedRunfilesLogPath);
    }

    /**
     * @return path to the directory containing the split logs (raw and proto file).
     */
    public Path getSplitLogsDir() {
      return getPath(splitLogsDir);
    }

    /**
     * @return path to the optional zip file of undeclared test outputs.
     */
    public Path getUndeclaredOutputsZipPath() {
      return getPath(undeclaredOutputsZipPath);
    }

    /**
     * @return path to the directory to hold undeclared test outputs.
     */
    public Path getUndeclaredOutputsDir() {
      return getPath(undeclaredOutputsDir);
    }

    /**
     * @return path to the directory to hold undeclared output annotations parts.
     */
    public Path getUndeclaredOutputsAnnotationsDir() {
      return getPath(undeclaredOutputsAnnotationsDir);
    }

    /**
     * @return path to the undeclared output manifest file.
     */
    public Path getUndeclaredOutputsManifestPath() {
      return getPath(undeclaredOutputsManifestPath);
    }

    /**
     * @return path to the undeclared output annotations file.
     */
    public Path getUndeclaredOutputsAnnotationsPath() {
      return getPath(undeclaredOutputsAnnotationsPath);
    }

    @Nullable
    public Path getTestShard() {
      return testShard == null ? null : getPath(testShard);
    }

    public Path getExitSafeFile() {
      return getPath(testExitSafe);
    }

    public Path getInfrastructureFailureFile() {
      return getPath(testInfrastructureFailure);
    }

    /**
     * @return path to the optionally created XML output file created by the test.
     */
    public Path getXmlOutputPath() {
      return getPath(xmlOutputPath);
    }
  }
}
