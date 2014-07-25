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

package com.google.devtools.build.lib.view.test;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.MayStream;
import com.google.devtools.build.lib.actions.MiddlemanFactory;
import com.google.devtools.build.lib.actions.NotifyOnActionCacheHit;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SuppressNoBuildAttemptError;
import com.google.devtools.build.lib.actions.TestMiddlemanObserver;
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.pkgcache.PackageUpToDateChecker;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.actions.ConfigurationAction;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.lib.view.config.RunUnder;
import com.google.devtools.build.lib.view.test.TestStatus.TestResultData;
import com.google.devtools.common.options.TriState;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Collection;
import java.util.logging.Level;

import javax.annotation.Nullable;

/**
 * An Action representing a test with the associated environment (runfiles,
 * environment variables, test result, etc). It consumes test executable and
 * runfiles artifacts and produces test result and test status artifacts.
 */
// Not final so that we can mock it in tests.
public class TestRunnerAction extends ConfigurationAction
    implements NotifyOnActionCacheHit, SuppressNoBuildAttemptError, MayStream {

  private static final String GUID = "94857c93-f11c-4cbc-8c1b-e0a281633f9e";

  private final Artifact testLog;
  private final Artifact cacheStatus;
  private final Path testWarningsPath;
  private final Path splitLogsPath;
  private final Path splitLogsDir;
  private final Path undeclaredOutputsDir;
  private final Path undeclaredOutputsZipPath;
  private final Path undeclaredOutputsAnnotationsDir;
  private final Path undeclaredOutputsManifestPath;
  private final Path undeclaredOutputsAnnotationsPath;
  private final Path xmlOutputPath;
  private final Path testShard;
  private final Path testExitSafe;
  private final Path testStderr;
  private final Path testInfrastructureFailure;
  private final Path baseDir;
  private final String namePrefix;
  private final PathFragment coverageData;
  private final PathFragment microCoverageData;
  private final TestTargetProperties testProperties;
  private final TestTargetExecutionSettings executionSettings;
  private final int shardNum;
  private final int runNumber;
  private Artifact schedulingMiddleman = null;
  private Action schedulingMiddlemanAction = null;

  // Mutable state related to test caching.
  private boolean checkedCaching = false;
  private boolean unconditionalExecution = false;

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
      Artifact testLog,
      Artifact cacheStatus,
      PathFragment coverageData,
      PathFragment microCoverageData,
      TestTargetProperties testProperties,
      TestTargetExecutionSettings executionSettings,
      int shardNum,
      int runNumber,
      BuildConfiguration configuration) {
    super(owner, inputs,
        ImmutableList.of(testLog, cacheStatus), configuration);
    Preconditions.checkNotNull(testProperties);
    Preconditions.checkNotNull(executionSettings);
    this.testLog = testLog;
    this.cacheStatus = cacheStatus;
    this.coverageData = coverageData;
    this.microCoverageData = microCoverageData;
    this.shardNum = shardNum;
    this.runNumber = runNumber;
    this.testProperties = testProperties;
    this.executionSettings = executionSettings;

    this.baseDir = configuration.getExecRoot().getRelative(cacheStatus.getExecPath())
        .getParentDirectory();
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
  }

  @Override
  public boolean shouldShowOutput(ErrorEventListener listener) {
    return true;
  }

  public final Path getBaseDir() {
    return baseDir;
  }

  public final String getNamePrefix() {
    return namePrefix;
  }

  /**
   * Sets scheduling dependencies, if any. Used to enforce execution
   * order for exclusive tests (or when exclusive test strategy is used).
   *
   * @param artifactFactory reference to the artifact factory
   * @param middlemanFactory reference to the middleman factory
   * @param dependencies collection of artifacts that must be scheduled prior to
   *                     this action. Can be empty or null.
   * @param observer this method must notify the observer about updates to the
   *                 test scheduling middleman.
   *
   * @return An artifact-action pair so that the build view can patch up the action graph by
   * linking up the generated artifact and action.
   */
  public Pair<Artifact, Action> setSchedulingDependencies(
      ArtifactFactory artifactFactory,
      MiddlemanFactory middlemanFactory,
      Collection<Artifact> dependencies,
      TestMiddlemanObserver observer) {
    if (schedulingMiddleman != null) {
      // Remove old exclusive test scheduling middleman from the factory since
      // it is no longer used.
      // Removing the scheduling middleman from the forward graph (via the observer) will lead
      // to a mis-count of this action's inputs unless we reset the schedulingMiddleman first.
      Artifact oldSchedulingMiddleman = schedulingMiddleman;
      schedulingMiddleman = null;
      artifactFactory.removeSchedulingMiddleman(oldSchedulingMiddleman);
      // In this call, the forward graph removes the artifact oldSchedulingMiddleman from the set
      // of inputs to the current object, the TestRunnerAction. As part of that process, the
      // DependentAction associated to this TestRunnerAction must be reset(), since the inputs to
      // this TestRunnerAction are changing. However, the inputs to this TestRunnerAction don't
      // officially change until schedulingMiddleman is set to null, which is why
      // it is done above.
      observer.remove(this, oldSchedulingMiddleman, schedulingMiddlemanAction);
      schedulingMiddlemanAction = null;
    }
    if (dependencies != null && !dependencies.isEmpty()) {
      Pair<Artifact, Action> result = middlemanFactory.createSchedulingMiddleman(
          getOwner(), executionSettings.getExecutable().getExecPathString(),
          "exclusive_test_" + runNumber + "_" + shardNum, dependencies,
          getConfiguration().getMiddlemanDirectory());
      schedulingMiddleman = result.getFirst();
      schedulingMiddlemanAction = result.getSecond();
      return result;
    } else {
      return null;
    }
  }

  @Override
  public Iterable<Artifact> getInputs() {
    if (schedulingMiddleman != null) {
      return Iterables.concat(super.getInputs(), ImmutableList.of(schedulingMiddleman));
    } else {
      return super.getInputs();
    }
  }

  @Override
  public int getInputCount() {
    return Iterables.size(getInputs());
  }

  @Override
  protected String computeKey() {
    Fingerprint f = new Fingerprint();
    f.addString(GUID);
    f.addStrings(executionSettings.getArgs());
    f.addString(executionSettings.getTestFilter() == null ? "" : executionSettings.getTestFilter());
    RunUnder runUnder = executionSettings.getRunUnder();
    f.addString(runUnder == null ? "" : runUnder.getValue());
    f.addStringMap(executionSettings.getTestEnv());
    f.addString(testProperties.getSize().toString());
    f.addString(testProperties.getTimeout().toString());
    f.addStrings(testProperties.getTags());
    f.addInt(testProperties.isLocal() ? 1 : 0);
    f.addInt(shardNum);
    f.addInt(executionSettings.getTotalShards());
    f.addInt(runNumber);
    f.addInt(configuration.getRunsPerTestForLabel(getOwner().getLabel()));
    f.addInt(configuration.isCodeCoverageEnabled() ? 1 : 0);
    return f.hexDigest();
  }

  @Override
  public boolean executeUnconditionally(PackageUpToDateChecker upToDateChecker) {
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
              this, readCacheStatus()));
    } catch (IOException e) {
      LoggingUtil.logToRemote(Level.WARNING, "Failed creating cached protocol buffer", e);
    }
  }

  @Override
  public ResourceSet estimateResourceConsumption(Executor executor) {
    // return null here to indicate that resources would be managed manually
    // during action execution.
    return null;
  }

  @Override
  protected String getRawProgressMessage() {
    return "Testing " + getTestName();
  }

  @Override
  public String describeStrategy(Executor executor) {
    return executor.getContext(TestActionContext.class).strategyLocality(this);
  }

  /**
   * Deletes <b>all</b> possible test outputs.
   *
   * TestRunnerAction potentially can create many more non-declared outputs - xml output,
   * coverage data file and logs for failed attempts. All those outputs are uniquely
   * identified by the test log base name with arbitrary prefix and extension.
   */
  @Override
  protected void deleteOutputs() throws IOException {
    super.deleteOutputs();

    // We do not rely on globs, as it causes quadratic behavior in --runs_per_test and test
    // shard count.

    // We also need to remove *.(xml|data|shard|warnings|zip) files if they are present.
    xmlOutputPath.delete();
    testWarningsPath.delete();
    // Note that splitLogsPath points to a file inside the splitLogsDir so
    // it's not necessary to delete it explicitly.
    FileSystemUtils.deleteTree(splitLogsDir);
    FileSystemUtils.deleteTree(undeclaredOutputsDir);
    FileSystemUtils.deleteTree(undeclaredOutputsAnnotationsDir);
    testStderr.delete();
    testExitSafe.delete();
    if (testShard != null) {
      testShard.delete();
    }
    testInfrastructureFailure.delete();

    // Coverage files use "coverage" instead of "test".
    String coveragePrefix = "coverage" + namePrefix.substring(4);

    // We cannot use coverageData artifact since it may be null. Generate coverage name instead.
    baseDir.getChild(coveragePrefix + ".dat").delete();
    // We cannot use microcoverageData artifact since it may be null. Generate filename instead.
    baseDir.getChild(coveragePrefix + ".micro.dat").delete();

    // Delete files fetched from remote execution.
    baseDir.getChild(namePrefix + ".zip").delete();
    deleteTestAttemptsDirMaybe(baseDir, namePrefix);
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
   * In rare cases, error messages will be printed to stderr instead of stdout. The test action is
   * responsible for appending anything in the stderr file to the real test.log.
   */
  public Path getTestStderr() {
    return testStderr;
  }

  public Artifact getCacheStatusArtifact() {
    return cacheStatus;
  }

  public Path getTestWarningsPath() {
    return testWarningsPath;
  }

  public Path getSplitLogsPath() {
    return splitLogsPath;
  }

  /**
   * @return path to the directory containing the split logs (raw and proto file).
   */
  public Path getSplitLogsDir() {
    return splitLogsDir;
  }

  /**
   * @return path to the optional zip file of undeclared test outputs.
   */
  public Path getUndeclaredOutputsZipPath() {
    return undeclaredOutputsZipPath;
  }

  /**
   * @return path to the directory to hold undeclared test outputs.
   */
  public Path getUndeclaredOutputsDir() {
    return undeclaredOutputsDir;
  }

  /**
   * @return path to the directory to hold undeclared output annotations parts.
   */
  public Path getUndeclaredOutputsAnnotationsDir() { return undeclaredOutputsAnnotationsDir; }

  /**
   * @return path to the undeclared output manifest file.
   */
  public Path getUndeclaredOutputsManifestPath() { return undeclaredOutputsManifestPath; }

  /**
   * @return path to the undeclared output annotations file.
   */
  public Path getUndeclaredOutputsAnnotationsPath() { return undeclaredOutputsAnnotationsPath; }

  public Path getTestShard() {
    return testShard;
  }

  public Path getExitSafeFile() {
    return testExitSafe;
  }

  public Path getInfrastructureFailureFile() {
    return testInfrastructureFailure;
  }
  /**
   * @return path to the optionally created XML output file created by the test.
   */
  public Path getXmlOutputPath() {
    return xmlOutputPath;
  }

  /**
   * @return coverage data artifact or null if code coverage was not requested.
   */
  public PathFragment getCoverageData() {
    return coverageData;
  }

  /**
   * @return microcoverage data artifact or null if code coverage was not requested.
   */
  public PathFragment getMicroCoverageData() {
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

  public boolean isExclusive() {
    return this.schedulingMiddleman != null;
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

  /**
   * Returns estimated local resource usage.
   */
  public ResourceSet estimateTestResourceUsage (boolean isRemoteExecution) {
    return isRemoteExecution ? ResourceSet.ZERO : testProperties.getLocalResourceUsage();
  }

  @Override
  public int estimateWorkload() {
    // Base test workload on 1/5 of timeout (most tests do not run for their entire timeout).
    return getTestProperties().getTimeout().getTimeout() / 5;
  }

  @Override
  public String getMnemonic() {
    return "TestRunner";
  }
}
