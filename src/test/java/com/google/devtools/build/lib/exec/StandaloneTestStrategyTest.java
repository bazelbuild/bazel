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

package com.google.devtools.build.lib.exec;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.TestConstants.WORKSPACE_NAME;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.fail;
import static org.mockito.Matchers.any;
import static org.mockito.Matchers.argThat;
import static org.mockito.Matchers.same;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.MoreCollectors;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.analysis.test.TestResult;
import com.google.devtools.build.lib.analysis.test.TestRunnerAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.TestStatus;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.exec.TestStrategy.TestOutputFormat;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import com.google.devtools.common.options.Options;
import java.io.IOException;
import java.io.OutputStream;
import java.time.Duration;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentCaptor;
import org.mockito.ArgumentMatcher;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.mockito.invocation.InvocationOnMock;
import org.mockito.stubbing.Answer;

/** Unit tests for {@link StandaloneTestStrategy}. */
@RunWith(JUnit4.class)
public final class StandaloneTestStrategyTest extends BuildViewTestCase {

  private static class TestedStandaloneTestStrategy extends StandaloneTestStrategy {
    TestResult postedResult = null;

    public TestedStandaloneTestStrategy(
        ExecutionOptions executionOptions, BinTools binTools, Path tmpDirRoot) {
      super(executionOptions, binTools, tmpDirRoot);
    }

    @Override
    protected void postTestResult(ActionExecutionContext actionExecutionContext, TestResult result)
        throws IOException {
      postedResult = result;
    }
  }

  @Mock private ActionExecutionContext actionExecutionContext;

  @Mock private SpawnActionContext spawnActionContext;

  private StoredEventHandler storedEvents = new StoredEventHandler();

  @Before
  public final void setUp() throws Exception {
    MockitoAnnotations.initMocks(this);
  }

  @Test
  public void testRunTestOnce() throws Exception {

    // setup a StandaloneTestStrategy

    ExecutionOptions executionOptions = ExecutionOptions.DEFAULTS;
    Path tmpDirRoot = TestStrategy.getTmpRoot(rootDirectory, outputBase, executionOptions);
    BinTools binTools = BinTools.forUnitTesting(directories, analysisMock.getEmbeddedTools());
    TestedStandaloneTestStrategy standaloneTestStrategy =
        new TestedStandaloneTestStrategy(executionOptions, binTools, tmpDirRoot);

    // setup a test action

    scratch.file("standalone/simple_test.sh", "this does not get executed, it is mocked out");

    scratch.file(
        "standalone/BUILD",
        "sh_test(",
        "    name = \"simple_test\",",
        "    size = \"small\",",
        "    srcs = [\"simple_test.sh\"],",
        ")");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//standalone:simple_test");
    List<Artifact> testStatusArtifacts =
        configuredTarget.getProvider(TestProvider.class).getTestParams().getTestStatusArtifacts();
    Artifact testStatusArtifact = Iterables.getOnlyElement(testStatusArtifacts);
    TestRunnerAction testRunnerAction = (TestRunnerAction) getGeneratingAction(testStatusArtifact);
    FileSystemUtils.createDirectoryAndParents(
        testRunnerAction.getTestLog().getPath().getParentDirectory());

    // setup a mock ActionExecutionContext

    when(actionExecutionContext.getClock()).thenReturn(BlazeClock.instance());
    when(actionExecutionContext.withFileOutErr(any())).thenReturn(actionExecutionContext);
    when(actionExecutionContext.getExecRoot()).thenReturn(outputBase.getRelative("execroot"));
    when(actionExecutionContext.getClientEnv()).thenReturn(ImmutableMap.of());
    when(actionExecutionContext.getEventHandler()).thenReturn(storedEvents);
    when(actionExecutionContext.getInputPath(any())).thenAnswer(this::getInputPathMock);
    when(actionExecutionContext.getPathResolver()).thenReturn(ArtifactPathResolver.IDENTITY);

    SpawnResult expectedSpawnResult =
        new SpawnResult.Builder()
            .setStatus(Status.SUCCESS)
            .setWallTime(Duration.ofMillis(10))
            .setRunnerName("test")
            .build();
    when(spawnActionContext.exec(any(), any())).thenReturn(ImmutableList.of(expectedSpawnResult));

    when(actionExecutionContext.getContext(same(SpawnActionContext.class)))
        .thenReturn(spawnActionContext);

    // actual StandaloneTestStrategy execution
    List<SpawnResult> spawnResults =
        standaloneTestStrategy.exec(testRunnerAction, actionExecutionContext);

    assertThat(spawnResults).containsExactly(expectedSpawnResult);
    TestResult result = standaloneTestStrategy.postedResult;
    assertThat(result).isNotNull();
    assertThat(result.isCached()).isFalse();
    assertThat(result.getTestStatusArtifact()).isEqualTo(testStatusArtifact);
    assertThat(result.getData().getTestPassed()).isTrue();
    assertThat(result.getData().getRemotelyCached()).isFalse();
    assertThat(result.getData().getIsRemoteStrategy()).isFalse();
    assertThat(result.getData().getRunDurationMillis()).isEqualTo(10);
    assertThat(result.getData().getTestTimesList()).containsExactly(10L);
    TestAttempt attempt =
        storedEvents
            .getPosts()
            .stream()
            .filter(TestAttempt.class::isInstance)
            .map(TestAttempt.class::cast)
            .collect(MoreCollectors.onlyElement());
    assertThat(attempt.getExecutionInfo().getStrategy()).isEqualTo("test");
    assertThat(attempt.getExecutionInfo().getHostname()).isEqualTo("");
  }

  @Test
  public void testRunFlakyTest() throws Exception {

    // setup a StandaloneTestStrategy

    ExecutionOptions executionOptions = ExecutionOptions.DEFAULTS;
    Path tmpDirRoot = TestStrategy.getTmpRoot(rootDirectory, outputBase, executionOptions);
    BinTools binTools = BinTools.forUnitTesting(directories, analysisMock.getEmbeddedTools());
    TestedStandaloneTestStrategy standaloneTestStrategy =
        new TestedStandaloneTestStrategy(executionOptions, binTools, tmpDirRoot);

    // setup a test action

    scratch.file("standalone/simple_test.sh", "this does not get executed, it is mocked out");

    scratch.file(
        "standalone/BUILD",
        "sh_test(",
        "    name = \"simple_test\",",
        "    size = \"small\",",
        "    srcs = [\"simple_test.sh\"],",
        "    flaky = True,",
        ")");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//standalone:simple_test");
    List<Artifact> testStatusArtifacts =
        configuredTarget.getProvider(TestProvider.class).getTestParams().getTestStatusArtifacts();
    Artifact testStatusArtifact = Iterables.getOnlyElement(testStatusArtifacts);
    TestRunnerAction testRunnerAction = (TestRunnerAction) getGeneratingAction(testStatusArtifact);
    assertThat(testRunnerAction.getTestProperties().isFlaky()).isTrue();
    FileSystemUtils.createDirectoryAndParents(
        testRunnerAction.getTestLog().getPath().getParentDirectory());

    // setup a mock ActionExecutionContext

    when(actionExecutionContext.getClock()).thenReturn(BlazeClock.instance());
    when(actionExecutionContext.withFileOutErr(any())).thenReturn(actionExecutionContext);
    when(actionExecutionContext.getExecRoot()).thenReturn(outputBase.getRelative("execroot"));
    when(actionExecutionContext.getClientEnv()).thenReturn(ImmutableMap.of());
    when(actionExecutionContext.getEventHandler()).thenReturn(storedEvents);
    when(actionExecutionContext.getInputPath(any())).thenAnswer(this::getInputPathMock);
    when(actionExecutionContext.getPathResolver()).thenReturn(ArtifactPathResolver.IDENTITY);

    SpawnResult failSpawnResult =
        new SpawnResult.Builder()
            .setStatus(Status.NON_ZERO_EXIT)
            .setExitCode(1)
            .setWallTime(Duration.ofMillis(10))
            .setRunnerName("test")
            .build();
    SpawnResult passSpawnResult =
        new SpawnResult.Builder()
            .setStatus(Status.SUCCESS)
            .setWallTime(Duration.ofMillis(15))
            .setRunnerName("test")
            .build();
    when(spawnActionContext.exec(any(), any()))
        .thenThrow(new SpawnExecException("test failed", failSpawnResult, false))
        .thenReturn(ImmutableList.of(passSpawnResult));

    when(actionExecutionContext.getContext(same(SpawnActionContext.class)))
        .thenReturn(spawnActionContext);

    // actual StandaloneTestStrategy execution
    List<SpawnResult> spawnResults =
        standaloneTestStrategy.exec(testRunnerAction, actionExecutionContext);

    assertThat(spawnResults).containsExactly(passSpawnResult);
    TestResult result = standaloneTestStrategy.postedResult;
    assertThat(result).isNotNull();
    assertThat(result.isCached()).isFalse();
    assertThat(result.getTestStatusArtifact()).isEqualTo(testStatusArtifact);
    assertThat(result.getData().getStatus()).isEqualTo(BlazeTestStatus.FLAKY);
    assertThat(result.getData().getTestPassed()).isTrue();
    assertThat(result.getData().getRemotelyCached()).isFalse();
    assertThat(result.getData().getIsRemoteStrategy()).isFalse();
    assertThat(result.getData().getRunDurationMillis()).isEqualTo(15L);
    assertThat(result.getData().getTestTimesList()).containsExactly(10L, 15L);
    List<TestAttempt> attempts =
        storedEvents
            .getPosts()
            .stream()
            .filter(TestAttempt.class::isInstance)
            .map(TestAttempt.class::cast)
            .collect(ImmutableList.toImmutableList());
    assertThat(attempts).hasSize(2);
    TestAttempt failedAttempt = attempts.get(0);
    assertThat(failedAttempt.getExecutionInfo().getStrategy()).isEqualTo("test");
    assertThat(failedAttempt.getExecutionInfo().getHostname()).isEqualTo("");
    assertThat(failedAttempt.getStatus()).isEqualTo(TestStatus.FAILED);
    assertThat(failedAttempt.getExecutionInfo().getCachedRemotely()).isFalse();
    TestAttempt okAttempt = attempts.get(1);
    assertThat(okAttempt.getStatus()).isEqualTo(TestStatus.PASSED);
    assertThat(okAttempt.getExecutionInfo().getStrategy()).isEqualTo("test");
    assertThat(okAttempt.getExecutionInfo().getHostname()).isEqualTo("");
  }

  @Test
  public void testRunTestRemotely() throws Exception {

    // setup a StandaloneTestStrategy

    ExecutionOptions executionOptions = ExecutionOptions.DEFAULTS;
    Path tmpDirRoot = TestStrategy.getTmpRoot(rootDirectory, outputBase, executionOptions);
    BinTools binTools = BinTools.forUnitTesting(directories, analysisMock.getEmbeddedTools());
    TestedStandaloneTestStrategy standaloneTestStrategy =
        new TestedStandaloneTestStrategy(executionOptions, binTools, tmpDirRoot);

    // setup a test action

    scratch.file("standalone/simple_test.sh", "this does not get executed, it is mocked out");

    scratch.file(
        "standalone/BUILD",
        "sh_test(",
        "    name = \"simple_test\",",
        "    size = \"small\",",
        "    srcs = [\"simple_test.sh\"],",
        ")");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//standalone:simple_test");
    List<Artifact> testStatusArtifacts =
        configuredTarget.getProvider(TestProvider.class).getTestParams().getTestStatusArtifacts();
    Artifact testStatusArtifact = Iterables.getOnlyElement(testStatusArtifacts);
    TestRunnerAction testRunnerAction = (TestRunnerAction) getGeneratingAction(testStatusArtifact);
    FileSystemUtils.createDirectoryAndParents(
        testRunnerAction.getTestLog().getPath().getParentDirectory());

    // setup a mock ActionExecutionContext

    when(actionExecutionContext.getClock()).thenReturn(BlazeClock.instance());
    when(actionExecutionContext.withFileOutErr(any())).thenReturn(actionExecutionContext);
    when(actionExecutionContext.getExecRoot()).thenReturn(outputBase.getRelative("execroot"));
    when(actionExecutionContext.getClientEnv()).thenReturn(ImmutableMap.of());
    when(actionExecutionContext.getEventHandler()).thenReturn(storedEvents);
    when(actionExecutionContext.getInputPath(any())).thenAnswer(this::getInputPathMock);
    when(actionExecutionContext.getPathResolver()).thenReturn(ArtifactPathResolver.IDENTITY);

    SpawnResult expectedSpawnResult =
        new SpawnResult.Builder()
            .setStatus(Status.SUCCESS)
            .setWallTime(Duration.ofMillis(10))
            .setRunnerName("remote")
            .setExecutorHostname("a-remote-host")
            .build();
    when(spawnActionContext.exec(any(), any())).thenReturn(ImmutableList.of(expectedSpawnResult));

    when(actionExecutionContext.getContext(same(SpawnActionContext.class)))
        .thenReturn(spawnActionContext);

    // actual StandaloneTestStrategy execution
    List<SpawnResult> spawnResults =
        standaloneTestStrategy.exec(testRunnerAction, actionExecutionContext);

    assertThat(spawnResults).containsExactly(expectedSpawnResult);
    TestResult result = standaloneTestStrategy.postedResult;
    assertThat(result).isNotNull();
    assertThat(result.isCached()).isFalse();
    assertThat(result.getTestStatusArtifact()).isEqualTo(testStatusArtifact);
    assertThat(result.getData().getTestPassed()).isTrue();
    assertThat(result.getData().getRemotelyCached()).isFalse();
    assertThat(result.getData().getIsRemoteStrategy()).isTrue();
    assertThat(result.getData().getRunDurationMillis()).isEqualTo(10);
    assertThat(result.getData().getTestTimesList()).containsExactly(10L);
    TestAttempt attempt =
        storedEvents
            .getPosts()
            .stream()
            .filter(TestAttempt.class::isInstance)
            .map(TestAttempt.class::cast)
            .collect(MoreCollectors.onlyElement());
    assertThat(attempt.getStatus()).isEqualTo(TestStatus.PASSED);
    assertThat(attempt.getExecutionInfo().getStrategy()).isEqualTo("remote");
    assertThat(attempt.getExecutionInfo().getHostname()).isEqualTo("a-remote-host");
  }

  @Test
  public void testRunRemotelyCachedTest() throws Exception {

    // setup a StandaloneTestStrategy

    ExecutionOptions executionOptions = ExecutionOptions.DEFAULTS;
    Path tmpDirRoot = TestStrategy.getTmpRoot(rootDirectory, outputBase, executionOptions);
    BinTools binTools = BinTools.forUnitTesting(directories, analysisMock.getEmbeddedTools());
    TestedStandaloneTestStrategy standaloneTestStrategy =
        new TestedStandaloneTestStrategy(executionOptions, binTools, tmpDirRoot);

    // setup a test action

    scratch.file("standalone/simple_test.sh", "this does not get executed, it is mocked out");

    scratch.file(
        "standalone/BUILD",
        "sh_test(",
        "    name = \"simple_test\",",
        "    size = \"small\",",
        "    srcs = [\"simple_test.sh\"],",
        ")");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//standalone:simple_test");
    List<Artifact> testStatusArtifacts =
        configuredTarget.getProvider(TestProvider.class).getTestParams().getTestStatusArtifacts();
    Artifact testStatusArtifact = Iterables.getOnlyElement(testStatusArtifacts);
    TestRunnerAction testRunnerAction = (TestRunnerAction) getGeneratingAction(testStatusArtifact);
    FileSystemUtils.createDirectoryAndParents(
        testRunnerAction.getTestLog().getPath().getParentDirectory());

    // setup a mock ActionExecutionContext

    when(actionExecutionContext.getClock()).thenReturn(BlazeClock.instance());
    when(actionExecutionContext.withFileOutErr(any())).thenReturn(actionExecutionContext);
    when(actionExecutionContext.getExecRoot()).thenReturn(outputBase.getRelative("execroot"));
    when(actionExecutionContext.getClientEnv()).thenReturn(ImmutableMap.of());
    when(actionExecutionContext.getEventHandler()).thenReturn(storedEvents);
    when(actionExecutionContext.getInputPath(any())).thenAnswer(this::getInputPathMock);
    when(actionExecutionContext.getPathResolver()).thenReturn(ArtifactPathResolver.IDENTITY);

    SpawnResult expectedSpawnResult =
        new SpawnResult.Builder()
            .setStatus(Status.SUCCESS)
            .setCacheHit(true)
            .setWallTime(Duration.ofMillis(10))
            .setRunnerName("remote cache")
            .build();
    when(spawnActionContext.exec(any(), any())).thenReturn(ImmutableList.of(expectedSpawnResult));

    when(actionExecutionContext.getContext(same(SpawnActionContext.class)))
        .thenReturn(spawnActionContext);

    // actual StandaloneTestStrategy execution
    List<SpawnResult> spawnResults =
        standaloneTestStrategy.exec(testRunnerAction, actionExecutionContext);

    // check that the rigged SpawnResult was returned
    assertThat(spawnResults).containsExactly(expectedSpawnResult);
    TestResult result = standaloneTestStrategy.postedResult;
    assertThat(result).isNotNull();
    assertThat(result.isCached()).isFalse();
    assertThat(result.getTestStatusArtifact()).isEqualTo(testStatusArtifact);
    assertThat(result.getData().getTestPassed()).isTrue();
    assertThat(result.getData().getRemotelyCached()).isTrue();
    assertThat(result.getData().getIsRemoteStrategy()).isFalse();
    assertThat(result.getData().getRunDurationMillis()).isEqualTo(10);
    assertThat(result.getData().getTestTimesList()).containsExactly(10L);
    TestAttempt attempt =
        storedEvents
            .getPosts()
            .stream()
            .filter(TestAttempt.class::isInstance)
            .map(TestAttempt.class::cast)
            .collect(MoreCollectors.onlyElement());
    assertThat(attempt.getExecutionInfo().getStrategy()).isEqualTo("remote cache");
    assertThat(attempt.getExecutionInfo().getHostname()).isEqualTo("");
  }

  @Test
  public void testThatTestLogAndOutputAreReturned() throws Exception {

    // setup a StandaloneTestStrategy

    ExecutionOptions executionOptions = Options.getDefaults(ExecutionOptions.class);
    executionOptions.testOutput = TestOutputFormat.ERRORS;
    executionOptions.splitXmlGeneration = false;
    Path tmpDirRoot = TestStrategy.getTmpRoot(rootDirectory, outputBase, executionOptions);
    BinTools binTools = BinTools.forUnitTesting(directories, analysisMock.getEmbeddedTools());
    TestedStandaloneTestStrategy standaloneTestStrategy =
        new TestedStandaloneTestStrategy(executionOptions, binTools, tmpDirRoot);

    // setup a test action

    scratch.file("standalone/failing_test.sh", "this does not get executed, it is mocked out");

    scratch.file(
        "standalone/BUILD",
        "sh_test(",
        "    name = \"failing_test\",",
        "    size = \"small\",",
        "    srcs = [\"failing_test.sh\"],",
        ")");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//standalone:failing_test");
    List<Artifact> testStatusArtifacts =
        configuredTarget.getProvider(TestProvider.class).getTestParams().getTestStatusArtifacts();
    Artifact testStatusArtifact = Iterables.getOnlyElement(testStatusArtifacts);
    TestRunnerAction testRunnerAction = (TestRunnerAction) getGeneratingAction(testStatusArtifact);
    FileSystemUtils.createDirectoryAndParents(
        testRunnerAction.getTestLog().getPath().getParentDirectory());
    // setup a mock ActionExecutionContext

    when(actionExecutionContext.getClock()).thenReturn(BlazeClock.instance());
    when(actionExecutionContext.withFileOutErr(any()))
        .thenAnswer(
            new Answer<ActionExecutionContext>() {
              @SuppressWarnings("unchecked")
              @Override
              public ActionExecutionContext answer(InvocationOnMock invocation) throws Throwable {
                FileOutErr outErr = (FileOutErr) invocation.getArguments()[0];
                try (OutputStream stream = outErr.getOutputStream()) {
                  stream.write("This will not appear in the test output: bla\n".getBytes(UTF_8));
                  stream.write((TestLogHelper.HEADER_DELIMITER + "\n").getBytes(UTF_8));
                  stream.write("This will appear in the test output: foo\n".getBytes(UTF_8));
                }
                return actionExecutionContext;
              }
            });
    reporter.removeHandler(failFastHandler);
    when(actionExecutionContext.getExecRoot()).thenReturn(outputBase.getRelative("execroot"));
    when(actionExecutionContext.getClientEnv()).thenReturn(ImmutableMap.of());
    when(actionExecutionContext.getEventHandler()).thenReturn(reporter);
    when(actionExecutionContext.getEventBus()).thenReturn(eventBus);
    when(actionExecutionContext.getInputPath(any())).thenAnswer(this::getInputPathMock);
    when(actionExecutionContext.getPathResolver()).thenReturn(ArtifactPathResolver.IDENTITY);

    Path outPath = tmpDirRoot.getRelative("test-out.txt");
    Path errPath = tmpDirRoot.getRelative("test-err.txt");
    FileOutErr outErr = new FileOutErr(outPath, errPath);
    when(actionExecutionContext.getFileOutErr()).thenReturn(outErr);

    SpawnResult expectedSpawnResult =
        new SpawnResult.Builder()
            .setStatus(Status.NON_ZERO_EXIT)
            .setExitCode(1)
            .setRunnerName("test")
            .build();
    when(spawnActionContext.exec(any(), any()))
        .thenThrow(
            new SpawnExecException(
                "Failure!!",
                expectedSpawnResult,
                /*forciblyRunRemotely=*/ false,
                /*catastrophe=*/ false));
    when(actionExecutionContext.getContext(same(SpawnActionContext.class)))
        .thenReturn(spawnActionContext);

    // actual StandaloneTestStrategy execution
    List<SpawnResult> spawnResults =
        standaloneTestStrategy.exec(testRunnerAction, actionExecutionContext);

    // check that the rigged SpawnResult was returned
    assertThat(spawnResults).containsExactly(expectedSpawnResult);
    // check that the test log contains all the output
    try {
      String logData = FileSystemUtils.readContent(testRunnerAction.getTestLog().getPath(), UTF_8);
      assertThat(logData).contains("bla");
      assertThat(logData).contains(TestLogHelper.HEADER_DELIMITER);
      assertThat(logData).contains("foo");
    } catch (IOException e) {
      fail("Test log missing: " + testRunnerAction.getTestLog().getPath());
    }
    // check that the test stdout contains all the expected output
    outErr.close(); // Create the output files.
    try {
      String outData = FileSystemUtils.readContent(outPath, UTF_8);
      assertThat(outData)
          .contains("==================== Test output for //standalone:failing_test:");
      assertThat(outData).doesNotContain("bla");
      assertThat(outData).doesNotContain(TestLogHelper.HEADER_DELIMITER);
      assertThat(outData).contains("foo");
      assertThat(outData)
          .contains(
              "================================================================================");
    } catch (IOException e) {
      fail("Test stdout file missing: " + outPath);
    }
    assertThat(errPath.exists()).isFalse();
  }

  @Test
  public void testThatTestLogAndOutputAreReturnedWithSplitXmlGeneration() throws Exception {

    // setup a StandaloneTestStrategy

    ExecutionOptions executionOptions = Options.getDefaults(ExecutionOptions.class);
    executionOptions.testOutput = TestOutputFormat.ERRORS;
    executionOptions.splitXmlGeneration = true;
    Path tmpDirRoot = TestStrategy.getTmpRoot(rootDirectory, outputBase, executionOptions);
    BinTools binTools = BinTools.forUnitTesting(directories, analysisMock.getEmbeddedTools());
    TestedStandaloneTestStrategy standaloneTestStrategy =
        new TestedStandaloneTestStrategy(executionOptions, binTools, tmpDirRoot);

    // setup a test action

    scratch.file("standalone/failing_test.sh", "this does not get executed, it is mocked out");

    scratch.file(
        "standalone/BUILD",
        "sh_test(",
        "    name = \"failing_test\",",
        "    size = \"small\",",
        "    srcs = [\"failing_test.sh\"],",
        ")");

    ConfiguredTarget configuredTarget = getConfiguredTarget("//standalone:failing_test");
    List<Artifact> testStatusArtifacts =
        configuredTarget.getProvider(TestProvider.class).getTestParams().getTestStatusArtifacts();
    Artifact testStatusArtifact = Iterables.getOnlyElement(testStatusArtifacts);
    TestRunnerAction testRunnerAction = (TestRunnerAction) getGeneratingAction(testStatusArtifact);
    FileSystemUtils.createDirectoryAndParents(
        testRunnerAction.getTestLog().getPath().getParentDirectory());
    // setup a mock ActionExecutionContext

    when(actionExecutionContext.getClock()).thenReturn(BlazeClock.instance());
    ArgumentCaptor<FileOutErr> outErrCaptor = ArgumentCaptor.forClass(FileOutErr.class);
    when(actionExecutionContext.withFileOutErr(outErrCaptor.capture()))
        .thenAnswer(
            new Answer<ActionExecutionContext>() {
              @SuppressWarnings("unchecked")
              @Override
              public ActionExecutionContext answer(InvocationOnMock invocation) throws Throwable {
                FileOutErr outErr = (FileOutErr) invocation.getArguments()[0];
                try (OutputStream stream = outErr.getOutputStream()) {
                  stream.write("This will not appear in the test output: bla\n".getBytes(UTF_8));
                  stream.write((TestLogHelper.HEADER_DELIMITER + "\n").getBytes(UTF_8));
                  stream.write("This will appear in the test output: foo\n".getBytes(UTF_8));
                }
                return actionExecutionContext;
              }
            });
    reporter.removeHandler(failFastHandler);
    when(actionExecutionContext.getExecRoot()).thenReturn(outputBase.getRelative("execroot"));
    when(actionExecutionContext.getClientEnv()).thenReturn(ImmutableMap.of());
    when(actionExecutionContext.getEventHandler()).thenReturn(reporter);
    when(actionExecutionContext.getEventBus()).thenReturn(eventBus);
    when(actionExecutionContext.getInputPath(any())).thenAnswer(this::getInputPathMock);
    when(actionExecutionContext.getPathResolver()).thenReturn(ArtifactPathResolver.IDENTITY);

    Path outPath = tmpDirRoot.getRelative("test-out.txt");
    Path errPath = tmpDirRoot.getRelative("test-err.txt");
    FileOutErr outErr = new FileOutErr(outPath, errPath);
    when(actionExecutionContext.getFileOutErr()).thenReturn(outErr);

    SpawnResult testSpawnResult =
        new SpawnResult.Builder()
            .setStatus(Status.NON_ZERO_EXIT)
            .setExitCode(1)
            .setRunnerName("test")
            .build();
    when(spawnActionContext.exec(argThat(new ArgumentMatcher<Spawn>() {
          @Override
          public boolean matches(Object argument) {
            return (argument instanceof Spawn) && ((Spawn) argument).getOutputFiles().size() != 1;
          }
        }), any()))
        .thenThrow(
            new SpawnExecException(
                "Failure!!",
                testSpawnResult,
                /*forciblyRunRemotely=*/ false,
                /*catastrophe=*/ false));

    SpawnResult xmlGeneratorSpawnResult =
        new SpawnResult.Builder()
            .setStatus(Status.SUCCESS)
            .setRunnerName("test")
            .build();
    when(spawnActionContext.exec(argThat(new ArgumentMatcher<Spawn>() {
          @Override
          public boolean matches(Object argument) {
            return (argument instanceof Spawn) && ((Spawn) argument).getOutputFiles().size() == 1;
          }
        }), any()))
        .thenReturn(ImmutableList.of(xmlGeneratorSpawnResult));
    when(actionExecutionContext.getContext(same(SpawnActionContext.class)))
        .thenReturn(spawnActionContext);

    // actual StandaloneTestStrategy execution
    List<SpawnResult> spawnResults =
        standaloneTestStrategy.exec(testRunnerAction, actionExecutionContext);

    // check that the rigged SpawnResult was returned
    assertThat(spawnResults).containsExactly(testSpawnResult, xmlGeneratorSpawnResult);
    // check that the test log contains all the output
    String logData = FileSystemUtils.readContent(testRunnerAction.getTestLog().getPath(), UTF_8);
    assertThat(logData).contains("bla");
    assertThat(logData).contains(TestLogHelper.HEADER_DELIMITER);
    assertThat(logData).contains("foo");
    // check that the test stdout contains all the expected output
    outErr.close(); // Create the output files.
    String outData = FileSystemUtils.readContent(outPath, UTF_8);
    assertThat(outData)
        .contains("==================== Test output for //standalone:failing_test:");
    assertThat(outData).doesNotContain("bla");
    assertThat(outData).doesNotContain(TestLogHelper.HEADER_DELIMITER);
    assertThat(outData).contains("foo");
    assertThat(outData)
        .contains(
            "================================================================================");
    assertThat(errPath.exists()).isFalse();
    assertThat(outErrCaptor.getAllValues()).hasSize(2);
    assertThat(outErrCaptor.getAllValues()).containsNoDuplicates();
  }

  @Test
  public void testEmptyOutputCreatesEmptyLogFile() throws Exception {
    // setup a StandaloneTestStrategy
    ExecutionOptions executionOptions = Options.getDefaults(ExecutionOptions.class);
    executionOptions.testOutput = TestOutputFormat.ALL;
    Path tmpDirRoot = TestStrategy.getTmpRoot(rootDirectory, outputBase, executionOptions);
    BinTools binTools = BinTools.forUnitTesting(directories, analysisMock.getEmbeddedTools());
    TestedStandaloneTestStrategy standaloneTestStrategy =
        new TestedStandaloneTestStrategy(executionOptions, binTools, tmpDirRoot);

    // setup a test action
    scratch.file("standalone/empty_test.sh", "this does not get executed, it is mocked out");
    scratch.file(
        "standalone/BUILD",
        "sh_test(",
        "    name = \"empty_test\",",
        "    size = \"small\",",
        "    srcs = [\"empty_test.sh\"],",
        ")");
    ConfiguredTarget configuredTarget = getConfiguredTarget("//standalone:empty_test");
    List<Artifact> testStatusArtifacts =
        configuredTarget.getProvider(TestProvider.class).getTestParams().getTestStatusArtifacts();
    Artifact testStatusArtifact = Iterables.getOnlyElement(testStatusArtifacts);
    TestRunnerAction testRunnerAction = (TestRunnerAction) getGeneratingAction(testStatusArtifact);
    FileSystemUtils.createDirectoryAndParents(
        testRunnerAction.getTestLog().getPath().getParentDirectory());

    // setup a mock ActionExecutionContext
    when(actionExecutionContext.getClock()).thenReturn(BlazeClock.instance());
    when(actionExecutionContext.withFileOutErr(any())).thenReturn(actionExecutionContext);
    when(actionExecutionContext.getExecRoot()).thenReturn(outputBase.getRelative("execroot"));
    when(actionExecutionContext.getClientEnv()).thenReturn(ImmutableMap.of());
    when(actionExecutionContext.getEventHandler()).thenReturn(reporter);
    when(actionExecutionContext.getEventBus()).thenReturn(eventBus);
    when(actionExecutionContext.getInputPath(any())).thenAnswer(this::getInputPathMock);
    when(actionExecutionContext.getPathResolver()).thenReturn(ArtifactPathResolver.IDENTITY);
    Path outPath = tmpDirRoot.getRelative("test-out.txt");
    Path errPath = tmpDirRoot.getRelative("test-err.txt");
    FileOutErr outErr = new FileOutErr(outPath, errPath);
    when(actionExecutionContext.getFileOutErr()).thenReturn(outErr);

    SpawnResult expectedSpawnResult =
        new SpawnResult.Builder().setStatus(Status.SUCCESS).setRunnerName("test").build();
    when(spawnActionContext.exec(any(), any())).thenReturn(ImmutableList.of(expectedSpawnResult));
    when(actionExecutionContext.getContext(same(SpawnActionContext.class)))
        .thenReturn(spawnActionContext);

    // actual StandaloneTestStrategy execution
    List<SpawnResult> spawnResults =
        standaloneTestStrategy.exec(testRunnerAction, actionExecutionContext);

    // check that the rigged SpawnResult was returned
    assertThat(spawnResults).containsExactly(expectedSpawnResult);
    // check that the test log contains all the output
    try {
      String logData = FileSystemUtils.readContent(testRunnerAction.getTestLog().getPath(), UTF_8);
      assertThat(logData).isEmpty();
    } catch (IOException e) {
      fail("Test log missing: " + testRunnerAction.getTestLog().getPath());
    }
    // check that the test stdout contains all the expected output
    outErr.close(); // Create the output files.
    try {
      String outData = FileSystemUtils.readContent(outPath, UTF_8);
      String emptyOutput =
          "==================== Test output for //standalone:empty_test:(\\s)*"
              + "================================================================================(\\s)*";
      assertThat(outData).matches(emptyOutput);
    } catch (IOException e) {
      fail("Test stdout file missing: " + outPath);
    }
    assertThat(errPath.exists()).isFalse();
  }

  private Path getInputPathMock(InvocationOnMock invocation) {
    return outputBase
        .getRelative("execroot/" + WORKSPACE_NAME)
        .getRelative(invocation.getArgumentAt(0, ActionInput.class).getExecPath());
  }
}
