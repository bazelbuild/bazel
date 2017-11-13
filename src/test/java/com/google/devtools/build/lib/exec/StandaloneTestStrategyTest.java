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
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.fail;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BinTools;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.analysis.test.TestResult;
import com.google.devtools.build.lib.analysis.test.TestRunnerAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.exec.TestStrategy.TestOutputFormat;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.Options;
import java.io.IOException;
import java.io.OutputStream;
import java.time.Duration;
import java.util.List;
import java.util.Set;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.mockito.invocation.InvocationOnMock;
import org.mockito.stubbing.Answer;

/** Unit tests for {@link StandaloneTestStrategy}. */
@RunWith(JUnit4.class)
public final class StandaloneTestStrategyTest extends BuildViewTestCase {

  private static class TestedStandaloneTestStrategy extends StandaloneTestStrategy {
    public TestedStandaloneTestStrategy(
        ExecutionOptions executionOptions, BinTools binTools, Path tmpDirRoot) {
      super(executionOptions, binTools, tmpDirRoot);
    }

    @Override
    protected void postTestResult(ActionExecutionContext actionExecutionContext, TestResult result)
        throws IOException {
      // Make postTestResult a no-op for testing purposes
    }
  }

  @Mock private ActionExecutionContext actionExecutionContext;

  @Mock private SpawnActionContext spawnActionContext;

  @Before
  public final void setUp() throws Exception {
    MockitoAnnotations.initMocks(this);
  }

  @Test
  public void testSpawnResultsAreReturned() throws Exception {

    // setup a StandaloneTestStrategy

    ExecutionOptions executionOptions = ExecutionOptions.DEFAULTS;
    Path tmpDirRoot = TestStrategy.getTmpRoot(rootDirectory, outputBase, executionOptions);
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
    when(actionExecutionContext.getEventHandler()).thenReturn(reporter);
    when(actionExecutionContext.getEventBus()).thenReturn(eventBus);

    SpawnResult expectedSpawnResult =
        new SpawnResult.Builder()
            .setStatus(Status.SUCCESS)
            .setWallTime(Duration.ofMillis(10))
            .build();
    when(spawnActionContext.exec(any(), any())).thenReturn(ImmutableSet.of(expectedSpawnResult));

    when(actionExecutionContext.getSpawnActionContext(any())).thenReturn(spawnActionContext);

    // actual StandaloneTestStrategy execution
    Set<SpawnResult> spawnResults =
        standaloneTestStrategy.exec(testRunnerAction, actionExecutionContext);

    // check that the rigged SpawnResult was returned
    assertThat(spawnResults).containsExactly(expectedSpawnResult);
  }

  @Test
  public void testThatTestLogAndOutputAreReturned() throws Exception {

    // setup a StandaloneTestStrategy

    ExecutionOptions executionOptions = Options.getDefaults(ExecutionOptions.class);
    executionOptions.testOutput = TestOutputFormat.ERRORS;
    Path tmpDirRoot = TestStrategy.getTmpRoot(rootDirectory, outputBase, executionOptions);
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
    Path outPath = tmpDirRoot.getRelative("test-out.txt");
    Path errPath = tmpDirRoot.getRelative("test-err.txt");
    FileOutErr outErr = new FileOutErr(outPath, errPath);
    when(actionExecutionContext.getFileOutErr()).thenReturn(outErr);

    SpawnResult expectedSpawnResult =
        new SpawnResult.Builder().setStatus(Status.EXECUTION_FAILED).setExitCode(1).build();
    when(spawnActionContext.exec(any(), any()))
        .thenThrow(
            new SpawnExecException(
                "Failure!!",
                expectedSpawnResult,
                /*forciblyRunRemotely=*/ false,
                /*catastrophe=*/ false));
    when(actionExecutionContext.getSpawnActionContext(any())).thenReturn(spawnActionContext);

    // actual StandaloneTestStrategy execution
    Set<SpawnResult> spawnResults =
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
  public void testEmptyOutputCreatesEmptyLogFile() throws Exception {
    // setup a StandaloneTestStrategy
    ExecutionOptions executionOptions = Options.getDefaults(ExecutionOptions.class);
    executionOptions.testOutput = TestOutputFormat.ALL;
    Path tmpDirRoot = TestStrategy.getTmpRoot(rootDirectory, outputBase, executionOptions);
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
    Path outPath = tmpDirRoot.getRelative("test-out.txt");
    Path errPath = tmpDirRoot.getRelative("test-err.txt");
    FileOutErr outErr = new FileOutErr(outPath, errPath);
    when(actionExecutionContext.getFileOutErr()).thenReturn(outErr);

    SpawnResult expectedSpawnResult = new SpawnResult.Builder().setStatus(Status.SUCCESS).build();
    when(spawnActionContext.exec(any(), any())).thenReturn(ImmutableSet.of(expectedSpawnResult));
    when(actionExecutionContext.getSpawnActionContext(any())).thenReturn(spawnActionContext);

    // actual StandaloneTestStrategy execution
    Set<SpawnResult> spawnResults =
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
}
