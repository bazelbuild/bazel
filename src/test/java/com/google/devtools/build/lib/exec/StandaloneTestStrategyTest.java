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
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.MoreCollectors;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecutorInitException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnContinuation;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.actions.SpawnStrategy;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.test.TestActionContext;
import com.google.devtools.build.lib.analysis.test.TestAttempt;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.analysis.test.TestResult;
import com.google.devtools.build.lib.analysis.test.TestRunnerAction;
import com.google.devtools.build.lib.analysis.test.TestStrategy;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.TestStatus;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.collect.nestedset.NestedSetExpander;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.exec.util.TestExecutorBuilder;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import com.google.devtools.common.options.Options;
import java.io.IOException;
import java.io.OutputStream;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

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
    protected void postTestResult(
        ActionExecutionContext actionExecutionContext, TestResult result) {
      postedResult = result;
    }
  }

  private static ActionContext.ActionContextRegistry toContextRegistry(
      SpawnStrategy spawnStrategy,
      BinTools binTools,
      FileSystem fileSystem,
      BlazeDirectories directories) {
    try {
      return new TestExecutorBuilder(fileSystem, directories, binTools)
          .addStrategy(spawnStrategy, "mock")
          .setDefaultStrategies("mock")
          .build();
    } catch (ExecutorInitException e) {
      throw new AssertionError(e);
    }
  }

  private class FakeActionExecutionContext extends ActionExecutionContext {
    private final ActionContext.ActionContextRegistry actionContextRegistry;

    public FakeActionExecutionContext(
        FileOutErr fileOutErr, SpawnStrategy spawnStrategy, BinTools binTools) {
      this(fileOutErr, toContextRegistry(spawnStrategy, binTools, fileSystem, directories));
    }

    public FakeActionExecutionContext(
        FileOutErr fileOutErr, ActionContext.ActionContextRegistry actionContextRegistry) {
      super(
          /*executor=*/ null,
          /*actionInputFileCache=*/ null,
          ActionInputPrefetcher.NONE,
          new ActionKeyContext(),
          /*metadataHandler=*/ null,
          LostInputsCheck.NONE,
          fileOutErr,
          /*eventHandler=*/ null,
          /*clientEnv=*/ ImmutableMap.of(),
          /*topLevelFilesets=*/ ImmutableMap.of(),
          /*artifactExpander=*/ null,
          /*actionFileSystem=*/ null,
          /*skyframeDepsResult=*/ null,
          NestedSetExpander.DEFAULT);
      this.actionContextRegistry = actionContextRegistry;
    }

    @Override
    public Clock getClock() {
      return BlazeClock.instance();
    }

    @Override
    @Nullable
    public <T extends ActionContext> T getContext(Class<T> type) {
      return actionContextRegistry.getContext(type);
    }

    @Override
    public ExtendedEventHandler getEventHandler() {
      return storedEvents;
    }

    @Override
    public Path getExecRoot() {
      return StandaloneTestStrategyTest.this.getExecRoot();
    }

    @Override
    public ActionExecutionContext withFileOutErr(FileOutErr fileOutErr) {
      return new FakeActionExecutionContext(fileOutErr, actionContextRegistry);
    }
  }

  @Mock private SpawnStrategy spawnStrategy;

  private StoredEventHandler storedEvents = new StoredEventHandler();

  @Before
  public final void setUp() throws Exception {
    MockitoAnnotations.initMocks(this);
    when(spawnStrategy.canExec(any(), any())).thenReturn(true);
  }

  private FileOutErr createTempOutErr(Path tmpDirRoot) {
    Path outPath = tmpDirRoot.getRelative("test-out.txt");
    Path errPath = tmpDirRoot.getRelative("test-err.txt");
    return new FileOutErr(outPath, errPath);
  }

  private TestRunnerAction getTestAction(String target) throws Exception {
    ConfiguredTarget configuredTarget = getConfiguredTarget(target);
    ImmutableList<Artifact.DerivedArtifact> testStatusArtifacts =
        configuredTarget.getProvider(TestProvider.class).getTestParams().getTestStatusArtifacts();
    Artifact testStatusArtifact = Iterables.getOnlyElement(testStatusArtifacts);
    TestRunnerAction action = (TestRunnerAction) getGeneratingAction(testStatusArtifact);
    action.getTestLog().getPath().getParentDirectory().createDirectoryAndParents();
    return action;
  }

  private List<TestRunnerAction> getTestActions(String target) throws Exception {
    ConfiguredTarget configuredTarget = getConfiguredTarget(target);
    ImmutableList<Artifact.DerivedArtifact> testStatusArtifacts =
        configuredTarget.getProvider(TestProvider.class).getTestParams().getTestStatusArtifacts();
    return testStatusArtifacts.stream()
        .map(
            (a) -> {
              TestRunnerAction action = (TestRunnerAction) getGeneratingAction(a);
              try {
                action.getTestLog().getPath().getParentDirectory().createDirectoryAndParents();
              } catch (IOException e) {
                throw new IllegalStateException(e);
              }
              return action;
            })
        .collect(Collectors.toList());
  }

  private List<SpawnResult> execute(
      TestRunnerAction testRunnerAction,
      ActionExecutionContext actionExecutionContext,
      TestActionContext testActionContext)
      throws ActionExecutionException, InterruptedException {
    return testRunnerAction.execute(actionExecutionContext, testActionContext).spawnResults();
  }

  @Test
  public void testCreateTmpDirForTest() throws Exception {
    // setup a test action
    scratch.file("standalone/simple_test.sh", "this does not get executed, it is mocked out");
    scratch.file(
        "standalone/BUILD",
        "sh_test(",
        "    name = \"simple_test\",",
        "    size = \"small\",",
        "    srcs = [\"simple_test.sh\"],",
        ")");
    TestRunnerAction testRunnerAction = getTestAction("//standalone:simple_test");

    String tmpDirName = TestStrategy.getTmpDirName(testRunnerAction);
    // Make sure the length of tmpDirName doesn't change unexpectedy: it cannot be too long
    // because Windows and macOS have limitations on file path length.
    // Note: It's OK to update 32 to a smaller number if tmpDirName gets shorter.
    assertThat(tmpDirName.length()).isEqualTo(32);
  }

  @Test
  public void testRunTestOnce() throws Exception {
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
    TestRunnerAction testRunnerAction = getTestAction("//standalone:simple_test");

    SpawnResult expectedSpawnResult =
        new SpawnResult.Builder()
            .setStatus(Status.SUCCESS)
            .setWallTime(Duration.ofMillis(10))
            .setRunnerName("test")
            .build();
    when(spawnStrategy.beginExecution(any(), any()))
        .thenReturn(SpawnContinuation.immediate(expectedSpawnResult));

    ActionExecutionContext actionExecutionContext =
        new FakeActionExecutionContext(createTempOutErr(tmpDirRoot), spawnStrategy, binTools);

    // actual StandaloneTestStrategy execution
    List<SpawnResult> spawnResults =
        execute(testRunnerAction, actionExecutionContext, standaloneTestStrategy);

    assertThat(spawnResults).contains(expectedSpawnResult);
    TestResult result = standaloneTestStrategy.postedResult;
    assertThat(result).isNotNull();
    assertThat(result.isCached()).isFalse();
    assertThat(result.getTestAction()).isSameInstanceAs(testRunnerAction);
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
    ExecutionOptions executionOptions = Options.getDefaults(ExecutionOptions.class);
    // TODO(ulfjack): Update this test for split xml generation.
    executionOptions.splitXmlGeneration = false;

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
    TestRunnerAction testRunnerAction = getTestAction("//standalone:simple_test");

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
    when(spawnStrategy.beginExecution(any(), any()))
        .thenReturn(
            SpawnContinuation.failedWithExecException(
                new SpawnExecException("test failed", failSpawnResult, false)))
        .thenReturn(SpawnContinuation.immediate(passSpawnResult));

    ActionExecutionContext actionExecutionContext =
        new FakeActionExecutionContext(createTempOutErr(tmpDirRoot), spawnStrategy, binTools);

    // actual StandaloneTestStrategy execution
    List<SpawnResult> spawnResults =
        execute(testRunnerAction, actionExecutionContext, standaloneTestStrategy);

    assertThat(spawnResults).containsExactly(failSpawnResult, passSpawnResult).inOrder();

    TestResult result = standaloneTestStrategy.postedResult;
    assertThat(result).isNotNull();
    assertThat(result.isCached()).isFalse();
    assertThat(result.getTestAction()).isSameInstanceAs(testRunnerAction);
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
    TestRunnerAction testRunnerAction = getTestAction("//standalone:simple_test");

    SpawnResult expectedSpawnResult =
        new SpawnResult.Builder()
            .setStatus(Status.SUCCESS)
            .setWallTime(Duration.ofMillis(10))
            .setRunnerName("remote")
            .setExecutorHostname("a-remote-host")
            .build();
    when(spawnStrategy.beginExecution(any(), any()))
        .thenReturn(SpawnContinuation.immediate(expectedSpawnResult));

    ActionExecutionContext actionExecutionContext =
        new FakeActionExecutionContext(createTempOutErr(tmpDirRoot), spawnStrategy, binTools);

    // actual StandaloneTestStrategy execution
    List<SpawnResult> spawnResults =
        execute(testRunnerAction, actionExecutionContext, standaloneTestStrategy);

    assertThat(spawnResults).contains(expectedSpawnResult);

    TestResult result = standaloneTestStrategy.postedResult;
    assertThat(result).isNotNull();
    assertThat(result.isCached()).isFalse();
    assertThat(result.getTestAction()).isSameInstanceAs(testRunnerAction);
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
    TestRunnerAction testRunnerAction = getTestAction("//standalone:simple_test");

    SpawnResult expectedSpawnResult =
        new SpawnResult.Builder()
            .setStatus(Status.SUCCESS)
            .setCacheHit(true)
            .setWallTime(Duration.ofMillis(10))
            .setRunnerName("remote cache")
            .build();
    when(spawnStrategy.beginExecution(any(), any()))
        .thenReturn(SpawnContinuation.immediate(expectedSpawnResult));

    ActionExecutionContext actionExecutionContext =
        new FakeActionExecutionContext(createTempOutErr(tmpDirRoot), spawnStrategy, binTools);

    // actual StandaloneTestStrategy execution
    List<SpawnResult> spawnResults =
        execute(testRunnerAction, actionExecutionContext, standaloneTestStrategy);

    // check that the rigged SpawnResult was returned
    assertThat(spawnResults).contains(expectedSpawnResult);

    TestResult result = standaloneTestStrategy.postedResult;
    assertThat(result).isNotNull();
    assertThat(result.isCached()).isFalse();
    assertThat(result.getTestAction()).isSameInstanceAs(testRunnerAction);
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
    ExecutionOptions executionOptions = Options.getDefaults(ExecutionOptions.class);
    executionOptions.testOutput = ExecutionOptions.TestOutputFormat.ERRORS;
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
    TestRunnerAction testRunnerAction = getTestAction("//standalone:failing_test");

    SpawnResult expectedSpawnResult =
        new SpawnResult.Builder()
            .setStatus(Status.NON_ZERO_EXIT)
            .setExitCode(1)
            .setRunnerName("test")
            .build();
    when(spawnStrategy.beginExecution(any(), any()))
        .thenAnswer(
            (invocation) -> {
              Spawn spawn = invocation.getArgument(0);
              if (spawn.getOutputFiles().size() != 1) {
                ActionExecutionContext context = invocation.getArgument(1);
                FileOutErr outErr = context.getFileOutErr();
                try (OutputStream stream = outErr.getOutputStream()) {
                  stream.write("This will not appear in the test output: bla\n".getBytes(UTF_8));
                  stream.write((TestLogHelper.HEADER_DELIMITER + "\n").getBytes(UTF_8));
                  stream.write("This will appear in the test output: foo\n".getBytes(UTF_8));
                }
                return SpawnContinuation.failedWithExecException(
                    new SpawnExecException(
                        "Failure!!",
                        expectedSpawnResult,
                        /*forciblyRunRemotely=*/ false,
                        /*catastrophe=*/ false));
              } else {
                return SpawnContinuation.immediate(
                    new SpawnResult.Builder()
                        .setStatus(Status.SUCCESS)
                        .setRunnerName("test")
                        .build());
              }
            });

    FileOutErr outErr = createTempOutErr(tmpDirRoot);
    ActionExecutionContext actionExecutionContext =
        new FakeActionExecutionContext(outErr, spawnStrategy, binTools);

    // actual StandaloneTestStrategy execution
    List<SpawnResult> spawnResults =
        execute(testRunnerAction, actionExecutionContext, standaloneTestStrategy);

    // check that the rigged SpawnResult was returned
    assertThat(spawnResults).contains(expectedSpawnResult);
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
      String outData = FileSystemUtils.readContent(outErr.getOutputPath(), UTF_8);
      assertThat(outData)
          .contains("==================== Test output for //standalone:failing_test:");
      assertThat(outData).doesNotContain("bla");
      assertThat(outData).doesNotContain(TestLogHelper.HEADER_DELIMITER);
      assertThat(outData).contains("foo");
      assertThat(outData)
          .contains(
              "================================================================================");
    } catch (IOException e) {
      fail("Test stdout file missing: " + outErr.getOutputPath());
    }
    assertThat(outErr.getErrorPath().exists()).isFalse();
  }

  @Test
  public void testThatTestLogAndOutputAreReturnedWithSplitXmlGeneration() throws Exception {
    ExecutionOptions executionOptions = Options.getDefaults(ExecutionOptions.class);
    executionOptions.testOutput = ExecutionOptions.TestOutputFormat.ERRORS;
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
        "    tags = [\"local\"],",
        ")");
    TestRunnerAction testRunnerAction = getTestAction("//standalone:failing_test");

    SpawnResult testSpawnResult =
        new SpawnResult.Builder()
            .setStatus(Status.NON_ZERO_EXIT)
            .setExitCode(1)
            .setRunnerName("test")
            .build();
    SpawnResult xmlGeneratorSpawnResult =
        new SpawnResult.Builder()
            .setStatus(Status.SUCCESS)
            .setRunnerName("test")
            .build();
    List<FileOutErr> called = new ArrayList<>();
    when(spawnStrategy.beginExecution(any(), any()))
        .thenAnswer(
            (invocation) -> {
              Spawn spawn = invocation.getArgument(0);
              // Test that both spawns have the local tag attached as a execution info
              assertThat(spawn.getExecutionInfo()).containsKey("local");
              ActionExecutionContext context = invocation.getArgument(1);
              FileOutErr outErr = context.getFileOutErr();
              called.add(outErr);
              if (spawn.getOutputFiles().size() != 1) {
                try (OutputStream stream = outErr.getOutputStream()) {
                  stream.write("This will not appear in the test output: bla\n".getBytes(UTF_8));
                  stream.write((TestLogHelper.HEADER_DELIMITER + "\n").getBytes(UTF_8));
                  stream.write("This will appear in the test output: foo\n".getBytes(UTF_8));
                }
                return SpawnContinuation.failedWithExecException(
                    new SpawnExecException(
                        "Failure!!",
                        testSpawnResult,
                        /*forciblyRunRemotely=*/ false,
                        /*catastrophe=*/ false));
              } else {
                String testName =
                    OS.getCurrent() == OS.WINDOWS
                        ? "standalone/failing_test.exe"
                        : "standalone/failing_test";
                assertThat(spawn.getEnvironment()).containsEntry("TEST_BINARY", testName);
                return SpawnContinuation.immediate(xmlGeneratorSpawnResult);
              }
            });

    FileOutErr outErr = createTempOutErr(tmpDirRoot);
    ActionExecutionContext actionExecutionContext =
        new FakeActionExecutionContext(outErr, spawnStrategy, binTools);

    // actual StandaloneTestStrategy execution
    List<SpawnResult> spawnResults =
        execute(testRunnerAction, actionExecutionContext, standaloneTestStrategy);

    // check that the rigged SpawnResult was returned
    assertThat(spawnResults).containsExactly(testSpawnResult, xmlGeneratorSpawnResult);
    // check that the test log contains all the output
    String logData = FileSystemUtils.readContent(testRunnerAction.getTestLog().getPath(), UTF_8);
    assertThat(logData).contains("bla");
    assertThat(logData).contains(TestLogHelper.HEADER_DELIMITER);
    assertThat(logData).contains("foo");
    // check that the test stdout contains all the expected output
    outErr.close(); // Create the output files.
    String outData = FileSystemUtils.readContent(outErr.getOutputPath(), UTF_8);
    assertThat(outData)
        .contains("==================== Test output for //standalone:failing_test:");
    assertThat(outData).doesNotContain("bla");
    assertThat(outData).doesNotContain(TestLogHelper.HEADER_DELIMITER);
    assertThat(outData).contains("foo");
    assertThat(outData)
        .contains(
            "================================================================================");
    assertThat(outErr.getErrorPath().exists()).isFalse();
    assertThat(called).hasSize(2);
    assertThat(called).containsNoDuplicates();
  }

  @Test
  public void testEmptyOutputCreatesEmptyLogFile() throws Exception {
    ExecutionOptions executionOptions = Options.getDefaults(ExecutionOptions.class);
    executionOptions.testOutput = ExecutionOptions.TestOutputFormat.ALL;
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
    TestRunnerAction testRunnerAction = getTestAction("//standalone:empty_test");

    SpawnResult expectedSpawnResult =
        new SpawnResult.Builder().setStatus(Status.SUCCESS).setRunnerName("test").build();
    when(spawnStrategy.beginExecution(any(), any()))
        .thenReturn(SpawnContinuation.immediate(expectedSpawnResult));

    FileOutErr outErr = createTempOutErr(tmpDirRoot);
    ActionExecutionContext actionExecutionContext =
        new FakeActionExecutionContext(outErr, spawnStrategy, binTools);

    // actual StandaloneTestStrategy execution
    List<SpawnResult> spawnResults =
        execute(testRunnerAction, actionExecutionContext, standaloneTestStrategy);

    // check that the rigged SpawnResult was returned
    assertThat(spawnResults).contains(expectedSpawnResult);
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
      String outData = FileSystemUtils.readContent(outErr.getOutputPath(), UTF_8);
      String emptyOutput =
          "==================== Test output for //standalone:empty_test:(\\s)*"
              + "================================================================================(\\s)*";
      assertThat(outData).matches(emptyOutput);
    } catch (IOException e) {
      fail("Test stdout file missing: " + outErr.getOutputPath());
    }
    assertThat(outErr.getErrorPath().exists()).isFalse();
  }

  @Test
  public void testAppendStdErrDoesNotBusyLoop() throws Exception {
    ExecutionOptions executionOptions = Options.getDefaults(ExecutionOptions.class);
    executionOptions.testOutput = ExecutionOptions.TestOutputFormat.ALL;
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
    TestRunnerAction testRunnerAction = getTestAction("//standalone:empty_test");

    SpawnResult expectedSpawnResult =
        new SpawnResult.Builder().setStatus(Status.SUCCESS).setRunnerName("test").build();
    when(spawnStrategy.beginExecution(any(), any()))
        .then(
            (invocation) -> {
              ((ActionExecutionContext) invocation.getArgument(1)).getFileOutErr().printErr("Foo");
              return SpawnContinuation.immediate(expectedSpawnResult);
            });

    FileOutErr outErr = createTempOutErr(tmpDirRoot);
    ActionExecutionContext actionExecutionContext =
        new FakeActionExecutionContext(outErr, spawnStrategy, binTools);

    // actual StandaloneTestStrategy execution
    execute(testRunnerAction, actionExecutionContext, standaloneTestStrategy);

    // check that the test stdout contains all the expected output
    try {
      String outData = FileSystemUtils.readContent(outErr.getOutputPath(), UTF_8);
      assertThat(outData).contains("Foo");
    } catch (IOException e) {
      fail("Test stdout file missing: " + outErr.getOutputPath());
    }
  }

  @Test
  public void testExperimentalCancelConcurrentTests() throws Exception {
    useConfiguration(
        "--runs_per_test=2",
        "--runs_per_test_detects_flakes",
        "--experimental_cancel_concurrent_tests");
    ExecutionOptions executionOptions = Options.getDefaults(ExecutionOptions.class);
    Path tmpDirRoot = TestStrategy.getTmpRoot(rootDirectory, outputBase, executionOptions);
    BinTools binTools = BinTools.forUnitTesting(directories, analysisMock.getEmbeddedTools());
    TestedStandaloneTestStrategy standaloneTestStrategy =
        new TestedStandaloneTestStrategy(executionOptions, binTools, tmpDirRoot);

    scratch.file("standalone/empty_test.sh", "this does not get executed, it is mocked out");
    scratch.file(
        "standalone/BUILD",
        "sh_test(",
        "    name = \"empty_test\",",
        "    size = \"small\",",
        "    srcs = [\"empty_test.sh\"],",
        ")");
    List<TestRunnerAction> testRunnerActions = getTestActions("//standalone:empty_test");
    assertThat(testRunnerActions).hasSize(2);

    TestRunnerAction actionA = testRunnerActions.get(0);
    TestRunnerAction actionB = testRunnerActions.get(1);
    ListenableFuture<Void> cancelFuture =
        standaloneTestStrategy.getTestCancelFuture(actionA.getOwner(), actionA.getShardNum());
    assertThat(cancelFuture)
        .isSameInstanceAs(
            standaloneTestStrategy.getTestCancelFuture(actionB.getOwner(), actionB.getShardNum()));

    SpawnResult expectedSpawnResult =
        new SpawnResult.Builder().setStatus(Status.SUCCESS).setRunnerName("test").build();
    when(spawnStrategy.beginExecution(any(), any()))
        .then(
            (invocation) -> {
              // Avoid triggering split XML generation by creating an empty XML file.
              FileSystemUtils.touchFile(actionA.resolve(getExecRoot()).getXmlOutputPath());
              return SpawnContinuation.immediate(expectedSpawnResult);
            });

    ActionExecutionContext actionExecutionContext =
        new FakeActionExecutionContext(createTempOutErr(tmpDirRoot), spawnStrategy, binTools);
    List<SpawnResult> resultA = execute(actionA, actionExecutionContext, standaloneTestStrategy);
    assertThat(cancelFuture.isCancelled()).isTrue();
    verify(spawnStrategy).beginExecution(any(), any());
    assertThat(resultA).hasSize(1);
    assertThat(standaloneTestStrategy.postedResult).isNotNull();
    assertThat(standaloneTestStrategy.postedResult.getData().getStatus())
        .isEqualTo(BlazeTestStatus.PASSED);
    assertThat(storedEvents.getEvents())
        .contains(Event.of(EventKind.PASS, null, "//standalone:empty_test (run 1 of 2)"));
    // Reset postedResult.
    standaloneTestStrategy.postedResult = null;

    when(spawnStrategy.beginExecution(any(), any()))
        .thenThrow(new AssertionError("failure: this should not have been called"));
    List<SpawnResult> resultB = execute(actionB, actionExecutionContext, standaloneTestStrategy);
    assertThat(resultB).isEmpty();
    assertThat(standaloneTestStrategy.postedResult).isNotNull();
    assertThat(standaloneTestStrategy.postedResult.getData().getStatus())
        .isEqualTo(BlazeTestStatus.INCOMPLETE);
    assertThat(storedEvents.getEvents())
        .contains(Event.of(EventKind.CANCELLED, null, "//standalone:empty_test (run 2 of 2)"));
    // Check that there are no ERROR events.
    assertThat(
            storedEvents.getEvents().stream()
                .filter((e) -> e.getKind() == EventKind.ERROR)
                .collect(Collectors.toList()))
        .isEmpty();
  }

  @Test
  public void testExperimentalCancelConcurrentTestsDoesNotTriggerOnFailedRun() throws Exception {
    useConfiguration(
        "--runs_per_test=2",
        "--runs_per_test_detects_flakes",
        "--experimental_cancel_concurrent_tests");
    ExecutionOptions executionOptions = Options.getDefaults(ExecutionOptions.class);
    Path tmpDirRoot = TestStrategy.getTmpRoot(rootDirectory, outputBase, executionOptions);
    BinTools binTools = BinTools.forUnitTesting(directories, analysisMock.getEmbeddedTools());
    TestedStandaloneTestStrategy standaloneTestStrategy =
        new TestedStandaloneTestStrategy(executionOptions, binTools, tmpDirRoot);

    scratch.file("standalone/empty_test.sh", "this does not get executed, it is mocked out");
    scratch.file(
        "standalone/BUILD",
        "sh_test(",
        "    name = \"empty_test\",",
        "    size = \"small\",",
        "    srcs = [\"empty_test.sh\"],",
        ")");
    List<TestRunnerAction> testRunnerActions = getTestActions("//standalone:empty_test");
    assertThat(testRunnerActions).hasSize(2);

    TestRunnerAction actionA = testRunnerActions.get(0);
    TestRunnerAction actionB = testRunnerActions.get(1);
    ListenableFuture<Void> cancelFuture =
        standaloneTestStrategy.getTestCancelFuture(actionA.getOwner(), actionA.getShardNum());
    assertThat(cancelFuture)
        .isSameInstanceAs(
            standaloneTestStrategy.getTestCancelFuture(actionB.getOwner(), actionB.getShardNum()));
    assertThat(cancelFuture.isCancelled()).isFalse();

    SpawnResult expectedSpawnResultA =
        new SpawnResult.Builder()
            .setStatus(Status.NON_ZERO_EXIT)
            .setExitCode(1)
            .setRunnerName("test")
            .build();
    when(spawnStrategy.beginExecution(any(), any()))
        .then(
            (invocation) -> {
              // Avoid triggering split XML generation by creating an empty XML file.
              FileSystemUtils.touchFile(actionA.resolve(getExecRoot()).getXmlOutputPath());
              return SpawnContinuation.failedWithExecException(
                  new SpawnExecException("", expectedSpawnResultA, false));
            });

    ActionExecutionContext actionExecutionContext =
        new FakeActionExecutionContext(createTempOutErr(tmpDirRoot), spawnStrategy, binTools);
    List<SpawnResult> resultA = execute(actionA, actionExecutionContext, standaloneTestStrategy);
    assertThat(cancelFuture.isCancelled()).isFalse();
    verify(spawnStrategy).beginExecution(any(), any());
    assertThat(resultA).hasSize(1);
    assertThat(standaloneTestStrategy.postedResult).isNotNull();
    assertThat(standaloneTestStrategy.postedResult.getData().getStatus())
        .isEqualTo(BlazeTestStatus.FAILED);
    assertContainsPrefixedEvent(
        storedEvents.getEvents(),
        Event.of(EventKind.FAIL, null, "//standalone:empty_test (run 1 of 2)"));
    // Reset postedResult.
    standaloneTestStrategy.postedResult = null;

    SpawnResult expectedSpawnResultB =
        new SpawnResult.Builder().setStatus(Status.SUCCESS).setRunnerName("test").build();
    when(spawnStrategy.beginExecution(any(), any()))
        .then(
            (invocation) -> {
              // Avoid triggering split XML generation by creating an empty XML file.
              FileSystemUtils.touchFile(actionB.resolve(getExecRoot()).getXmlOutputPath());
              return SpawnContinuation.immediate(expectedSpawnResultB);
            });
    List<SpawnResult> resultB = execute(actionB, actionExecutionContext, standaloneTestStrategy);
    assertThat(cancelFuture.isCancelled()).isTrue();
    assertThat(resultB).hasSize(1);
    assertThat(standaloneTestStrategy.postedResult).isNotNull();
    assertThat(standaloneTestStrategy.postedResult.getData().getStatus())
        .isEqualTo(BlazeTestStatus.PASSED);
    assertThat(storedEvents.getEvents())
        .contains(Event.of(EventKind.PASS, null, "//standalone:empty_test (run 2 of 2)"));
  }

  private static void assertContainsPrefixedEvent(Iterable<Event> events, Event event) {
    for (Event e : events) {
      if (e.getKind() == event.getKind() && e.getMessage().startsWith(event.getMessage())) {
        return;
      }
    }
    assertThat(events).contains(event);
  }

  @Test
  public void testExperimentalCancelConcurrentTestsAllFailed() throws Exception {
    useConfiguration(
        "--runs_per_test=2",
        "--runs_per_test_detects_flakes",
        "--experimental_cancel_concurrent_tests");
    ExecutionOptions executionOptions = Options.getDefaults(ExecutionOptions.class);
    Path tmpDirRoot = TestStrategy.getTmpRoot(rootDirectory, outputBase, executionOptions);
    BinTools binTools = BinTools.forUnitTesting(directories, analysisMock.getEmbeddedTools());
    TestedStandaloneTestStrategy standaloneTestStrategy =
        new TestedStandaloneTestStrategy(executionOptions, binTools, tmpDirRoot);

    scratch.file("standalone/empty_test.sh", "this does not get executed, it is mocked out");
    scratch.file(
        "standalone/BUILD",
        "sh_test(",
        "    name = \"empty_test\",",
        "    size = \"small\",",
        "    srcs = [\"empty_test.sh\"],",
        ")");
    List<TestRunnerAction> testRunnerActions = getTestActions("//standalone:empty_test");
    assertThat(testRunnerActions).hasSize(2);

    TestRunnerAction actionA = testRunnerActions.get(0);
    TestRunnerAction actionB = testRunnerActions.get(1);
    ListenableFuture<Void> cancelFuture =
        standaloneTestStrategy.getTestCancelFuture(actionA.getOwner(), actionA.getShardNum());
    assertThat(cancelFuture)
        .isSameInstanceAs(
            standaloneTestStrategy.getTestCancelFuture(actionB.getOwner(), actionB.getShardNum()));
    assertThat(cancelFuture.isCancelled()).isFalse();

    SpawnResult expectedSpawnResult =
        new SpawnResult.Builder()
            .setStatus(Status.NON_ZERO_EXIT)
            .setExitCode(1)
            .setRunnerName("test")
            .build();
    when(spawnStrategy.beginExecution(any(), any()))
        .then(
            (invocation) -> {
              // Avoid triggering split XML generation by creating an empty XML file.
              FileSystemUtils.touchFile(actionA.resolve(getExecRoot()).getXmlOutputPath());
              return SpawnContinuation.failedWithExecException(
                  new SpawnExecException("", expectedSpawnResult, false));
            });

    ActionExecutionContext actionExecutionContext =
        new FakeActionExecutionContext(createTempOutErr(tmpDirRoot), spawnStrategy, binTools);
    List<SpawnResult> resultA = execute(actionA, actionExecutionContext, standaloneTestStrategy);
    assertThat(cancelFuture.isCancelled()).isFalse();
    verify(spawnStrategy).beginExecution(any(), any());
    assertThat(resultA).hasSize(1);
    assertThat(standaloneTestStrategy.postedResult).isNotNull();
    assertThat(standaloneTestStrategy.postedResult.getData().getStatus())
        .isEqualTo(BlazeTestStatus.FAILED);
    assertContainsPrefixedEvent(
        storedEvents.getEvents(),
        Event.of(EventKind.FAIL, null, "//standalone:empty_test (run 1 of 2)"));
    // Reset postedResult.
    standaloneTestStrategy.postedResult = null;

    when(spawnStrategy.beginExecution(any(), any()))
        .then(
            (invocation) -> {
              // Avoid triggering split XML generation by creating an empty XML file.
              FileSystemUtils.touchFile(actionB.resolve(getExecRoot()).getXmlOutputPath());
              return SpawnContinuation.failedWithExecException(
                  new SpawnExecException("", expectedSpawnResult, false));
            });
    List<SpawnResult> resultB = execute(actionB, actionExecutionContext, standaloneTestStrategy);
    assertThat(cancelFuture.isCancelled()).isFalse();
    assertThat(resultB).hasSize(1);
    assertThat(standaloneTestStrategy.postedResult).isNotNull();
    assertThat(standaloneTestStrategy.postedResult.getData().getStatus())
        .isEqualTo(BlazeTestStatus.FAILED);
    assertContainsPrefixedEvent(
        storedEvents.getEvents(),
        Event.of(EventKind.FAIL, null, "//standalone:empty_test (run 2 of 2)"));
  }
}
