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
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BinTools;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.analysis.test.TestResult;
import com.google.devtools.build.lib.analysis.test.TestRunnerAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.List;
import java.util.Set;
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

    scratch.file(
        "standalone/simple_test.sh", "echo \"All tests passed, you are awesome!\"", "exit 0");

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

    // setup a mock ActionExecutionContext

    when(actionExecutionContext.getClock()).thenReturn(BlazeClock.instance());
    when(actionExecutionContext.withFileOutErr(any())).thenReturn(actionExecutionContext);
    when(actionExecutionContext.getExecRoot()).thenReturn(outputBase.getRelative("execroot"));
    when(actionExecutionContext.getClientEnv()).thenReturn(ImmutableMap.of());
    when(actionExecutionContext.getEventHandler()).thenReturn(reporter);
    when(actionExecutionContext.getEventBus()).thenReturn(eventBus);

    long expectedWallTimeMillis = 10;
    SpawnResult expectedSpawnResult =
        new SpawnResult.Builder()
            .setStatus(SpawnResult.Status.SUCCESS)
            .setWallTimeMillis(expectedWallTimeMillis)
            .build();
    when(spawnActionContext.exec(any(), any())).thenReturn(ImmutableSet.of(expectedSpawnResult));

    when(actionExecutionContext.getSpawnActionContext(any())).thenReturn(spawnActionContext);

    // actual StandaloneTestStrategy execution

    Set<SpawnResult> spawnResults =
        standaloneTestStrategy.exec(testRunnerAction, actionExecutionContext);

    // check that the rigged SpawnResult was returned

    assertThat(spawnResults).containsExactly(expectedSpawnResult);
    SpawnResult spawnResult = Iterables.getOnlyElement(spawnResults);
    assertThat(spawnResult.status()).isEqualTo(SpawnResult.Status.SUCCESS);
    assertThat(spawnResult.getWallTimeMillis()).isEqualTo(expectedWallTimeMillis);
  }
}
