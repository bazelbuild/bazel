// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.dynamic;

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.truth.Truth.assertThat;
import static java.util.concurrent.TimeUnit.MILLISECONDS;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNotNull;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyZeroInteractions;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.DynamicStrategyRegistry;
import com.google.devtools.build.lib.actions.SandboxedSpawnStrategy;
import com.google.devtools.build.lib.actions.SandboxedSpawnStrategy.StopConcurrentSpawns;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.exec.ExecutionPolicy;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.server.FailureDetails.Execution;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.testutil.TestFileOutErr;
import com.google.devtools.build.lib.testutil.TestUtils;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.function.Function;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

/** Unit tests for {@link DynamicSpawnStrategy}. */
@RunWith(JUnit4.class)
public class DynamicSpawnStrategyUnitTest {

  private static final SpawnResult SUCCESSFUL_SPAWN_RESULT =
      new SpawnResult.Builder().setRunnerName("test").setStatus(Status.SUCCESS).build();
  private static final FailureDetail FAILURE_DETAIL =
      FailureDetail.newBuilder().setExecution(Execution.getDefaultInstance()).build();

  private ExecutorService executorServiceForCleanup;

  @Mock private Function<Spawn, Optional<Spawn>> mockGetPostProcessingSpawn;

  @Before
  public void initMocks() {
    MockitoAnnotations.initMocks(this);
  }

  @After
  public void stopExecutorService() throws InterruptedException {
    executorServiceForCleanup.shutdown();
    assertThat(
            executorServiceForCleanup.awaitTermination(
                TestUtils.WAIT_TIMEOUT_MILLISECONDS, MILLISECONDS))
        .isTrue();
  }

  @Test
  public void exec_remoteOnlySpawn_doesNotGetLocalPostProcessingSpawn() throws Exception {
    DynamicSpawnStrategy dynamicSpawnStrategy =
        createDynamicSpawnStrategy(
            ExecutionPolicy.REMOTE_EXECUTION_ONLY, mockGetPostProcessingSpawn);
    SandboxedSpawnStrategy local = createMockSpawnStrategy();
    SandboxedSpawnStrategy remote = createMockSpawnStrategy();
    ArgumentCaptor<Spawn> remoteSpawnCaptor = ArgumentCaptor.forClass(Spawn.class);
    when(remote.exec(remoteSpawnCaptor.capture(), any(), any()))
        .thenReturn(ImmutableList.of(SUCCESSFUL_SPAWN_RESULT));
    ActionExecutionContext actionExecutionContext = createMockActionExecutionContext(local, remote);
    Spawn spawn = new SpawnBuilder().build();

    ImmutableList<SpawnResult> results = dynamicSpawnStrategy.exec(spawn, actionExecutionContext);

    assertThat(results).containsExactly(SUCCESSFUL_SPAWN_RESULT);
    verify(mockGetPostProcessingSpawn, never()).apply(any());
    verify(local, never()).exec(any(), any(), any());
    assertThat(remoteSpawnCaptor.getAllValues()).containsExactly(spawn);
  }

  @Test
  public void exec_localOnlySpawn_runsLocalPostProcessingSpawn() throws Exception {
    Spawn spawn = new SpawnBuilder("command").build();
    Spawn postProcessingSpawn = new SpawnBuilder("extra_command").build();
    DynamicSpawnStrategy dynamicSpawnStrategy =
        createDynamicSpawnStrategy(
            ExecutionPolicy.LOCAL_EXECUTION_ONLY, ignored -> Optional.of(postProcessingSpawn));
    SandboxedSpawnStrategy local = createMockSpawnStrategy();
    ArgumentCaptor<Spawn> localSpawnCaptor = ArgumentCaptor.forClass(Spawn.class);
    when(local.exec(localSpawnCaptor.capture(), any(), any()))
        .thenReturn(ImmutableList.of(SUCCESSFUL_SPAWN_RESULT));
    SandboxedSpawnStrategy remote = createMockSpawnStrategy();
    ActionExecutionContext actionExecutionContext = createMockActionExecutionContext(local, remote);

    ImmutableList<SpawnResult> results = dynamicSpawnStrategy.exec(spawn, actionExecutionContext);

    assertThat(results).containsExactly(SUCCESSFUL_SPAWN_RESULT, SUCCESSFUL_SPAWN_RESULT);
    verifyZeroInteractions(remote);
    assertThat(localSpawnCaptor.getAllValues())
        .containsExactly(spawn, postProcessingSpawn)
        .inOrder();
  }

  @Test
  public void exec_failedLocalSpawn_doesNotGetLocalPostProcessingSpawn() throws Exception {
    testExecFailedLocalSpawnDoesNotGetLocalPostProcessingSpawn(
        new SpawnResult.Builder()
            .setRunnerName("test")
            .setStatus(Status.TIMEOUT)
            .setExitCode(SpawnResult.POSIX_TIMEOUT_EXIT_CODE)
            .setFailureDetail(FAILURE_DETAIL)
            .build());
  }

  @Test
  public void exec_nonZeroExitCodeLocalSpawn_doesNotGetLocalPostProcessingSpawn() throws Exception {
    testExecFailedLocalSpawnDoesNotGetLocalPostProcessingSpawn(
        new SpawnResult.Builder()
            .setRunnerName("test")
            .setStatus(Status.EXECUTION_FAILED)
            .setExitCode(123)
            .setFailureDetail(FAILURE_DETAIL)
            .build());
  }

  private void testExecFailedLocalSpawnDoesNotGetLocalPostProcessingSpawn(SpawnResult failedResult)
      throws Exception {
    DynamicSpawnStrategy dynamicSpawnStrategy =
        createDynamicSpawnStrategy(
            ExecutionPolicy.LOCAL_EXECUTION_ONLY, mockGetPostProcessingSpawn);
    SandboxedSpawnStrategy local = createMockSpawnStrategy();
    ArgumentCaptor<Spawn> localSpawnCaptor = ArgumentCaptor.forClass(Spawn.class);
    when(local.exec(localSpawnCaptor.capture(), any(), any()))
        .thenReturn(ImmutableList.of(failedResult));
    SandboxedSpawnStrategy remote = createMockSpawnStrategy();
    ActionExecutionContext actionExecutionContext = createMockActionExecutionContext(local, remote);
    Spawn spawn = new SpawnBuilder().build();

    ImmutableList<SpawnResult> results = dynamicSpawnStrategy.exec(spawn, actionExecutionContext);

    assertThat(results).containsExactly(failedResult);
    assertThat(localSpawnCaptor.getAllValues()).containsExactly(spawn);
    verify(remote, never()).exec(any(), any(), any());
    verify(mockGetPostProcessingSpawn, never()).apply(any());
  }

  @Test
  public void exec_runAnywhereSpawn_runsLocalPostProcessingSpawn() throws Exception {
    Spawn spawn = new SpawnBuilder().build();
    Spawn postProcessingSpawn = new SpawnBuilder("extra_command").build();
    DynamicSpawnStrategy dynamicSpawnStrategy =
        createDynamicSpawnStrategy(
            ExecutionPolicy.ANYWHERE, ignored -> Optional.of(postProcessingSpawn));
    SandboxedSpawnStrategy local = createMockSpawnStrategy();
    // Make sure that local execution does not win the race before remote starts.
    Semaphore remoteStarted = new Semaphore(0);
    // Only the first spawn should be able to stop the concurrent remote execution (get the output
    // lock).
    when(local.exec(eq(spawn), any(), /*stopConcurrentSpawns=*/ isNotNull()))
        .thenAnswer(
            invocation -> {
              remoteStarted.acquire();
              StopConcurrentSpawns stopConcurrentSpawns = invocation.getArgument(2);
              stopConcurrentSpawns.stop();
              return ImmutableList.of(SUCCESSFUL_SPAWN_RESULT);
            });
    when(local.exec(eq(postProcessingSpawn), any(), /*stopConcurrentSpawns=*/ isNull()))
        .thenReturn(ImmutableList.of(SUCCESSFUL_SPAWN_RESULT));
    SandboxedSpawnStrategy remote = createMockSpawnStrategy();
    when(remote.exec(eq(spawn), any(), any()))
        .thenAnswer(
            invocation -> {
              remoteStarted.release();
              Thread.sleep(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
              throw new AssertionError("Timed out waiting for interruption");
            });
    ActionExecutionContext actionExecutionContext = createMockActionExecutionContext(local, remote);

    ImmutableList<SpawnResult> results = dynamicSpawnStrategy.exec(spawn, actionExecutionContext);

    assertThat(results).containsExactly(SUCCESSFUL_SPAWN_RESULT, SUCCESSFUL_SPAWN_RESULT);
  }

  private DynamicSpawnStrategy createDynamicSpawnStrategy(
      ExecutionPolicy executionPolicy,
      Function<Spawn, Optional<Spawn>> getPostProcessingSpawnForLocalExecution) {
    checkState(
        executorServiceForCleanup == null,
        "Creating the DynamicSpawnStrategy twice in the same test is not supported.");
    executorServiceForCleanup = Executors.newCachedThreadPool();
    return new DynamicSpawnStrategy(
        executorServiceForCleanup,
        new DynamicExecutionOptions(),
        ignored -> executionPolicy,
        getPostProcessingSpawnForLocalExecution);
  }

  private static ActionExecutionContext createMockActionExecutionContext(
      SandboxedSpawnStrategy localStrategy, SandboxedSpawnStrategy remoteStrategy) {
    ActionExecutionContext actionExecutionContext = mock(ActionExecutionContext.class);
    when(actionExecutionContext.getFileOutErr()).thenReturn(new TestFileOutErr());
    when(actionExecutionContext.getContext(DynamicStrategyRegistry.class))
        .thenReturn(
            new DynamicStrategyRegistry() {
              @Override
              public ImmutableList<SandboxedSpawnStrategy> getDynamicSpawnActionContexts(
                  Spawn spawn, DynamicMode dynamicMode) {
                switch (dynamicMode) {
                  case LOCAL:
                    return ImmutableList.of(localStrategy);
                  case REMOTE:
                    return ImmutableList.of(remoteStrategy);
                }
                throw new AssertionError("Unexpected mode: " + dynamicMode);
              }

              @Override
              public void notifyUsedDynamic(ActionContextRegistry actionContextRegistry) {}
            });
    when(actionExecutionContext.withFileOutErr(any())).thenReturn(actionExecutionContext);
    return actionExecutionContext;
  }

  private static SandboxedSpawnStrategy createMockSpawnStrategy() throws InterruptedException {
    SandboxedSpawnStrategy strategy = mock(SandboxedSpawnStrategy.class);
    when(strategy.canExec(any(), any())).thenReturn(true);
    when(strategy.beginExecution(any(), any())).thenThrow(UnsupportedOperationException.class);
    return strategy;
  }
}
