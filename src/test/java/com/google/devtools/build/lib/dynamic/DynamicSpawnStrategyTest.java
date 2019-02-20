// Copyright 2018 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.BaseSpawn;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.ExecutorInitException;
import com.google.devtools.build.lib.actions.LocalHostCapacity;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SandboxedSpawnActionContext;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil.NullAction;
import com.google.devtools.build.lib.exec.ExecutionPolicy;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DynamicSpawnStrategy}. */
@RunWith(JUnit4.class)
public class DynamicSpawnStrategyTest {
  protected FileSystem fileSystem;
  protected Path testRoot;
  private ExecutorService executorService;
  private MockLocalSpawnStrategy localStrategy;
  private MockRemoteSpawnStrategy remoteStrategy;
  private SpawnActionContext dynamicSpawnStrategy;
  private Artifact inputArtifact;
  private Artifact outputArtifact;
  private FileOutErr outErr;
  private ActionExecutionContext actionExecutionContext;
  private DynamicExecutionOptions options;
  private final ActionKeyContext actionKeyContext = new ActionKeyContext();

  abstract static class MockSpawnStrategy implements SandboxedSpawnActionContext {
    private final Path testRoot;
    private final int delayMs;
    private volatile Spawn executedSpawn;
    private CountDownLatch succeeded = new CountDownLatch(1);
    private boolean failsDuringExecution;
    private CountDownLatch beforeExecutionWaitFor;
    private Callable<List<SpawnResult>> execute;

    public MockSpawnStrategy(Path testRoot, int delayMs) {
      this.testRoot = testRoot;
      this.delayMs = delayMs;
    }

    @Override
    public List<SpawnResult> exec(Spawn spawn, ActionExecutionContext actionExecutionContext)
        throws ExecException, InterruptedException {
      return exec(spawn, actionExecutionContext, null);
    }

    @Override
    public boolean canExec(Spawn spawn) {
      return true;
    }

    @Override
    public List<SpawnResult> exec(
        Spawn spawn,
        ActionExecutionContext actionExecutionContext,
        AtomicReference<Class<? extends SpawnActionContext>> writeOutputFiles)
        throws ExecException, InterruptedException {
      executedSpawn = spawn;

      if (beforeExecutionWaitFor != null) {
        beforeExecutionWaitFor.countDown();
        beforeExecutionWaitFor.await();
      }

      if (delayMs > 0) {
        Thread.sleep(delayMs);
      }

      List<SpawnResult> spawnResults = ImmutableList.of();
      if (execute != null) {
        try {
          spawnResults = execute.call();
        } catch (ExecException | InterruptedException e) {
          throw e;
        } catch (Exception e) {
          Throwables.throwIfUnchecked(e);
          throw new IllegalStateException(e);
        }
      }
      if (failsDuringExecution) {
        try {
          FileSystemUtils.appendIsoLatin1(
              actionExecutionContext.getFileOutErr().getOutputPath(),
              "action failed with " + getClass().getSimpleName());
        } catch (IOException e) {
          throw new IllegalStateException(e);
        }
        throw new UserExecException(getClass().getSimpleName() + " failed to execute the Spawn");
      }

      if (writeOutputFiles != null && !writeOutputFiles.compareAndSet(null, getClass())) {
        throw new InterruptedException(getClass() + " could not acquire barrier");
      } else {
        for (ActionInput output : spawn.getOutputFiles()) {
          try {
            FileSystemUtils.writeIsoLatin1(
                testRoot.getRelative(output.getExecPath()), getClass().getSimpleName());
          } catch (IOException e) {
            throw new IllegalStateException(e);
          }
        }
      }

      try {
        FileSystemUtils.appendIsoLatin1(
            actionExecutionContext.getFileOutErr().getOutputPath(),
            "output files written with " + getClass().getSimpleName());
      } catch (IOException e) {
        throw new IllegalStateException(e);
      }

      succeeded.countDown();

      return spawnResults;
    }

    public Spawn getExecutedSpawn() {
      return executedSpawn;
    }

    boolean succeeded() {
      return succeeded.getCount() == 0;
    }

    CountDownLatch getSucceededLatch() {
      return succeeded;
    }

    public void failsDuringExecution() {
      failsDuringExecution = true;
    }

    public void beforeExecutionWaitFor(CountDownLatch countDownLatch) {
      beforeExecutionWaitFor = countDownLatch;
    }

    void setExecute(Callable<List<SpawnResult>> execute) {
      this.execute = execute;
    }
  }

  @ExecutionStrategy(
    name = {"mock-remote"},
    contextType = SpawnActionContext.class
  )
  private static class MockRemoteSpawnStrategy extends MockSpawnStrategy {
    public MockRemoteSpawnStrategy(Path testRoot, int delayMs) {
      super(testRoot, delayMs);
    }
  }

  @ExecutionStrategy(
    name = {"mock-local"},
    contextType = SpawnActionContext.class
  )
  private static class MockLocalSpawnStrategy extends MockSpawnStrategy {
    public MockLocalSpawnStrategy(Path testRoot, int delayMs) {
      super(testRoot, delayMs);
    }
  }

  private static class DynamicSpawnStrategyUnderTest extends DynamicSpawnStrategy {
    public DynamicSpawnStrategyUnderTest(
        ExecutorService executorService,
        DynamicExecutionOptions options,
        Function<Spawn, ExecutionPolicy> executionPolicy) {
      super(executorService, options, executionPolicy);
    }
  }

  @Before
  public void setUp() throws Exception {
    ResourceManager.instance().setAvailableResources(LocalHostCapacity.getLocalHostCapacity());
    ResourceManager.instance()
        .setRamUtilizationPercentage(ResourceManager.DEFAULT_RAM_UTILIZATION_PERCENTAGE);
    ResourceManager.instance().resetResourceUsage();

    fileSystem = FileSystems.getNativeFileSystem();
    testRoot = fileSystem.getPath(TestUtils.tmpDir());
    FileSystemUtils.deleteTreesBelow(testRoot);
    executorService = Executors.newCachedThreadPool();
    inputArtifact =
        new Artifact(
            PathFragment.create("input.txt"), ArtifactRoot.asSourceRoot(Root.fromPath(testRoot)));
    outputArtifact =
        new Artifact(
            PathFragment.create("output.txt"), ArtifactRoot.asSourceRoot(Root.fromPath(testRoot)));
    outErr = new FileOutErr(testRoot.getRelative("stdout"), testRoot.getRelative("stderr"));
    actionExecutionContext =
        ActionsTestUtil.createContext(null, actionKeyContext, outErr, testRoot, null, null);
  }

  void createSpawnStrategy(int localDelay, int remoteDelay) throws ExecutorInitException {
    localStrategy = new MockLocalSpawnStrategy(testRoot, localDelay);
    remoteStrategy = new MockRemoteSpawnStrategy(testRoot, remoteDelay);
    options = new DynamicExecutionOptions();
    options.dynamicLocalStrategy = "mock-local";
    options.dynamicRemoteStrategy = "mock-remote";
    options.dynamicWorkerStrategy = "mock-local";
    options.internalSpawnScheduler = true;
    options.localExecutionDelay = 0;
    dynamicSpawnStrategy =
        new DynamicSpawnStrategyUnderTest(executorService, options, this::getExecutionPolicy);
    dynamicSpawnStrategy.executorCreated(ImmutableList.of(localStrategy, remoteStrategy));
  }

  ExecutionPolicy getExecutionPolicy(Spawn spawn) {
    if (spawn.getExecutionInfo().containsKey("local")) {
      return ExecutionPolicy.LOCAL_EXECUTION_ONLY;
    } else if (spawn.getExecutionInfo().containsKey("remote")) {
      return ExecutionPolicy.REMOTE_EXECUTION_ONLY;
    } else {
      return ExecutionPolicy.ANYWHERE;
    }
  }

  @After
  public void tearDown() throws Exception {
    executorService.shutdownNow();
  }

  Spawn getSpawnForTest(boolean forceLocal, boolean forceRemote) {
    Preconditions.checkArgument(
        !(forceLocal && forceRemote), "Cannot force local and remote at the same time");
    ActionExecutionMetadata action =
        new NullAction(ImmutableList.of(inputArtifact), outputArtifact);
    return new BaseSpawn(
        ImmutableList.<String>of(),
        ImmutableMap.<String, String>of(),
        forceLocal
            ? ImmutableMap.of("local", "1")
            : forceRemote ? ImmutableMap.of("remote", "1") : ImmutableMap.<String, String>of(),
        action,
        ResourceSet.create(1, 0, 0));
  }

  @Test
  public void nonRemotableSpawnRunsLocally() throws Exception {
    Spawn spawn = getSpawnForTest(true, false);
    createSpawnStrategy(0, 0);

    dynamicSpawnStrategy.exec(spawn, actionExecutionContext);

    assertThat(localStrategy.getExecutedSpawn()).isEqualTo(spawn);
    assertThat(localStrategy.succeeded()).isTrue();
    assertThat(remoteStrategy.getExecutedSpawn()).isNull();
    assertThat(remoteStrategy.succeeded()).isFalse();

    assertThat(outErr.outAsLatin1()).contains("output files written with MockLocalSpawnStrategy");
    assertThat(outErr.outAsLatin1()).doesNotContain("MockRemoteSpawnStrategy");
  }

  @Test
  public void nonLocallyExecutableSpawnRunsRemotely() throws Exception {
    Spawn spawn = getSpawnForTest(false, true);
    createSpawnStrategy(0, 0);

    dynamicSpawnStrategy.exec(spawn, actionExecutionContext);

    assertThat(localStrategy.getExecutedSpawn()).isNull();
    assertThat(localStrategy.succeeded()).isFalse();
    assertThat(remoteStrategy.getExecutedSpawn()).isEqualTo(spawn);
    assertThat(remoteStrategy.succeeded()).isTrue();

    assertThat(outErr.outAsLatin1()).contains("output files written with MockRemoteSpawnStrategy");
    assertThat(outErr.outAsLatin1()).doesNotContain("MockLocalSpawnStrategy");
  }

  @Test
  public void actionSucceedsIfLocalExecutionSucceedsEvenIfRemoteFailsLater() throws Exception {
    Spawn spawn = getSpawnForTest(false, false);
    createSpawnStrategy(0, 2000);
    CountDownLatch countDownLatch = new CountDownLatch(2);
    localStrategy.beforeExecutionWaitFor(countDownLatch);
    remoteStrategy.beforeExecutionWaitFor(countDownLatch);
    remoteStrategy.failsDuringExecution();

    dynamicSpawnStrategy.exec(spawn, actionExecutionContext);

    assertThat(localStrategy.getExecutedSpawn()).isEqualTo(spawn);
    assertThat(localStrategy.succeeded()).isTrue();
    assertThat(remoteStrategy.getExecutedSpawn()).isEqualTo(spawn);
    assertThat(remoteStrategy.succeeded()).isFalse();

    assertThat(outErr.outAsLatin1()).contains("output files written with MockLocalSpawnStrategy");
    assertThat(outErr.outAsLatin1()).doesNotContain("MockRemoteSpawnStrategy");
  }

  @Test
  public void actionSucceedsIfRemoteExecutionSucceedsEvenIfLocalFailsLater() throws Exception {
    Spawn spawn = getSpawnForTest(false, false);
    createSpawnStrategy(2000, 0);
    CountDownLatch countDownLatch = new CountDownLatch(2);
    localStrategy.beforeExecutionWaitFor(countDownLatch);
    localStrategy.failsDuringExecution();
    remoteStrategy.beforeExecutionWaitFor(countDownLatch);

    dynamicSpawnStrategy.exec(spawn, actionExecutionContext);

    assertThat(localStrategy.getExecutedSpawn()).isEqualTo(spawn);
    assertThat(localStrategy.succeeded()).isFalse();
    assertThat(remoteStrategy.getExecutedSpawn()).isEqualTo(spawn);
    assertThat(remoteStrategy.succeeded()).isTrue();

    assertThat(outErr.outAsLatin1()).contains("output files written with MockRemoteSpawnStrategy");
    assertThat(outErr.outAsLatin1()).doesNotContain("MockLocalSpawnStrategy");
  }

  @Test
  public void actionFailsIfLocalFailsImmediatelyEvenIfRemoteSucceedsLater() throws Exception {
    Spawn spawn = getSpawnForTest(false, false);
    createSpawnStrategy(0, 2000);
    CountDownLatch countDownLatch = new CountDownLatch(2);
    localStrategy.beforeExecutionWaitFor(countDownLatch);
    localStrategy.failsDuringExecution();
    remoteStrategy.beforeExecutionWaitFor(countDownLatch);

    try {
      dynamicSpawnStrategy.exec(spawn, actionExecutionContext);
      fail("Expected dynamicSpawnStrategy to throw an ExecException");
    } catch (ExecException e) {
      assertThat(e).hasMessageThat().matches("MockLocalSpawnStrategy failed to execute the Spawn");
    }

    assertThat(localStrategy.getExecutedSpawn()).isEqualTo(spawn);
    assertThat(localStrategy.succeeded()).isFalse();
    assertThat(remoteStrategy.getExecutedSpawn()).isEqualTo(spawn);
    assertThat(remoteStrategy.succeeded()).isFalse();

    assertThat(outErr.outAsLatin1()).contains("action failed with MockLocalSpawnStrategy");
    assertThat(outErr.outAsLatin1()).doesNotContain("MockRemoteSpawnStrategy");
  }

  @Test
  public void actionFailsIfRemoteFailsImmediatelyEvenIfLocalSucceedsLater() throws Exception {
    Spawn spawn = getSpawnForTest(false, false);
    createSpawnStrategy(2000, 0);
    CountDownLatch countDownLatch = new CountDownLatch(2);
    localStrategy.beforeExecutionWaitFor(countDownLatch);
    remoteStrategy.beforeExecutionWaitFor(countDownLatch);
    remoteStrategy.failsDuringExecution();

    try {
      dynamicSpawnStrategy.exec(spawn, actionExecutionContext);
      fail("Expected dynamicSpawnStrategy to throw an ExecException");
    } catch (ExecException e) {
      assertThat(e).hasMessageThat().matches("MockRemoteSpawnStrategy failed to execute the Spawn");
    }

    assertThat(localStrategy.getExecutedSpawn()).isEqualTo(spawn);
    assertThat(localStrategy.succeeded()).isFalse();
    assertThat(remoteStrategy.getExecutedSpawn()).isEqualTo(spawn);
    assertThat(remoteStrategy.succeeded()).isFalse();

    assertThat(outErr.outAsLatin1()).contains("action failed with MockRemoteSpawnStrategy");
    assertThat(outErr.outAsLatin1()).doesNotContain("MockLocalSpawnStrategy");
  }

  @Test
  public void actionFailsIfLocalAndRemoteFail() throws Exception {
    Spawn spawn = getSpawnForTest(false, false);
    createSpawnStrategy(0, 0);
    CountDownLatch countDownLatch = new CountDownLatch(2);
    localStrategy.beforeExecutionWaitFor(countDownLatch);
    remoteStrategy.beforeExecutionWaitFor(countDownLatch);
    localStrategy.failsDuringExecution();
    remoteStrategy.failsDuringExecution();

    try {
      dynamicSpawnStrategy.exec(spawn, actionExecutionContext);
      fail("Expected dynamicSpawnStrategy to throw an ExecException");
    } catch (ExecException e) {
      assertThat(e)
          .hasMessageThat()
          .matches("Mock(Local|Remote)SpawnStrategy failed to execute the Spawn");
    }

    assertThat(localStrategy.getExecutedSpawn()).isEqualTo(spawn);
    assertThat(localStrategy.succeeded()).isFalse();
    assertThat(remoteStrategy.getExecutedSpawn()).isEqualTo(spawn);
    assertThat(remoteStrategy.succeeded()).isFalse();
  }

  @Test
  public void noDeadlockWithSingleThreadedExecutor() throws Exception {
    final Spawn spawn = getSpawnForTest(/*forceLocal=*/ false, /*forceRemote=*/ false);

    // Replace the executorService with a single threaded one.
    executorService = Executors.newSingleThreadExecutor();
    createSpawnStrategy(/*localDelay=*/ 0, /*remoteDelay=*/ 0);

    dynamicSpawnStrategy.exec(spawn, actionExecutionContext);

    assertThat(localStrategy.getExecutedSpawn()).isEqualTo(spawn);
    assertThat(localStrategy.succeeded()).isTrue();

    /**
     * The single-threaded executorService#invokeAny does not comply to the contract where
     * the callables are *always* called sequentially. In this case, both spawns will start
     * executing, but the local one will always succeed as it's the first to be called. The remote
     * one will then be cancelled, or is null if the local one completes before the remote one
     * starts.
     *
     * See the documentation of {@link BoundedExectorService#invokeAny(Collection)}, specifically:
     * "The following is less efficient (it goes on submitting tasks even if there is some task
     * already finished), but quite straight-forward.".
     */
    assertThat(remoteStrategy.getExecutedSpawn()).isAnyOf(spawn, null);
    assertThat(remoteStrategy.succeeded()).isFalse();
  }

  @Test
  public void interruptDuringExecutionDoesActuallyInterruptTheExecution() throws Exception {
    final Spawn spawn = getSpawnForTest(false, false);
    createSpawnStrategy(60000, 60000);
    CountDownLatch countDownLatch = new CountDownLatch(2);
    localStrategy.beforeExecutionWaitFor(countDownLatch);
    remoteStrategy.beforeExecutionWaitFor(countDownLatch);

    TestThread testThread =
        new TestThread() {
          @Override
          public void runTest() throws Exception {
            try {
              dynamicSpawnStrategy.exec(spawn, actionExecutionContext);
            } catch (InterruptedException e) {
              // This is expected.
            }
          }
        };
    testThread.start();
    countDownLatch.await(5, TimeUnit.SECONDS);
    testThread.interrupt();
    testThread.joinAndAssertState(5000);

    assertThat(outErr.getOutputPath().exists()).isFalse();
    assertThat(outErr.getErrorPath().exists()).isFalse();
  }

  private void strategyWaitsForBothSpawnsToFinish(boolean interruptThread, boolean executionFails)
      throws Exception {
    final Spawn spawn = getSpawnForTest(false, false);
    createSpawnStrategy(0, 0);
    CountDownLatch waitToFinish = new CountDownLatch(1);
    CountDownLatch wasInterrupted = new CountDownLatch(1);
    CountDownLatch executionCanProceed = new CountDownLatch(2);
    localStrategy.setExecute(
        () -> {
          executionCanProceed.countDown();
          try {
            Thread.sleep(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
            throw new IllegalStateException("Should have been interrupted");
          } catch (InterruptedException e) {
            // Expected.
          }
          wasInterrupted.countDown();
          try {
            Preconditions.checkState(
                waitToFinish.await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS));
          } catch (InterruptedException e) {
            throw new IllegalStateException(e);
          }
          return ImmutableList.of();
        });
    if (executionFails) {
      remoteStrategy.failsDuringExecution();
    }
    remoteStrategy.beforeExecutionWaitFor(executionCanProceed);

    TestThread testThread =
        new TestThread() {
          @Override
          public void runTest() {
            try {
              dynamicSpawnStrategy.exec(spawn, actionExecutionContext);
              Preconditions.checkState(!interruptThread && !executionFails);
            } catch (InterruptedException e) {
              Preconditions.checkState(interruptThread && !executionFails);
              Preconditions.checkState(!Thread.currentThread().isInterrupted());
            } catch (ExecException e) {
              Preconditions.checkState(executionFails);
              Preconditions.checkState(Thread.currentThread().isInterrupted() == interruptThread);
            }
          }
        };
    testThread.start();
    if (!executionFails) {
      assertThat(
              remoteStrategy
                  .getSucceededLatch()
                  .await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS))
          .isTrue();
    }
    assertThat(wasInterrupted.await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS)).isTrue();
    assertThat(testThread.isAlive()).isTrue();
    if (interruptThread) {
      testThread.interrupt();
    }
    // Wait up to 5 seconds for this thread to finish. It should not have finished.
    testThread.join(5000);
    assertThat(testThread.isAlive()).isTrue();
    waitToFinish.countDown();
    testThread.joinAndAssertState(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
  }

  @Test
  public void strategyWaitsForBothSpawnsToFinish() throws Exception {
    strategyWaitsForBothSpawnsToFinish(false, false);
  }

  @Test
  public void strategyWaitsForBothSpawnsToFinishEvenIfInterrupted() throws Exception {
    strategyWaitsForBothSpawnsToFinish(true, false);
  }

  @Test
  public void strategyWaitsForBothSpawnsToFinishOnFailure() throws Exception {
    strategyWaitsForBothSpawnsToFinish(false, true);
  }

  @Test
  public void strategyWaitsForBothSpawnsToFinishOnFailureEvenIfInterrupted() throws Exception {
    strategyWaitsForBothSpawnsToFinish(true, true);
  }

  @Test
  public void strategyPropagatesFasterLocalException() throws Exception {
    strategyPropagatesException(true);
  }

  @Test
  public void strategyPropagatesFasterRemoteException() throws Exception {
    strategyPropagatesException(false);
  }

  private void strategyPropagatesException(boolean preferLocal) throws Exception {
    final Spawn spawn = getSpawnForTest(false, false);
    createSpawnStrategy(!preferLocal ? 60000 : 0, preferLocal ? 60000 : 0);

    String message = "Mock spawn execution exception";
    Callable<List<SpawnResult>> execute = () -> {
      throw new IllegalStateException(message);
    };
    localStrategy.setExecute(execute);
    remoteStrategy.setExecute(execute);

    try {
      dynamicSpawnStrategy.exec(spawn, actionExecutionContext);
      fail("Expected dynamicSpawnStrategy to throw an ExecException");
    } catch (ExecException e) {
      assertThat(e).hasMessageThat().matches("java.lang.IllegalStateException: " + message);
    }

    Spawn executedSpawn = localStrategy.getExecutedSpawn();
    executedSpawn = executedSpawn == null ? remoteStrategy.getExecutedSpawn() : executedSpawn;
    assertThat(executedSpawn).isEqualTo(spawn);
    assertThat(localStrategy.succeeded()).isFalse();
    assertThat(remoteStrategy.succeeded()).isFalse();
  }
}
