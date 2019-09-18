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

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.BaseSpawn;
import com.google.devtools.build.lib.actions.EmptyRunfilesSupplier;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.ExecutorInitException;
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
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.util.FileSystems;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import javax.annotation.Nullable;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DynamicSpawnStrategy}. */
@RunWith(JUnit4.class)
public class DynamicSpawnStrategyTest {
  private Path testRoot;
  private ExecutorService executorServiceForCleanup;
  private FileOutErr outErr;
  private ActionExecutionContext actionExecutionContext;
  private final ActionKeyContext actionKeyContext = new ActionKeyContext();

  /** Syntactic sugar to decrease and await for a latch in a single line. */
  private static void countDownAndWait(CountDownLatch countDownLatch) throws InterruptedException {
    countDownLatch.countDown();
    countDownLatch.await();
  }

  /**
   * Minimal implementation of a strategy for testing purposes.
   *
   * <p>All the logic in here must be applicable to all tests. If any test needs to special-case
   * some aspect of this logic, then it must extend this subclass as necessary.
   *
   * <p>Note, however, that this class lacks {@link ExecutionStrategy} annotations and is the main
   * reason why this is marked as abstract. You must use any of the direct subclasses of this
   * strategy instead (optionally extending them).
   */
  private abstract static class MockSpawnStrategy implements SandboxedSpawnActionContext {
    /** Identifier of this class for error reporting purposes. */
    private final String name;

    /** Base location of the file hierarchy where the stdout/stderr of the spawn is written to. */
    private final Path testRoot;

    /** Lazily set to the spawn passed to {@link #exec} as soon as that hook is invoked. */
    @Nullable private volatile Spawn executedSpawn;

    /** Tracks whether {@link #exec} completed successfully or not. */
    private CountDownLatch succeeded = new CountDownLatch(1);

    @FunctionalInterface
    interface DoExec {
      List<SpawnResult> run(
          MockSpawnStrategy self, Spawn spawn, ActionExecutionContext actionExecutionContext)
          throws ExecException, InterruptedException;
    }

    /** Hook to implement per-test custom logic. */
    private final DoExec doExec;

    MockSpawnStrategy(String name, Path testRoot, DoExec doExec) {
      this.name = name;
      this.testRoot = testRoot;
      this.doExec = doExec;
    }

    /** Helper to record an execution failure from within {@link #doExec}. */
    void failExecution(ActionExecutionContext actionExecutionContext) throws ExecException {
      try {
        FileSystemUtils.appendIsoLatin1(
            actionExecutionContext.getFileOutErr().getOutputPath(), "action failed with " + name);
      } catch (IOException e) {
        throw new IllegalStateException(e);
      }
      throw new UserExecException(name + " failed to execute the Spawn");
    }

    @Override
    public List<SpawnResult> exec(
        Spawn spawn,
        ActionExecutionContext actionExecutionContext,
        @Nullable StopConcurrentSpawns stopConcurrentSpawns)
        throws ExecException, InterruptedException {
      executedSpawn = spawn;

      List<SpawnResult> result = doExec.run(this, spawn, actionExecutionContext);

      if (stopConcurrentSpawns != null) {
        stopConcurrentSpawns.stop();
      }

      for (ActionInput output : spawn.getOutputFiles()) {
        try {
          FileSystemUtils.writeIsoLatin1(testRoot.getRelative(output.getExecPath()), name);
        } catch (IOException e) {
          throw new IllegalStateException(e);
        }
      }

      try {
        FileSystemUtils.appendIsoLatin1(
            actionExecutionContext.getFileOutErr().getOutputPath(),
            "output files written with " + name);
      } catch (IOException e) {
        throw new IllegalStateException(e);
      }

      succeeded.countDown();

      return result;
    }

    @Override
    public List<SpawnResult> exec(Spawn spawn, ActionExecutionContext actionExecutionContext) {
      throw new IllegalStateException("Not expected to be called");
    }

    @Override
    public boolean canExec(Spawn spawn) {
      return true;
    }

    @Nullable
    public Spawn getExecutedSpawn() {
      return executedSpawn;
    }

    /** Returns true if {@link #exec} was called and completed successfully; does not block. */
    boolean succeeded() {
      return succeeded.getCount() == 0;
    }

    /** Blocks until {@link #exec} completes and returns true in that case. */
    boolean awaitSuccess() throws InterruptedException {
      return succeeded.await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS);
    }
  }

  /** Extends the mock strategy with an execution strategy annotation for remote execution. */
  @ExecutionStrategy(
      name = {"mock-remote"},
      contextType = SpawnActionContext.class)
  private static class MockRemoteSpawnStrategy extends MockSpawnStrategy {
    MockRemoteSpawnStrategy(Path testRoot) {
      this(testRoot, (self, spawn, actionExecutionContext) -> ImmutableList.of());
    }

    MockRemoteSpawnStrategy(Path testRoot, DoExec doExec) {
      super("MockRemoteSpawnStrategy", testRoot, doExec);
    }
  }

  /** Extends the mock strategy with an execution strategy annotation for local execution. */
  @ExecutionStrategy(
      name = {"mock-local"},
      contextType = SpawnActionContext.class)
  private static class MockLocalSpawnStrategy extends MockSpawnStrategy {
    MockLocalSpawnStrategy(Path testRoot) {
      this(testRoot, (self, spawn, actionExecutionContext) -> ImmutableList.of());
    }

    MockLocalSpawnStrategy(Path testRoot, DoExec doExec) {
      super("MockLocalSpawnStrategy", testRoot, doExec);
    }
  }

  /** Extends the mock strategy with an execution strategy annotation for sandboxed execution. */
  @ExecutionStrategy(
      name = {"mock-sandboxed"},
      contextType = SpawnActionContext.class)
  private static class MockSandboxedSpawnStrategy extends MockSpawnStrategy {
    MockSandboxedSpawnStrategy(Path testRoot) {
      this(testRoot, (self, spawn, actionExecutionContext) -> ImmutableList.of());
    }

    MockSandboxedSpawnStrategy(Path testRoot, DoExec doExec) {
      super("MockSandboxedSpawnStrategy", testRoot, doExec);
    }
  }

  @Before
  public void setUp() throws Exception {
    testRoot = FileSystems.getNativeFileSystem().getPath(TestUtils.tmpDir());
    testRoot.deleteTreesBelow();
    outErr = new FileOutErr(testRoot.getRelative("stdout"), testRoot.getRelative("stderr"));
    actionExecutionContext =
        ActionsTestUtil.createContext(
            /*executor=*/ null,
            /*eventHandler=*/ null,
            actionKeyContext,
            outErr,
            testRoot,
            /*metadataHandler=*/ null,
            /*actionGraph=*/ null);
  }

  /**
   * Creates a new dynamic spawn strategy with different strategies for local and remote execution.
   *
   * @param localStrategy the strategy for local execution
   * @param remoteStrategy the strategy for remote execution
   * @param executorService the executor to pass to the dynamic strategy
   * @return the constructed dynamic strategy
   * @throws ExecutorInitException if creating the strategy with the given parameters fails
   */
  private SpawnActionContext createSpawnStrategyWithExecutor(
      MockLocalSpawnStrategy localStrategy,
      MockRemoteSpawnStrategy remoteStrategy,
      ExecutorService executorService)
      throws ExecutorInitException {
    DynamicExecutionOptions options = new DynamicExecutionOptions();
    options.dynamicLocalStrategy =
        Lists.newArrayList(Maps.immutableEntry("", ImmutableList.of("mock-local")));
    options.dynamicRemoteStrategy =
        Lists.newArrayList(Maps.immutableEntry("", ImmutableList.of("mock-remote")));
    options.dynamicWorkerStrategy = "mock-local";
    options.localExecutionDelay = 0;

    checkState(executorServiceForCleanup == null);
    executorServiceForCleanup = executorService;

    DynamicExecutionModule.setDefaultStrategiesByMnemonic(options);
    SpawnActionContext dynamicSpawnStrategy =
        new DynamicSpawnStrategy(
            executorService, options, DynamicSpawnStrategyTest::getExecutionPolicy);
    dynamicSpawnStrategy.executorCreated(ImmutableList.of(localStrategy, remoteStrategy));
    return dynamicSpawnStrategy;
  }

  /**
   * Creates a new dynamic spawn strategy with different strategies for local and remote execution
   * and a default multi-threaded executor service.
   *
   * @param localStrategy the strategy for local execution
   * @param remoteStrategy the strategy for remote execution
   * @return the constructed dynamic strategy
   * @throws ExecutorInitException if creating the strategy with the given parameters fails
   */
  private SpawnActionContext createSpawnStrategy(
      MockLocalSpawnStrategy localStrategy, MockRemoteSpawnStrategy remoteStrategy)
      throws ExecutorInitException {
    return createSpawnStrategyWithExecutor(
        localStrategy, remoteStrategy, Executors.newCachedThreadPool());
  }

  /**
   * Creates a new dynamic spawn strategy with different strategies for local, remote, and sandboxed
   * execution.
   *
   * <p>TODO(jmmv): This overload should not be necessary now that we do not special-case the
   * handling of sandboxed strategies any longer. Remove once the sandbox-specific flags are gone.
   *
   * @param localStrategy the default strategy for local execution
   * @param remoteStrategy the default strategy for remote execution
   * @param sandboxedStrategy the strategy to use when the mnemonic matches {@code testMnemonic}.
   * @return the constructed dynamic strategy
   * @throws ExecutorInitException if creating the strategy with the given parameters fails
   */
  private SpawnActionContext createSpawnStrategy(
      MockLocalSpawnStrategy localStrategy,
      MockRemoteSpawnStrategy remoteStrategy,
      MockSandboxedSpawnStrategy sandboxedStrategy)
      throws ExecutorInitException {
    DynamicExecutionOptions options = new DynamicExecutionOptions();
    options.dynamicLocalStrategy =
        Lists.newArrayList(
            Maps.immutableEntry("", ImmutableList.of("mock-local")),
            Maps.immutableEntry("testMnemonic", ImmutableList.of("mock-sandboxed")));
    options.dynamicRemoteStrategy =
        Lists.newArrayList(
            Maps.immutableEntry("", ImmutableList.of("mock-remote")),
            Maps.immutableEntry("testMnemonic", ImmutableList.of("mock-sandboxed")));
    options.dynamicWorkerStrategy = "mock-local";
    options.internalSpawnScheduler = true;
    options.localExecutionDelay = 0;

    checkState(executorServiceForCleanup == null);
    executorServiceForCleanup = Executors.newCachedThreadPool();

    DynamicExecutionModule.setDefaultStrategiesByMnemonic(options);
    SpawnActionContext dynamicSpawnStrategy =
        new DynamicSpawnStrategy(
            executorServiceForCleanup, options, DynamicSpawnStrategyTest::getExecutionPolicy);
    dynamicSpawnStrategy.executorCreated(
        ImmutableList.of(localStrategy, remoteStrategy, sandboxedStrategy));
    return dynamicSpawnStrategy;
  }

  private static ExecutionPolicy getExecutionPolicy(Spawn spawn) {
    if (spawn.getExecutionInfo().containsKey("local")) {
      return ExecutionPolicy.LOCAL_EXECUTION_ONLY;
    } else if (spawn.getExecutionInfo().containsKey("remote")) {
      return ExecutionPolicy.REMOTE_EXECUTION_ONLY;
    } else {
      return ExecutionPolicy.ANYWHERE;
    }
  }

  private static class NullActionWithMnemonic extends NullAction {
    private final String mnemonic;

    private NullActionWithMnemonic(String mnemonic, List<Artifact> inputs, Artifact... outputs) {
      super(inputs, outputs);
      this.mnemonic = mnemonic;
    }

    @Override
    public String getMnemonic() {
      return mnemonic;
    }
  }

  @After
  public void tearDown() throws Exception {
    if (executorServiceForCleanup != null) {
      executorServiceForCleanup.shutdownNow();
    }
  }

  /** Constructs a new spawn with a custom mnemonic and execution info. */
  private Spawn newCustomSpawn(String mnemonic, ImmutableMap<String, String> executionInfo) {
    Artifact inputArtifact =
        ActionsTestUtil.createArtifact(
            ArtifactRoot.asSourceRoot(Root.fromPath(testRoot)), "input.txt");
    Artifact outputArtifact =
        ActionsTestUtil.createArtifact(
            ArtifactRoot.asSourceRoot(Root.fromPath(testRoot)), "output.txt");

    ActionExecutionMetadata action =
        new NullActionWithMnemonic(mnemonic, ImmutableList.of(inputArtifact), outputArtifact);
    return new BaseSpawn(
        ImmutableList.of(),
        ImmutableMap.of(),
        executionInfo,
        EmptyRunfilesSupplier.INSTANCE,
        action,
        ResourceSet.create(1, 0, 0));
  }

  /** Constructs a new spawn that can be run locally and remotely with arbitrary settings. */
  private Spawn newDynamicSpawn() {
    return newCustomSpawn("Null", ImmutableMap.of());
  }

  @Test
  public void nonRemotableSpawnRunsLocally() throws Exception {
    MockLocalSpawnStrategy localStrategy = new MockLocalSpawnStrategy(testRoot);
    MockRemoteSpawnStrategy remoteStrategy = new MockRemoteSpawnStrategy(testRoot);
    SpawnActionContext dynamicSpawnStrategy = createSpawnStrategy(localStrategy, remoteStrategy);

    Spawn spawn = newCustomSpawn("Null", ImmutableMap.of("local", "1"));
    dynamicSpawnStrategy.exec(spawn, actionExecutionContext);

    assertThat(localStrategy.getExecutedSpawn()).isEqualTo(spawn);
    assertThat(localStrategy.succeeded()).isTrue();
    assertThat(remoteStrategy.getExecutedSpawn()).isNull();
    assertThat(remoteStrategy.succeeded()).isFalse();

    assertThat(outErr.outAsLatin1()).contains("output files written with MockLocalSpawnStrategy");
    assertThat(outErr.outAsLatin1()).doesNotContain("MockRemoteSpawnStrategy");
  }

  @Test
  public void localSpawnUsesStrategyByMnemonicWithWorkerFlagDisabled() throws Exception {
    MockLocalSpawnStrategy localStrategy = new MockLocalSpawnStrategy(testRoot);
    MockRemoteSpawnStrategy remoteStrategy = new MockRemoteSpawnStrategy(testRoot);
    MockSandboxedSpawnStrategy sandboxedStrategy = new MockSandboxedSpawnStrategy(testRoot);
    SpawnActionContext dynamicSpawnStrategy =
        createSpawnStrategy(localStrategy, remoteStrategy, sandboxedStrategy);

    Spawn spawn = newCustomSpawn("testMnemonic", ImmutableMap.of("local", "1"));
    dynamicSpawnStrategy.exec(spawn, actionExecutionContext);

    assertThat(localStrategy.getExecutedSpawn()).isNull();
    assertThat(localStrategy.succeeded()).isFalse();
    assertThat(remoteStrategy.getExecutedSpawn()).isNull();
    assertThat(remoteStrategy.succeeded()).isFalse();
    assertThat(sandboxedStrategy.getExecutedSpawn()).isEqualTo(spawn);
    assertThat(sandboxedStrategy.succeeded()).isTrue();

    assertThat(outErr.outAsLatin1())
        .contains("output files written with MockSandboxedSpawnStrategy");
    assertThat(outErr.outAsLatin1()).doesNotContain("MockLocalSpawnStrategy");
    assertThat(outErr.outAsLatin1()).doesNotContain("MockRemoteSpawnStrategy");
  }

  @Test
  public void nonLocallyExecutableSpawnRunsRemotely() throws Exception {
    MockLocalSpawnStrategy localStrategy = new MockLocalSpawnStrategy(testRoot);
    MockRemoteSpawnStrategy remoteStrategy = new MockRemoteSpawnStrategy(testRoot);
    SpawnActionContext dynamicSpawnStrategy = createSpawnStrategy(localStrategy, remoteStrategy);

    Spawn spawn = newCustomSpawn("Null", ImmutableMap.of("remote", "1"));
    dynamicSpawnStrategy.exec(spawn, actionExecutionContext);

    assertThat(localStrategy.getExecutedSpawn()).isNull();
    assertThat(localStrategy.succeeded()).isFalse();
    assertThat(remoteStrategy.getExecutedSpawn()).isEqualTo(spawn);
    assertThat(remoteStrategy.succeeded()).isTrue();

    assertThat(outErr.outAsLatin1()).contains("output files written with MockRemoteSpawnStrategy");
    assertThat(outErr.outAsLatin1()).doesNotContain("MockLocalSpawnStrategy");
  }

  @Test
  public void remoteSpawnUsesStrategyByMnemonic() throws Exception {
    MockLocalSpawnStrategy localStrategy = new MockLocalSpawnStrategy(testRoot);
    MockRemoteSpawnStrategy remoteStrategy = new MockRemoteSpawnStrategy(testRoot);
    MockSandboxedSpawnStrategy sandboxedStrategy = new MockSandboxedSpawnStrategy(testRoot);
    SpawnActionContext dynamicSpawnStrategy =
        createSpawnStrategy(localStrategy, remoteStrategy, sandboxedStrategy);

    Spawn spawn = newCustomSpawn("testMnemonic", ImmutableMap.of("remote", "1"));
    dynamicSpawnStrategy.exec(spawn, actionExecutionContext);

    assertThat(localStrategy.getExecutedSpawn()).isNull();
    assertThat(localStrategy.succeeded()).isFalse();
    assertThat(remoteStrategy.getExecutedSpawn()).isNull();
    assertThat(remoteStrategy.succeeded()).isFalse();
    assertThat(sandboxedStrategy.getExecutedSpawn()).isEqualTo(spawn);
    assertThat(sandboxedStrategy.succeeded()).isTrue();

    assertThat(outErr.outAsLatin1())
        .contains("output files written with MockSandboxedSpawnStrategy");
    assertThat(outErr.outAsLatin1()).doesNotContain("MockLocalSpawnStrategy");
    assertThat(outErr.outAsLatin1()).doesNotContain("MockRemoteSpawnStrategy");
  }

  @Test
  public void actionSucceedsIfLocalExecutionSucceedsEvenIfRemoteFailsLater() throws Exception {
    CountDownLatch countDownLatch = new CountDownLatch(2);

    MockLocalSpawnStrategy localStrategy =
        new MockLocalSpawnStrategy(
            testRoot,
            (self, spawn, actionExecutionContext) -> {
              countDownAndWait(countDownLatch);
              return ImmutableList.of();
            });

    MockRemoteSpawnStrategy remoteStrategy =
        new MockRemoteSpawnStrategy(
            testRoot,
            (self, spawn, actionExecutionContext) -> {
              countDownAndWait(countDownLatch);
              Thread.sleep(2000);
              self.failExecution(actionExecutionContext);
              return ImmutableList.of();
            });

    SpawnActionContext dynamicSpawnStrategy = createSpawnStrategy(localStrategy, remoteStrategy);

    Spawn spawn = newDynamicSpawn();
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
    CountDownLatch countDownLatch = new CountDownLatch(2);

    MockLocalSpawnStrategy localStrategy =
        new MockLocalSpawnStrategy(
            testRoot,
            (self, spawn, actionExecutionContext) -> {
              countDownAndWait(countDownLatch);
              Thread.sleep(2000);
              self.failExecution(actionExecutionContext);
              return ImmutableList.of();
            });

    MockRemoteSpawnStrategy remoteStrategy =
        new MockRemoteSpawnStrategy(
            testRoot,
            (self, spawn, actionExecutionContext) -> {
              countDownAndWait(countDownLatch);
              return ImmutableList.of();
            });

    SpawnActionContext dynamicSpawnStrategy = createSpawnStrategy(localStrategy, remoteStrategy);

    Spawn spawn = newDynamicSpawn();
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
    CountDownLatch countDownLatch = new CountDownLatch(2);

    MockLocalSpawnStrategy localStrategy =
        new MockLocalSpawnStrategy(
            testRoot,
            (self, spawn, actionExecutionContext) -> {
              countDownAndWait(countDownLatch);
              self.failExecution(actionExecutionContext);
              return ImmutableList.of();
            });

    MockRemoteSpawnStrategy remoteStrategy =
        new MockRemoteSpawnStrategy(
            testRoot,
            (self, spawn, actionExecutionContext) -> {
              countDownAndWait(countDownLatch);
              Thread.sleep(2000);
              return ImmutableList.of();
            });

    SpawnActionContext dynamicSpawnStrategy = createSpawnStrategy(localStrategy, remoteStrategy);

    Spawn spawn = newDynamicSpawn();
    ExecException e =
        assertThrows(
            ExecException.class, () -> dynamicSpawnStrategy.exec(spawn, actionExecutionContext));
    assertThat(e).hasMessageThat().matches("MockLocalSpawnStrategy failed to execute the Spawn");

    assertThat(localStrategy.getExecutedSpawn()).isEqualTo(spawn);
    assertThat(localStrategy.succeeded()).isFalse();
    assertThat(remoteStrategy.getExecutedSpawn()).isEqualTo(spawn);
    assertThat(remoteStrategy.succeeded()).isFalse();

    assertThat(outErr.outAsLatin1()).contains("action failed with MockLocalSpawnStrategy");
    assertThat(outErr.outAsLatin1()).doesNotContain("MockRemoteSpawnStrategy");
  }

  @Test
  public void actionFailsIfRemoteFailsImmediatelyEvenIfLocalSucceedsLater() throws Exception {
    CountDownLatch countDownLatch = new CountDownLatch(2);

    MockLocalSpawnStrategy localStrategy =
        new MockLocalSpawnStrategy(
            testRoot,
            (self, spawn, actionExecutionContext) -> {
              countDownAndWait(countDownLatch);
              Thread.sleep(2000);
              return ImmutableList.of();
            });

    MockRemoteSpawnStrategy remoteStrategy =
        new MockRemoteSpawnStrategy(
            testRoot,
            (self, spawn, actionExecutionContext) -> {
              countDownAndWait(countDownLatch);
              self.failExecution(actionExecutionContext);
              return ImmutableList.of();
            });

    SpawnActionContext dynamicSpawnStrategy = createSpawnStrategy(localStrategy, remoteStrategy);

    Spawn spawn = newDynamicSpawn();
    ExecException e =
        assertThrows(
            ExecException.class, () -> dynamicSpawnStrategy.exec(spawn, actionExecutionContext));
    assertThat(e).hasMessageThat().matches("MockRemoteSpawnStrategy failed to execute the Spawn");

    assertThat(localStrategy.getExecutedSpawn()).isEqualTo(spawn);
    assertThat(localStrategy.succeeded()).isFalse();
    assertThat(remoteStrategy.getExecutedSpawn()).isEqualTo(spawn);
    assertThat(remoteStrategy.succeeded()).isFalse();

    assertThat(outErr.outAsLatin1()).contains("action failed with MockRemoteSpawnStrategy");
    assertThat(outErr.outAsLatin1()).doesNotContain("MockLocalSpawnStrategy");
  }

  @Test
  public void actionFailsIfLocalAndRemoteFail() throws Exception {
    CountDownLatch countDownLatch = new CountDownLatch(2);

    MockLocalSpawnStrategy localStrategy =
        new MockLocalSpawnStrategy(
            testRoot,
            (self, spawn, actionExecutionContext) -> {
              countDownAndWait(countDownLatch);
              self.failExecution(actionExecutionContext);
              return ImmutableList.of();
            });

    MockRemoteSpawnStrategy remoteStrategy =
        new MockRemoteSpawnStrategy(
            testRoot,
            (self, spawn, actionExecutionContext) -> {
              countDownAndWait(countDownLatch);
              self.failExecution(actionExecutionContext);
              return ImmutableList.of();
            });

    SpawnActionContext dynamicSpawnStrategy = createSpawnStrategy(localStrategy, remoteStrategy);

    Spawn spawn = newDynamicSpawn();
    ExecException e =
        assertThrows(
            ExecException.class, () -> dynamicSpawnStrategy.exec(spawn, actionExecutionContext));
    assertThat(e)
        .hasMessageThat()
        .matches("Mock(Local|Remote)SpawnStrategy failed to execute the Spawn");

    assertThat(localStrategy.getExecutedSpawn()).isEqualTo(spawn);
    assertThat(localStrategy.succeeded()).isFalse();
    assertThat(remoteStrategy.getExecutedSpawn()).isEqualTo(spawn);
    assertThat(remoteStrategy.succeeded()).isFalse();
  }

  @Test
  public void noDeadlockWithSingleThreadedExecutor() throws Exception {
    MockLocalSpawnStrategy localStrategy = new MockLocalSpawnStrategy(testRoot);
    MockRemoteSpawnStrategy remoteStrategy = new MockRemoteSpawnStrategy(testRoot);
    SpawnActionContext dynamicSpawnStrategy =
        createSpawnStrategyWithExecutor(
            localStrategy, remoteStrategy, Executors.newSingleThreadExecutor());

    Spawn spawn = newDynamicSpawn();
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
    CountDownLatch countDownLatch = new CountDownLatch(2);

    MockLocalSpawnStrategy localStrategy =
        new MockLocalSpawnStrategy(
            testRoot,
            (self, spawn, actionExecutionContext) -> {
              countDownAndWait(countDownLatch);
              Thread.sleep(60000);
              return ImmutableList.of();
            });

    MockRemoteSpawnStrategy remoteStrategy =
        new MockRemoteSpawnStrategy(
            testRoot,
            (self, spawn, actionExecutionContext) -> {
              countDownAndWait(countDownLatch);
              Thread.sleep(60000);
              return ImmutableList.of();
            });

    SpawnActionContext dynamicSpawnStrategy = createSpawnStrategy(localStrategy, remoteStrategy);

    TestThread testThread =
        new TestThread(
            () -> {
              try {
                Spawn spawn = newDynamicSpawn();
                dynamicSpawnStrategy.exec(spawn, actionExecutionContext);
              } catch (InterruptedException e) {
                // This is expected.
              }
            });
    testThread.start();
    countDownLatch.await(5, TimeUnit.SECONDS);
    testThread.interrupt();
    testThread.joinAndAssertState(5000);

    assertThat(outErr.getOutputPath().exists()).isFalse();
    assertThat(outErr.getErrorPath().exists()).isFalse();
  }

  private void strategyWaitsForBothSpawnsToFinish(boolean interruptThread, boolean executionFails)
      throws Exception {
    CountDownLatch waitToFinish = new CountDownLatch(1);
    CountDownLatch wasInterrupted = new CountDownLatch(1);
    CountDownLatch executionCanProceed = new CountDownLatch(2);

    MockLocalSpawnStrategy localStrategy =
        new MockLocalSpawnStrategy(
            testRoot,
            (self, spawn, actionExecutionContext) -> {
              executionCanProceed.countDown();
              try {
                Thread.sleep(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
                throw new IllegalStateException("Should have been interrupted");
              } catch (InterruptedException e) {
                // Expected.
              }
              wasInterrupted.countDown();
              try {
                checkState(waitToFinish.await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS));
              } catch (InterruptedException e) {
                throw new IllegalStateException(e);
              }
              return ImmutableList.of();
            });

    MockRemoteSpawnStrategy remoteStrategy =
        new MockRemoteSpawnStrategy(
            testRoot,
            (self, spawn, actionExecutionContext) -> {
              if (executionFails) {
                self.failExecution(actionExecutionContext);
              }
              countDownAndWait(executionCanProceed);
              return ImmutableList.of();
            });

    SpawnActionContext dynamicSpawnStrategy = createSpawnStrategy(localStrategy, remoteStrategy);

    TestThread testThread =
        new TestThread(
            () -> {
              try {
                Spawn spawn = newDynamicSpawn();
                dynamicSpawnStrategy.exec(spawn, actionExecutionContext);
                checkState(!interruptThread && !executionFails);
              } catch (InterruptedException e) {
                checkState(interruptThread && !executionFails);
                checkState(!Thread.currentThread().isInterrupted());
              } catch (ExecException e) {
                checkState(executionFails);
                checkState(Thread.currentThread().isInterrupted() == interruptThread);
              }
            });
    testThread.start();
    if (!executionFails) {
      assertThat(remoteStrategy.awaitSuccess()).isTrue();
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
    String message = "Mock spawn execution exception";

    MockLocalSpawnStrategy localStrategy =
        new MockLocalSpawnStrategy(
            testRoot,
            (self, spawn, actionExecutionContext) -> {
              if (!preferLocal) {
                Thread.sleep(60000);
              }
              throw new IllegalStateException(message);
            });

    MockRemoteSpawnStrategy remoteStrategy =
        new MockRemoteSpawnStrategy(
            testRoot,
            (self, spawn, actionExecutionContext) -> {
              if (preferLocal) {
                Thread.sleep(60000);
              }
              throw new IllegalStateException(message);
            });

    SpawnActionContext dynamicSpawnStrategy = createSpawnStrategy(localStrategy, remoteStrategy);

    Spawn spawn = newDynamicSpawn();
    ExecException e =
        assertThrows(
            ExecException.class, () -> dynamicSpawnStrategy.exec(spawn, actionExecutionContext));
    assertThat(e).hasMessageThat().matches("java.lang.IllegalStateException: " + message);

    Spawn executedSpawn = localStrategy.getExecutedSpawn();
    executedSpawn = executedSpawn == null ? remoteStrategy.getExecutedSpawn() : executedSpawn;
    assertThat(executedSpawn).isEqualTo(spawn);
    assertThat(localStrategy.succeeded()).isFalse();
    assertThat(remoteStrategy.succeeded()).isFalse();
  }
}
