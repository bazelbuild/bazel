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

package com.google.devtools.build.lib.exec.local;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth8.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;
import static org.junit.Assume.assumeTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.matches;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.ByteStreams;
import com.google.common.io.Files;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.CommandLines.ParamFileActionInput;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.LocalHostCapacity;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.cache.MetadataInjector;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.exec.RunfilesTreeUpdater;
import com.google.devtools.build.lib.exec.SpawnRunner.ProgressStatus;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.runtime.ProcessWrapper;
import com.google.devtools.build.lib.shell.JavaSubprocessFactory;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import com.google.devtools.build.lib.shell.SubprocessBuilder.StreamAction;
import com.google.devtools.build.lib.shell.SubprocessFactory;
import com.google.devtools.build.lib.testutil.BlazeTestUtils;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.unix.UnixFileSystem;
import com.google.devtools.build.lib.util.NetUtil;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Converters.RegexPatternConverter;
import com.google.devtools.common.options.Options;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadLocalRandom;
import java.util.logging.Filter;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentCaptor;
import org.mockito.Mockito;

/** Unit tests for {@link LocalSpawnRunner}. */
@RunWith(JUnit4.class)
public class LocalSpawnRunnerTest {

  private static class TestedLocalSpawnRunner extends LocalSpawnRunner {
    public TestedLocalSpawnRunner(
        Path execRoot,
        LocalExecutionOptions localExecutionOptions,
        ResourceManager resourceManager,
        ProcessWrapper processWrapper,
        LocalEnvProvider localEnvProvider) {
      super(
          execRoot,
          localExecutionOptions,
          resourceManager,
          localEnvProvider,
          /*binTools=*/ null,
          processWrapper,
          Mockito.mock(RunfilesTreeUpdater.class));
    }

    // Rigged to act on supplied filesystem (e.g. InMemoryFileSystem) for testing purposes
    // TODO(b/70572634): Update FileSystem abstraction to support createTempDirectory() from
    // the java.nio.file.Files package.
    @Override
    protected Path createActionTemp(Path execRoot) throws IOException {
      Path tempDirPath;
      do {
        String idStr =
            Long.toHexString(Thread.currentThread().getId())
                + "_"
                + Long.toHexString(ThreadLocalRandom.current().nextLong());
        tempDirPath = execRoot.getRelative("tmp" + idStr);
      } while (tempDirPath.exists());
      if (!tempDirPath.createDirectory()) {
        throw new IOException(String.format("Could not create temp directory '%s'", tempDirPath));
      }
      return tempDirPath;
    }
  }

  private static class FinishedSubprocess implements Subprocess {
    private final int exitCode;

    public FinishedSubprocess(int exitCode) {
      this.exitCode = exitCode;
    }

    @Override
    public boolean destroy() {
      return false;
    }

    @Override
    public int exitValue() {
      return exitCode;
    }

    @Override
    public boolean finished() {
      return true;
    }

    @Override
    public boolean isAlive() {
      return false;
    }

    @Override
    public boolean timedout() {
      return false;
    }

    @Override
    public void waitFor() throws InterruptedException {
      // Do nothing.
    }

    @Override
    public OutputStream getOutputStream() {
      return ByteStreams.nullOutputStream();
    }

    @Override
    public InputStream getInputStream() {
      return new ByteArrayInputStream(new byte[0]);
    }

    @Override
    public InputStream getErrorStream() {
      return new ByteArrayInputStream(new byte[0]);
    }

    @Override
    public void close() {
      // Do nothing.
    }
  }

  private static final Spawn SIMPLE_SPAWN =
      new SpawnBuilder("/bin/echo", "Hi!").withEnvironment("VARIABLE", "value").build();

  private static final class SubprocessInterceptor implements SubprocessFactory {
    @Override
    public Subprocess create(SubprocessBuilder params) throws IOException {
      throw new UnsupportedOperationException();
    }
  }

  private final class SpawnExecutionContextForTesting implements SpawnExecutionContext {
    private final List<ProgressStatus> reportedStatus = new ArrayList<>();
    private final TreeMap<PathFragment, ActionInput> inputMapping = new TreeMap<>();

    private long timeoutMillis;
    private boolean prefetchCalled;
    private boolean lockOutputFilesCalled;
    private FileOutErr fileOutErr;

    public SpawnExecutionContextForTesting(FileOutErr fileOutErr) {
      this.fileOutErr = fileOutErr;
    }

    @Override
    public int getId() {
      return 0;
    }

    @Override
    public void prefetchInputs() throws IOException {
      prefetchCalled = true;
    }

    @Override
    public void lockOutputFiles() throws InterruptedException {
      lockOutputFilesCalled = true;
    }

    @Override
    public boolean speculating() {
      return false;
    }

    @Override
    public MetadataProvider getMetadataProvider() {
      return mockFileCache;
    }

    @Override
    public ArtifactExpander getArtifactExpander() {
      throw new UnsupportedOperationException();
    }

    @Override
    public Duration getTimeout() {
      return Duration.ofMillis(timeoutMillis);
    }

    @Override
    public FileOutErr getFileOutErr() {
      return fileOutErr;
    }

    @Override
    public SortedMap<PathFragment, ActionInput> getInputMapping(PathFragment baseDirectory) {
      return inputMapping;
    }

    @Override
    public void report(ProgressStatus state, String name) {
      reportedStatus.add(state);
    }

    @Override
    public MetadataInjector getMetadataInjector() {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean isRewindingEnabled() {
      return false;
    }

    @Override
    public void checkForLostInputs() {}

    @Override
    public <T extends ActionContext> T getContext(Class<T> identifyingType) {
      throw new UnsupportedOperationException();
    }
  }

  private final MetadataProvider mockFileCache = mock(MetadataProvider.class);
  private final ResourceManager resourceManager = ResourceManager.instanceForTestingOnly();

  private Logger logger;

  private static ImmutableMap<String, String> keepLocalEnvUnchanged(
      Map<String, String> env, BinTools binTools, String fallbackTmpDir) {
    return ImmutableMap.copyOf(env);
  }

  @Before
  public final void suppressLogging() {
    logger = Logger.getLogger(TestedLocalSpawnRunner.class.getName());
    logger.setFilter(new Filter() {
      @Override
      public boolean isLoggable(LogRecord record) {
        return false;
      }
    });
  }

  private FileSystem setupEnvironmentForFakeExecution() {
    // Prevent any subprocess execution at all.
    SubprocessBuilder.setDefaultSubprocessFactory(new SubprocessInterceptor());
    resourceManager.setAvailableResources(
        ResourceSet.create(/*memoryMb=*/1, /*cpuUsage=*/1, /*localTestCount=*/1));
    return new InMemoryFileSystem(DigestHashFunction.SHA256);
  }

  private static ProcessWrapper makeProcessWrapper(FileSystem fs, LocalExecutionOptions options) {
    return new ProcessWrapper(
        fs.getPath("/process-wrapper"),
        options.getLocalSigkillGraceSeconds(),
        /*gracefulSigterm=*/ false);
  }

  /**
   * Enables real execution by default.
   *
   * <p>Tests should call setupEnvironmentForFakeExecution() if they do not want real execution.
   */
  @Before
  public final void setupEnvironmentForRealExecution() {
    SubprocessBuilder.setDefaultSubprocessFactory(JavaSubprocessFactory.INSTANCE);
    resourceManager.setAvailableResources(LocalHostCapacity.getLocalHostCapacity());
  }

  @Test
  public void vanillaZeroExit() throws Exception {
    // TODO(#3536): Make this test work on Windows.
    // The Command API implicitly absolutizes the path, and we get weird paths on Windows:
    // T:\execroot\execroot\_bin\process-wrapper
    assumeTrue(OS.getCurrent() != OS.WINDOWS);

    FileSystem fs = setupEnvironmentForFakeExecution();

    SubprocessFactory factory = mock(SubprocessFactory.class);
    ArgumentCaptor<SubprocessBuilder> captor = ArgumentCaptor.forClass(SubprocessBuilder.class);
    when(factory.create(captor.capture())).thenReturn(new FinishedSubprocess(0));
    SubprocessBuilder.setDefaultSubprocessFactory(factory);

    LocalExecutionOptions options = Options.getDefaults(LocalExecutionOptions.class);
    options.localSigkillGraceSeconds = 456;
    LocalSpawnRunner runner =
        new TestedLocalSpawnRunner(
            fs.getPath("/execroot"),
            options,
            resourceManager,
            makeProcessWrapper(fs, options),
            LocalSpawnRunnerTest::keepLocalEnvUnchanged);

    FileOutErr fileOutErr = new FileOutErr(fs.getPath("/out/stdout"), fs.getPath("/out/stderr"));
    SpawnExecutionContextForTesting policy = new SpawnExecutionContextForTesting(fileOutErr);
    policy.timeoutMillis = 123 * 1000L;
    assertThat(fs.getPath("/execroot").createDirectory()).isTrue();
    SpawnResult result = runner.execAsync(SIMPLE_SPAWN, policy).get();
    verify(factory).create(any(SubprocessBuilder.class));
    assertThat(result.status()).isEqualTo(SpawnResult.Status.SUCCESS);
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.getExecutorHostName()).isEqualTo(NetUtil.getCachedShortHostName());

    assertThat(captor.getValue().getArgv())
        .containsExactlyElementsIn(
            ImmutableList.of(
                "/process-wrapper", "--timeout=123", "--kill_delay=456", "/bin/echo", "Hi!"));
    assertThat(captor.getValue().getEnv()).containsExactly("VARIABLE", "value");
    assertThat(captor.getValue().getTimeoutMillis()).isEqualTo(0);
    assertThat(captor.getValue().getStdout()).isEqualTo(StreamAction.REDIRECT);
    assertThat(captor.getValue().getStdoutFile()).isEqualTo(new File("/out/stdout"));
    assertThat(captor.getValue().getStderr()).isEqualTo(StreamAction.REDIRECT);
    assertThat(captor.getValue().getStderrFile()).isEqualTo(new File("/out/stderr"));

    assertThat(policy.lockOutputFilesCalled).isTrue();
    assertThat(policy.reportedStatus)
        .containsExactly(ProgressStatus.SCHEDULING, ProgressStatus.EXECUTING).inOrder();
  }

  @Test
  public void testParamFiles() throws Exception {
    // TODO(#3536): Make this test work on Windows.
    // The Command API implicitly absolutizes the path, and we get weird paths on Windows:
    // T:\execroot\execroot\_bin\process-wrapper
    assumeTrue(OS.getCurrent() != OS.WINDOWS);

    FileSystem fs = setupEnvironmentForFakeExecution();

    SubprocessFactory factory = mock(SubprocessFactory.class);
    when(factory.create(any())).thenReturn(new FinishedSubprocess(0));
    SubprocessBuilder.setDefaultSubprocessFactory(factory);

    LocalExecutionOptions options = Options.getDefaults(LocalExecutionOptions.class);
    options.localSigkillGraceSeconds = 456;
    Path execRoot = fs.getPath("/execroot");
    LocalSpawnRunner runner =
        new TestedLocalSpawnRunner(
            execRoot,
            options,
            resourceManager,
            makeProcessWrapper(fs, options),
            LocalSpawnRunnerTest::keepLocalEnvUnchanged);
    ParamFileActionInput paramFileActionInput =
        new ParamFileActionInput(
            PathFragment.create("some/dir/params"),
            ImmutableList.of("--foo", "--bar"),
            ParameterFileType.UNQUOTED,
            UTF_8);
    Spawn spawn =
        new SpawnBuilder("/bin/echo", "Hi!")
            .withInput(paramFileActionInput)
            .withEnvironment("VARIABLE", "value")
            .build();
    FileOutErr fileOutErr = new FileOutErr(fs.getPath("/out/stdout"), fs.getPath("/out/stderr"));
    SpawnExecutionContextForTesting policy = new SpawnExecutionContextForTesting(fileOutErr);
    policy.timeoutMillis = 123 * 1000L;
    assertThat(fs.getPath("/execroot").createDirectory()).isTrue();
    SpawnResult result = runner.execAsync(spawn, policy).get();
    assertThat(result.status()).isEqualTo(SpawnResult.Status.SUCCESS);
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.getExecutorHostName()).isEqualTo(NetUtil.getCachedShortHostName());
    Path paramFile = execRoot.getRelative("some/dir/params");
    assertThat(paramFile.exists()).isTrue();
    try (InputStream inputStream = paramFile.getInputStream()) {
      assertThat(new String(ByteStreams.toByteArray(inputStream), UTF_8).split("\n"))
          .asList()
          .containsExactly("--foo", "--bar");
    }
  }

  @Test
  public void noProcessWrapper() throws Exception {
    // TODO(#3536): Make this test work on Windows.
    // The Command API implicitly absolutizes the path, and we get weird paths on Windows:
    // T:\execroot\bin\echo
    assumeTrue(OS.getCurrent() != OS.WINDOWS);

    FileSystem fs = setupEnvironmentForFakeExecution();

    SubprocessFactory factory = mock(SubprocessFactory.class);
    ArgumentCaptor<SubprocessBuilder> captor = ArgumentCaptor.forClass(SubprocessBuilder.class);
    when(factory.create(captor.capture())).thenReturn(new FinishedSubprocess(0));
    SubprocessBuilder.setDefaultSubprocessFactory(factory);

    LocalExecutionOptions options = Options.getDefaults(LocalExecutionOptions.class);
    options.localSigkillGraceSeconds = 456;
    LocalSpawnRunner runner =
        new TestedLocalSpawnRunner(
            fs.getPath("/execroot"),
            options,
            resourceManager,
            /*processWrapper=*/ null,
            LocalSpawnRunnerTest::keepLocalEnvUnchanged);

    FileOutErr fileOutErr = new FileOutErr(fs.getPath("/out/stdout"), fs.getPath("/out/stderr"));
    SpawnExecutionContextForTesting policy = new SpawnExecutionContextForTesting(fileOutErr);
    policy.timeoutMillis = 123 * 1000L;
    assertThat(fs.getPath("/execroot").createDirectory()).isTrue();
    SpawnResult result = runner.execAsync(SIMPLE_SPAWN, policy).get();
    verify(factory).create(any());
    assertThat(result.status()).isEqualTo(SpawnResult.Status.SUCCESS);
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.getExecutorHostName()).isEqualTo(NetUtil.getCachedShortHostName());

    assertThat(captor.getValue().getArgv())
        .containsExactlyElementsIn(ImmutableList.of("/bin/echo", "Hi!"));
    assertThat(captor.getValue().getEnv()).containsExactly("VARIABLE", "value");
    // Without the process wrapper, we use the Command API to enforce the timeout.
    assertThat(captor.getValue().getTimeoutMillis()).isEqualTo(policy.timeoutMillis);

    assertThat(policy.lockOutputFilesCalled).isTrue();
  }

  @Test
  public void nonZeroExit() throws Exception {
    // TODO(#3536): Make this test work on Windows.
    // The Command API implicitly absolutizes the path, and we get weird paths on Windows:
    // T:\execroot\execroot\_bin\process-wrapper
    assumeTrue(OS.getCurrent() != OS.WINDOWS);

    FileSystem fs = setupEnvironmentForFakeExecution();

    SubprocessFactory factory = mock(SubprocessFactory.class);
    ArgumentCaptor<SubprocessBuilder> captor = ArgumentCaptor.forClass(SubprocessBuilder.class);
    when(factory.create(captor.capture())).thenReturn(new FinishedSubprocess(3));
    SubprocessBuilder.setDefaultSubprocessFactory(factory);

    LocalExecutionOptions options = Options.getDefaults(LocalExecutionOptions.class);
    LocalSpawnRunner runner =
        new TestedLocalSpawnRunner(
            fs.getPath("/execroot"),
            options,
            resourceManager,
            makeProcessWrapper(fs, options),
            LocalSpawnRunnerTest::keepLocalEnvUnchanged);

    assertThat(fs.getPath("/execroot").createDirectory()).isTrue();
    FileOutErr fileOutErr = new FileOutErr(fs.getPath("/out/stdout"), fs.getPath("/out/stderr"));
    SpawnExecutionContextForTesting policy = new SpawnExecutionContextForTesting(fileOutErr);
    SpawnResult result = runner.execAsync(SIMPLE_SPAWN, policy).get();
    verify(factory).create(any(SubprocessBuilder.class));
    assertThat(result.status()).isEqualTo(SpawnResult.Status.NON_ZERO_EXIT);
    assertThat(result.exitCode()).isEqualTo(3);
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.getExecutorHostName()).isEqualTo(NetUtil.getCachedShortHostName());

    assertThat(captor.getValue().getArgv())
        .containsExactlyElementsIn(
            ImmutableList.of(
                "/process-wrapper", "--timeout=0", "--kill_delay=15", "/bin/echo", "Hi!"));
    assertThat(captor.getValue().getEnv()).containsExactly("VARIABLE", "value");
    assertThat(captor.getValue().getStdout()).isEqualTo(StreamAction.REDIRECT);
    assertThat(captor.getValue().getStdoutFile()).isEqualTo(new File("/out/stdout"));
    assertThat(captor.getValue().getStderr()).isEqualTo(StreamAction.REDIRECT);
    assertThat(captor.getValue().getStderrFile()).isEqualTo(new File("/out/stderr"));

    assertThat(policy.lockOutputFilesCalled).isTrue();
  }

  @Test
  public void processStartupThrows() throws Exception {
    FileSystem fs = setupEnvironmentForFakeExecution();

    SubprocessFactory factory = mock(SubprocessFactory.class);
    ArgumentCaptor<SubprocessBuilder> captor = ArgumentCaptor.forClass(SubprocessBuilder.class);
    when(factory.create(captor.capture())).thenThrow(new IOException("I'm sorry, Dave"));
    SubprocessBuilder.setDefaultSubprocessFactory(factory);

    LocalExecutionOptions options = Options.getDefaults(LocalExecutionOptions.class);
    LocalSpawnRunner runner =
        new TestedLocalSpawnRunner(
            fs.getPath("/execroot"),
            options,
            resourceManager,
            makeProcessWrapper(fs, options),
            LocalSpawnRunnerTest::keepLocalEnvUnchanged);

    assertThat(fs.getPath("/out").createDirectory()).isTrue();
    assertThat(fs.getPath("/execroot").createDirectory()).isTrue();
    FileOutErr fileOutErr = new FileOutErr(fs.getPath("/out/stdout"), fs.getPath("/out/stderr"));
    SpawnExecutionContextForTesting policy = new SpawnExecutionContextForTesting(fileOutErr);
    SpawnResult result = runner.execAsync(SIMPLE_SPAWN, policy).get();
    verify(factory).create(any(SubprocessBuilder.class));
    assertThat(result.status()).isEqualTo(SpawnResult.Status.EXECUTION_FAILED);
    assertThat(result.exitCode()).isEqualTo(-1);
    assertThat(result.setupSuccess()).isFalse();
    assertThat(result.getWallTime()).isEmpty();
    assertThat(result.getUserTime()).isEmpty();
    assertThat(result.getSystemTime()).isEmpty();
    assertThat(result.getExecutorHostName()).isEqualTo(NetUtil.getCachedShortHostName());

    assertThat(FileSystemUtils.readContent(fs.getPath("/out/stderr"), UTF_8))
        .isEqualTo("Action failed to execute: java.io.IOException: I'm sorry, Dave\n");

    assertThat(policy.lockOutputFilesCalled).isTrue();
  }

  @Test
  public void disallowLocalExecution() throws Exception {
    FileSystem fs = setupEnvironmentForFakeExecution();

    LocalExecutionOptions options = Options.getDefaults(LocalExecutionOptions.class);
    options.allowedLocalAction = new RegexPatternConverter().convert("none");
    LocalSpawnRunner runner =
        new TestedLocalSpawnRunner(
            fs.getPath("/execroot"),
            options,
            resourceManager,
            makeProcessWrapper(fs, options),
            LocalSpawnRunnerTest::keepLocalEnvUnchanged);

    assertThat(fs.getPath("/execroot").createDirectory()).isTrue();
    FileOutErr fileOutErr = new FileOutErr();
    SpawnExecutionContextForTesting policy = new SpawnExecutionContextForTesting(fileOutErr);
    SpawnResult reply = runner.execAsync(SIMPLE_SPAWN, policy).get();
    assertThat(reply.status()).isEqualTo(SpawnResult.Status.EXECUTION_DENIED);
    assertThat(reply.exitCode()).isEqualTo(-1);
    assertThat(reply.setupSuccess()).isFalse();
    assertThat(reply.getWallTime()).isEmpty();
    assertThat(reply.getUserTime()).isEmpty();
    assertThat(reply.getSystemTime()).isEmpty();
    assertThat(reply.getExecutorHostName()).isEqualTo(NetUtil.getCachedShortHostName());

    // TODO(ulfjack): Maybe we should only lock after checking?
    assertThat(policy.lockOutputFilesCalled).isTrue();
  }

  @Test
  public void interruptedException() throws Exception {
    FileSystem fs = setupEnvironmentForFakeExecution();

    SubprocessFactory factory = mock(SubprocessFactory.class);
    ArgumentCaptor<SubprocessBuilder> captor = ArgumentCaptor.forClass(SubprocessBuilder.class);
    when(factory.create(captor.capture())).thenReturn(new FinishedSubprocess(3) {
      private boolean destroyed;

      @Override
      public boolean destroy() {
        destroyed = true;
        return true;
      }

      @Override
      public void waitFor() throws InterruptedException {
        if (!destroyed) {
          throw new InterruptedException();
        }
      }
    });
    SubprocessBuilder.setDefaultSubprocessFactory(factory);

    LocalExecutionOptions options = Options.getDefaults(LocalExecutionOptions.class);
    LocalSpawnRunner runner =
        new TestedLocalSpawnRunner(
            fs.getPath("/execroot"),
            options,
            resourceManager,
            makeProcessWrapper(fs, options),
            LocalSpawnRunnerTest::keepLocalEnvUnchanged);

    FileOutErr fileOutErr = new FileOutErr(fs.getPath("/out/stdout"), fs.getPath("/out/stderr"));
    SpawnExecutionContextForTesting policy = new SpawnExecutionContextForTesting(fileOutErr);
    assertThat(fs.getPath("/execroot").createDirectory()).isTrue();
    assertThrows(InterruptedException.class, () -> runner.execAsync(SIMPLE_SPAWN, policy).get());
    Thread.interrupted();
    assertThat(policy.lockOutputFilesCalled).isTrue();
  }

  @Test
  public void interruptWaitsForProcessExit() throws Exception {
    assumeTrue(OS.getCurrent() != OS.WINDOWS);

    Path tempDir = TestUtils.createUniqueTmpDir(new JavaIoFileSystem(DigestHashFunction.SHA256));

    LocalSpawnRunner runner =
        new LocalSpawnRunner(
            tempDir,
            Options.getDefaults(LocalExecutionOptions.class),
            resourceManager,
            LocalEnvProvider.forCurrentOs(ImmutableMap.of()),
            /*binTools=*/ null,
            /*processWrapper=*/ null,
            Mockito.mock(RunfilesTreeUpdater.class));
    FileOutErr fileOutErr =
        new FileOutErr(tempDir.getRelative("stdout"), tempDir.getRelative("stderr"));
    SpawnExecutionContextForTesting policy = new SpawnExecutionContextForTesting(fileOutErr);

    // This test to exercise a race condition by attempting an operation multiple times. We can get
    // false positives (the test passing without us catching a problem), so try a few times. When
    // implementing this fix on 2019-09-11, this specific configuration was sufficient to catch the
    // previously-existent bug.
    int tries = 10;
    int delaySeconds = 1;

    Path content = tempDir.getChild("content");
    Path started = tempDir.getChild("started");
    // Start a subprocess that blocks until it is killed, and when it is, writes some output to
    // a temporary file after some delay.
    String script =
        "trap 'sleep "
            + delaySeconds
            + "; echo foo >"
            + content.getPathString()
            + "; exit 1' TERM; "
            + "touch "
            + started.getPathString()
            + "; "
            + "while :; do "
            + "  echo 'waiting to be killed'; "
            + "  sleep 1; "
            + "done";
    Spawn spawn = new SpawnBuilder("/bin/sh", "-c", script).build();

    ExecutorService executor = Executors.newSingleThreadExecutor();
    try {
      for (int i = 0; i < tries; i++) {
        content.delete();
        started.delete();
        Semaphore interruptCaught = new Semaphore(0);
        Future<?> future =
            executor.submit(
                () -> {
                  try {
                    runner.exec(spawn, policy);
                  } catch (InterruptedException e) {
                    interruptCaught.release();
                  } catch (Throwable t) {
                    throw new IllegalStateException(t);
                  }
                });
        // Wait until we know the subprocess has started so that delivering a termination signal
        // to it triggers the delayed write to the file.
        while (!started.exists()) {
          Thread.sleep(1);
        }
        future.cancel(true);
        interruptCaught.acquireUninterruptibly();
        // At this point, the subprocess must have fully stopped so write some content to the file
        // and expect that these contents remain unmodified.
        FileSystemUtils.writeContent(content, StandardCharsets.UTF_8, "bar");
        // Wait for longer than the spawn takes to exit before we check the file contents to ensure
        // that we properly awaited for termination of the subprocess.
        Thread.sleep(delaySeconds * 2 * 1000);
        assertThat(FileSystemUtils.readContent(content, StandardCharsets.UTF_8)).isEqualTo("bar");
      }
    } finally {
      executor.shutdown();
    }
  }

  @Test
  public void checkPrefetchCalled() throws Exception {
    FileSystem fs = setupEnvironmentForFakeExecution();

    SubprocessFactory factory = mock(SubprocessFactory.class);
    when(factory.create(any())).thenReturn(new FinishedSubprocess(0));
    SubprocessBuilder.setDefaultSubprocessFactory(factory);

    LocalExecutionOptions options = Options.getDefaults(LocalExecutionOptions.class);
    LocalSpawnRunner runner =
        new TestedLocalSpawnRunner(
            fs.getPath("/execroot"),
            options,
            resourceManager,
            makeProcessWrapper(fs, options),
            LocalSpawnRunnerTest::keepLocalEnvUnchanged);

    FileOutErr fileOutErr = new FileOutErr(fs.getPath("/out/stdout"), fs.getPath("/out/stderr"));
    SpawnExecutionContextForTesting policy = new SpawnExecutionContextForTesting(fileOutErr);
    policy.timeoutMillis = 123 * 1000L;
    assertThat(fs.getPath("/execroot").createDirectory()).isTrue();
    runner.execAsync(SIMPLE_SPAWN, policy).get();
    assertThat(policy.prefetchCalled).isTrue();
  }

  @Test
  public void checkNoPrefetchCalled() throws Exception {
    FileSystem fs = setupEnvironmentForFakeExecution();

    SubprocessFactory factory = mock(SubprocessFactory.class);
    when(factory.create(any())).thenReturn(new FinishedSubprocess(0));
    SubprocessBuilder.setDefaultSubprocessFactory(factory);

    LocalExecutionOptions options = Options.getDefaults(LocalExecutionOptions.class);
    LocalSpawnRunner runner =
        new TestedLocalSpawnRunner(
            fs.getPath("/execroot"),
            options,
            resourceManager,
            makeProcessWrapper(fs, options),
            LocalSpawnRunnerTest::keepLocalEnvUnchanged);

    FileOutErr fileOutErr = new FileOutErr(fs.getPath("/out/stdout"), fs.getPath("/out/stderr"));
    SpawnExecutionContextForTesting policy = new SpawnExecutionContextForTesting(fileOutErr);
    policy.timeoutMillis = 123 * 1000L;

    Spawn spawn = new SpawnBuilder("/bin/echo", "Hi!")
        .withExecutionInfo(ExecutionRequirements.DISABLE_LOCAL_PREFETCH, "").build();
    assertThat(fs.getPath("/execroot").createDirectory()).isTrue();
    runner.execAsync(spawn, policy).get();
    assertThat(policy.prefetchCalled).isFalse();
  }

  @Test
  public void checkLocalEnvProviderCalled() throws Exception {
    FileSystem fs = setupEnvironmentForFakeExecution();

    SubprocessFactory factory = mock(SubprocessFactory.class);
    when(factory.create(any())).thenReturn(new FinishedSubprocess(0));
    SubprocessBuilder.setDefaultSubprocessFactory(factory);
    LocalEnvProvider localEnvProvider = mock(LocalEnvProvider.class);

    LocalExecutionOptions options = Options.getDefaults(LocalExecutionOptions.class);
    LocalSpawnRunner runner =
        new TestedLocalSpawnRunner(
            fs.getPath("/execroot"),
            options,
            resourceManager,
            makeProcessWrapper(fs, options),
            localEnvProvider);

    FileOutErr fileOutErr = new FileOutErr(fs.getPath("/out/stdout"), fs.getPath("/out/stderr"));
    SpawnExecutionContextForTesting policy = new SpawnExecutionContextForTesting(fileOutErr);
    policy.timeoutMillis = 123 * 1000L;
    assertThat(fs.getPath("/execroot").createDirectory()).isTrue();

    runner.execAsync(SIMPLE_SPAWN, policy).get();
    verify(localEnvProvider)
        .rewriteLocalEnv(
            any(),
            any(),
            matches("^/execroot/tmp[0-9a-fA-F]+_[0-9a-fA-F]+/work$"));
  }

  /**
   * Copies the {@code process-wrapper} tool into the path under the temporary execRoot where the
   * {@link LocalSpawnRunner} expects to find it.
   */
  private void copyProcessWrapperIntoExecRoot(Path wrapperPath) throws IOException {
    File realProcessWrapperFile =
        new File(
            PathFragment.create(BlazeTestUtils.runfilesDir())
                .getRelative(TestConstants.PROCESS_WRAPPER_PATH)
                .getPathString());
    assertThat(realProcessWrapperFile.exists()).isTrue();

    wrapperPath.createDirectoryAndParents();
    File wrapperFile = wrapperPath.getPathFile();

    wrapperPath.delete();
    Files.copy(realProcessWrapperFile, wrapperFile);
    assertThat(wrapperPath.exists()).isTrue();

    wrapperPath.setExecutable(true);
  }

  /**
   * Copies the {@code spend_cpu_time} test util into the temporary execRoot so that the {@link
   * LocalSpawnRunner} can execute it.
   */
  private Path copyCpuTimeSpenderIntoExecRoot(Path execRoot) throws IOException {
    File realCpuTimeSpenderFile =
        new File(
            PathFragment.create(BlazeTestUtils.runfilesDir())
                .getRelative(TestConstants.CPU_TIME_SPENDER_PATH)
                .getPathString());
    assertThat(realCpuTimeSpenderFile.exists()).isTrue();

    Path execRootCpuTimeSpenderPath = execRoot.getRelative("spend-cpu-time");
    File execRootCpuTimeSpenderFile = execRootCpuTimeSpenderPath.getPathFile();

    assertThat(execRootCpuTimeSpenderPath.exists()).isFalse();
    Files.copy(realCpuTimeSpenderFile, execRootCpuTimeSpenderFile);
    assertThat(execRootCpuTimeSpenderPath.exists()).isTrue();

    execRootCpuTimeSpenderPath.setExecutable(true);

    return execRootCpuTimeSpenderPath;
  }

  private Path getTemporaryRoot(FileSystem fs, String name) throws IOException {
    Path tempDirPath = TestUtils.createUniqueTmpDir(fs);
    assertThat(tempDirPath.exists()).isTrue();

    Path root = tempDirPath.getRelative(name);
    assertThat(root.createDirectory()).isTrue();
    assertThat(root.exists()).isTrue();

    return root;
  }

  /**
   * Returns an execRoot {@link Path} inside a new temporary directory.
   *
   * <p>The temporary directory will be automatically deleted on exit.
   */
  private Path getTemporaryExecRoot(FileSystem fs) throws IOException {
    return getTemporaryRoot(fs, "execRoot");
  }


  private Path getTemporaryEmbeddedBin(FileSystem fs) throws  IOException {
    return getTemporaryRoot(fs, "embedded_bin");
  }

  @Test
  public void hasExecutionStatistics_whenOptionIsEnabled() throws Exception {
    // TODO(b/62588075) Currently no process-wrapper or execution statistics support in Windows.
    assumeTrue(OS.getCurrent() != OS.WINDOWS);

    FileSystem fs = new UnixFileSystem(DigestHashFunction.SHA256, /*hashAttributeName=*/ "");

    LocalExecutionOptions options = Options.getDefaults(LocalExecutionOptions.class);
    options.collectLocalExecutionStatistics = true;

    Duration minimumWallTimeToSpend = Duration.ofSeconds(10);

    Duration minimumUserTimeToSpend = minimumWallTimeToSpend;
    // Under normal loads we should be able to use a much lower bound for maxUserTime, but be
    // generous here in case of hardware issues.
    Duration maximumUserTimeToSpend = minimumUserTimeToSpend.plus(Duration.ofSeconds(20));

    Duration minimumSystemTimeToSpend = Duration.ZERO;
    // Under normal loads we should be able to use a much lower bound for maxSysTime, but be
    // generous here in case of hardware issues.
    Duration maximumSystemTimeToSpend = minimumSystemTimeToSpend.plus(Duration.ofSeconds(20));

    Path execRoot = getTemporaryExecRoot(fs);
    Path embeddedBinaries = getTemporaryEmbeddedBin(fs);
    BinTools binTools = BinTools.forEmbeddedBin(embeddedBinaries,
        ImmutableList.of("process-wrapper"));
    Path processWrapperPath = binTools.getEmbeddedPath("process-wrapper");
    copyProcessWrapperIntoExecRoot(processWrapperPath);
    Path cpuTimeSpenderPath = copyCpuTimeSpenderIntoExecRoot(execRoot);

    LocalSpawnRunner runner =
        new LocalSpawnRunner(
            execRoot,
            options,
            resourceManager,
            LocalSpawnRunnerTest::keepLocalEnvUnchanged,
            binTools,
            new ProcessWrapper(
                processWrapperPath, /*killDelay=*/ Duration.ZERO, /*gracefulSigterm=*/ false),
            Mockito.mock(RunfilesTreeUpdater.class));

    Spawn spawn =
        new SpawnBuilder(
                cpuTimeSpenderPath.getPathString(),
                String.valueOf(minimumUserTimeToSpend.getSeconds()),
                String.valueOf(minimumSystemTimeToSpend.getSeconds()))
            .build();

    FileOutErr fileOutErr = new FileOutErr(fs.getPath("/dev/null"), fs.getPath("/dev/null"));
    SpawnExecutionContextForTesting policy = new SpawnExecutionContextForTesting(fileOutErr);

    SpawnResult spawnResult = runner.execAsync(spawn, policy).get();

    assertThat(spawnResult.status()).isEqualTo(SpawnResult.Status.SUCCESS);
    assertThat(spawnResult.exitCode()).isEqualTo(0);
    assertThat(spawnResult.setupSuccess()).isTrue();
    assertThat(spawnResult.getExecutorHostName()).isEqualTo(NetUtil.getCachedShortHostName());

    assertThat(spawnResult.getWallTime()).isPresent();
    assertThat(spawnResult.getWallTime().get()).isAtLeast(minimumWallTimeToSpend);
    // Under heavy starvation, max wall time could be anything, so don't check it here.
    assertThat(spawnResult.getUserTime()).isPresent();
    assertThat(spawnResult.getUserTime().get()).isAtLeast(minimumUserTimeToSpend);
    assertThat(spawnResult.getUserTime().get()).isAtMost(maximumUserTimeToSpend);
    assertThat(spawnResult.getSystemTime()).isPresent();
    assertThat(spawnResult.getSystemTime().get()).isAtLeast(minimumSystemTimeToSpend);
    assertThat(spawnResult.getSystemTime().get()).isAtMost(maximumSystemTimeToSpend);
    assertThat(spawnResult.getNumBlockOutputOperations().get()).isAtLeast(0L);
    assertThat(spawnResult.getNumBlockInputOperations().get()).isAtLeast(0L);
    assertThat(spawnResult.getNumInvoluntaryContextSwitches().get()).isAtLeast(0L);
  }

  @Test
  public void hasNoExecutionStatistics_whenOptionIsDisabled() throws Exception {
    // TODO(b/62588075) Currently no process-wrapper or execution statistics support in Windows.
    assumeTrue(OS.getCurrent() != OS.WINDOWS);

    FileSystem fs = new UnixFileSystem(DigestHashFunction.SHA256, /*hashAttributeName=*/ "");

    LocalExecutionOptions options = Options.getDefaults(LocalExecutionOptions.class);
    options.collectLocalExecutionStatistics = false;

    Duration minimumWallTimeToSpend = Duration.ofSeconds(1);

    Duration minimumUserTimeToSpend = minimumWallTimeToSpend;
    Duration minimumSystemTimeToSpend = Duration.ZERO;

    Path execRoot = getTemporaryExecRoot(fs);
    Path embeddedBinaries = getTemporaryEmbeddedBin(fs);
    BinTools binTools = BinTools.forEmbeddedBin(embeddedBinaries,
        ImmutableList.of("process-wrapper"));
    Path processWrapperPath = binTools.getEmbeddedPath("process-wrapper");
    copyProcessWrapperIntoExecRoot(processWrapperPath);
    Path cpuTimeSpenderPath = copyCpuTimeSpenderIntoExecRoot(execRoot);

    LocalSpawnRunner runner =
        new LocalSpawnRunner(
            execRoot,
            options,
            resourceManager,
            LocalSpawnRunnerTest::keepLocalEnvUnchanged,
            binTools,
            new ProcessWrapper(
                processWrapperPath, /*killDelay=*/ Duration.ZERO, /*gracefulSigterm=*/ false),
            Mockito.mock(RunfilesTreeUpdater.class));

    Spawn spawn =
        new SpawnBuilder(
                cpuTimeSpenderPath.getPathString(),
                String.valueOf(minimumUserTimeToSpend.getSeconds()),
                String.valueOf(minimumSystemTimeToSpend.getSeconds()))
            .build();

    FileOutErr fileOutErr = new FileOutErr(fs.getPath("/dev/null"), fs.getPath("/dev/null"));
    SpawnExecutionContextForTesting policy = new SpawnExecutionContextForTesting(fileOutErr);

    SpawnResult spawnResult = runner.execAsync(spawn, policy).get();

    assertThat(spawnResult.status()).isEqualTo(SpawnResult.Status.SUCCESS);
    assertThat(spawnResult.exitCode()).isEqualTo(0);
    assertThat(spawnResult.setupSuccess()).isTrue();
    assertThat(spawnResult.getExecutorHostName()).isEqualTo(NetUtil.getCachedShortHostName());

    assertThat(spawnResult.getWallTime()).isPresent();
    assertThat(spawnResult.getWallTime().get()).isAtLeast(minimumWallTimeToSpend);
    // Under heavy starvation, max wall time could be anything, so don't check it here.
    assertThat(spawnResult.getUserTime()).isEmpty();
    assertThat(spawnResult.getSystemTime()).isEmpty();
    assertThat(spawnResult.getNumBlockOutputOperations()).isEmpty();
    assertThat(spawnResult.getNumBlockInputOperations()).isEmpty();
    assertThat(spawnResult.getNumInvoluntaryContextSwitches()).isEmpty();
  }

  // Check that relative paths in the Spawn are absolutized relative to the execroot passed to the
  // LocalSpawnRunner.
  @Test
  public void relativePath() throws Exception {
    // TODO(#3536): Make this test work on Windows.
    // The Command API implicitly absolutizes the path, and we get weird paths on Windows:
    // T:\execroot\execroot\_bin\process-wrapper
    assumeTrue(OS.getCurrent() != OS.WINDOWS);

    FileSystem fs = setupEnvironmentForFakeExecution();

    SubprocessFactory factory = mock(SubprocessFactory.class);
    ArgumentCaptor<SubprocessBuilder> captor = ArgumentCaptor.forClass(SubprocessBuilder.class);
    when(factory.create(captor.capture())).thenReturn(new FinishedSubprocess(0));
    SubprocessBuilder.setDefaultSubprocessFactory(factory);

    LocalSpawnRunner runner =
        new TestedLocalSpawnRunner(
            fs.getPath("/execroot"),
            Options.getDefaults(LocalExecutionOptions.class),
            resourceManager,
            /*processWrapper=*/ null,
            LocalSpawnRunnerTest::keepLocalEnvUnchanged);

    FileOutErr fileOutErr = new FileOutErr(fs.getPath("/out/stdout"), fs.getPath("/out/stderr"));
    SpawnExecutionContextForTesting policy = new SpawnExecutionContextForTesting(fileOutErr);
    assertThat(fs.getPath("/execroot").createDirectory()).isTrue();
    runner.execAsync(new SpawnBuilder("foo/bar", "Hi!").build(), policy).get();
    verify(factory).create(any(SubprocessBuilder.class));

    assertThat(captor.getValue().getArgv()).containsExactly("/execroot/foo/bar", "Hi!");
  }
}
