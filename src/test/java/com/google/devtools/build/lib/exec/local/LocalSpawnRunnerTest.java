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
import static org.junit.Assert.fail;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.exec.ActionInputPrefetcher;
import com.google.devtools.build.lib.exec.SpawnResult;
import com.google.devtools.build.lib.exec.SpawnRunner.ProgressStatus;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionPolicy;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.shell.JavaSubprocessFactory;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import com.google.devtools.build.lib.util.NetUtil;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Filter;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import java.util.regex.Pattern;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentCaptor;

/**
 * Unit tests for {@link LocalSpawnRunner}.
 */
@RunWith(JUnit4.class)
public class LocalSpawnRunnerTest {
  private static final boolean USE_WRAPPER = true;
  private static final boolean NO_WRAPPER = false;

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

  private static final class SubprocessInterceptor implements Subprocess.Factory {
    @Override
    public Subprocess create(SubprocessBuilder params) throws IOException {
      throw new UnsupportedOperationException();
    }
  }

  private final class SpawnExecutionPolicyForTesting implements SpawnExecutionPolicy {
    private final List<ProgressStatus> reportedStatus = new ArrayList<>();
    private final TreeMap<PathFragment, ActionInput> inputMapping = new TreeMap<>();

    @Override
    public void lockOutputFiles() throws InterruptedException {
      calledLockOutputFiles = true;
    }

    @Override
    public ActionInputFileCache getActionInputFileCache() {
      return mockFileCache;
    }

    @Override
    public long getTimeoutMillis() {
      return timeoutMillis;
    }

    @Override
    public FileOutErr getFileOutErr() {
      return outErr;
    }

    @Override
    public SortedMap<PathFragment, ActionInput> getInputMapping() {
      return inputMapping;
    }

    @Override
    public void report(ProgressStatus state) {
      reportedStatus.add(state);
    }
  }

  private FileSystem fs;
  private final ActionInputFileCache mockFileCache = mock(ActionInputFileCache.class);
  private final ResourceManager resourceManager = ResourceManager.instanceForTestingOnly();

  private Logger logger;
  private final AtomicInteger execCount = new AtomicInteger();
  private FileOutErr outErr;
  private long timeoutMillis = 0;
  private boolean calledLockOutputFiles;

  private final SpawnExecutionPolicyForTesting policy = new SpawnExecutionPolicyForTesting();

  @Before
  public final void setup() throws Exception  {
    logger = Logger.getAnonymousLogger();
    logger.setFilter(new Filter() {
      @Override
      public boolean isLoggable(LogRecord record) {
        return false;
      }
    });
    fs = new InMemoryFileSystem();
    // Prevent any subprocess execution at all.
    SubprocessBuilder.setSubprocessFactory(new SubprocessInterceptor());
    resourceManager.setAvailableResources(
        ResourceSet.create(/*memoryMb=*/1, /*cpuUsage=*/1, /*ioUsage=*/1, /*localTestCount=*/1));
  }

  @After
  public final void tearDown() {
    SubprocessBuilder.setSubprocessFactory(JavaSubprocessFactory.INSTANCE);
  }

  @Test
  public void vanillaZeroExit() throws Exception {
    Subprocess.Factory factory = mock(Subprocess.Factory.class);
    ArgumentCaptor<SubprocessBuilder> captor = ArgumentCaptor.forClass(SubprocessBuilder.class);
    when(factory.create(captor.capture())).thenReturn(new FinishedSubprocess(0));
    SubprocessBuilder.setSubprocessFactory(factory);

    LocalExecutionOptions options = Options.getDefaults(LocalExecutionOptions.class);
    options.localSigkillGraceSeconds = 456;
    LocalSpawnRunner runner = new LocalSpawnRunner(
        logger, execCount, fs.getPath("/execroot"), ActionInputPrefetcher.NONE, options,
        resourceManager, USE_WRAPPER);

    timeoutMillis = 123 * 1000L;
    outErr = new FileOutErr(fs.getPath("/out/stdout"), fs.getPath("/out/stderr"));
    SpawnResult result = runner.exec(SIMPLE_SPAWN, policy);
    verify(factory).create(any(SubprocessBuilder.class));
    assertThat(result.status()).isEqualTo(SpawnResult.Status.SUCCESS);
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.getExecutorHostName()).isEqualTo(NetUtil.findShortHostName());

    assertThat(captor.getValue().getArgv())
        .isEqualTo(ImmutableList.of(
            // process-wrapper timeout grace_time stdout stderr
            "/execroot/_bin/process-wrapper", "123.0", "456.0", "/out/stdout", "/out/stderr",
            "/bin/echo", "Hi!"));
    assertThat(captor.getValue().getEnv()).containsExactly("VARIABLE", "value");
    assertThat(captor.getValue().getTimeoutMillis()).isEqualTo(-1);

    assertThat(calledLockOutputFiles).isTrue();
    assertThat(policy.reportedStatus)
        .containsExactly(ProgressStatus.SCHEDULING, ProgressStatus.EXECUTING).inOrder();
  }

  @Test
  public void noProcessWrapper() throws Exception {
    Subprocess.Factory factory = mock(Subprocess.Factory.class);
    ArgumentCaptor<SubprocessBuilder> captor = ArgumentCaptor.forClass(SubprocessBuilder.class);
    when(factory.create(captor.capture())).thenReturn(new FinishedSubprocess(0));
    SubprocessBuilder.setSubprocessFactory(factory);

    LocalExecutionOptions options = Options.getDefaults(LocalExecutionOptions.class);
    options.localSigkillGraceSeconds = 456;
    LocalSpawnRunner runner = new LocalSpawnRunner(
        logger, execCount, fs.getPath("/execroot"), ActionInputPrefetcher.NONE, options,
        resourceManager, NO_WRAPPER);

    timeoutMillis = 123 * 1000L;
    outErr = new FileOutErr(fs.getPath("/out/stdout"), fs.getPath("/out/stderr"));
    SpawnResult result = runner.exec(SIMPLE_SPAWN, policy);
    verify(factory).create(any());
    assertThat(result.status()).isEqualTo(SpawnResult.Status.SUCCESS);
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.getExecutorHostName()).isEqualTo(NetUtil.findShortHostName());

    assertThat(captor.getValue().getArgv())
        .isEqualTo(ImmutableList.of("/bin/echo", "Hi!"));
    assertThat(captor.getValue().getEnv()).containsExactly("VARIABLE", "value");
    // Without the process wrapper, we use the Command API to enforce the timeout.
    assertThat(captor.getValue().getTimeoutMillis()).isEqualTo(timeoutMillis);

    assertThat(calledLockOutputFiles).isTrue();
  }

  @Test
  public void nonZeroExit() throws Exception {
    Subprocess.Factory factory = mock(Subprocess.Factory.class);
    ArgumentCaptor<SubprocessBuilder> captor = ArgumentCaptor.forClass(SubprocessBuilder.class);
    when(factory.create(captor.capture())).thenReturn(new FinishedSubprocess(3));
    SubprocessBuilder.setSubprocessFactory(factory);

    LocalExecutionOptions options = Options.getDefaults(LocalExecutionOptions.class);
    LocalSpawnRunner runner = new LocalSpawnRunner(
        logger, execCount, fs.getPath("/execroot"), ActionInputPrefetcher.NONE, options,
        resourceManager, USE_WRAPPER);

    outErr = new FileOutErr(fs.getPath("/out/stdout"), fs.getPath("/out/stderr"));
    SpawnResult result = runner.exec(SIMPLE_SPAWN, policy);
    verify(factory).create(any(SubprocessBuilder.class));
    assertThat(result.status()).isEqualTo(SpawnResult.Status.SUCCESS);
    assertThat(result.exitCode()).isEqualTo(3);
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.getExecutorHostName()).isEqualTo(NetUtil.findShortHostName());

    assertThat(captor.getValue().getArgv())
        .isEqualTo(ImmutableList.of(
            // process-wrapper timeout grace_time stdout stderr
            "/execroot/_bin/process-wrapper", "0.0", "15.0", "/out/stdout", "/out/stderr",
            "/bin/echo", "Hi!"));
    assertThat(captor.getValue().getEnv()).containsExactly("VARIABLE", "value");

    assertThat(calledLockOutputFiles).isTrue();
  }

  @Test
  public void processStartupThrows() throws Exception {
    Subprocess.Factory factory = mock(Subprocess.Factory.class);
    ArgumentCaptor<SubprocessBuilder> captor = ArgumentCaptor.forClass(SubprocessBuilder.class);
    when(factory.create(captor.capture())).thenThrow(new IOException("I'm sorry, Dave"));
    SubprocessBuilder.setSubprocessFactory(factory);

    LocalExecutionOptions options = Options.getDefaults(LocalExecutionOptions.class);
    LocalSpawnRunner runner = new LocalSpawnRunner(
        logger, execCount, fs.getPath("/execroot"), ActionInputPrefetcher.NONE, options,
        resourceManager, USE_WRAPPER);

    assertThat(fs.getPath("/out").createDirectory()).isTrue();
    outErr = new FileOutErr(fs.getPath("/out/stdout"), fs.getPath("/out/stderr"));
    SpawnResult result = runner.exec(SIMPLE_SPAWN, policy);
    verify(factory).create(any(SubprocessBuilder.class));
    assertThat(result.status()).isEqualTo(SpawnResult.Status.EXECUTION_FAILED);
    assertThat(result.exitCode()).isEqualTo(-1);
    assertThat(result.setupSuccess()).isFalse();
    assertThat(result.getWallTimeMillis()).isEqualTo(0);
    assertThat(result.getExecutorHostName()).isEqualTo(NetUtil.findShortHostName());

    assertThat(FileSystemUtils.readContent(fs.getPath("/out/stderr"), StandardCharsets.UTF_8))
        .isEqualTo("Action failed to execute: java.io.IOException: I'm sorry, Dave\n");

    assertThat(calledLockOutputFiles).isTrue();
  }

  @Test
  public void disallowLocalExecution() throws Exception {
    LocalExecutionOptions options = Options.getDefaults(LocalExecutionOptions.class);
    options.allowedLocalAction = Pattern.compile("none");
    LocalSpawnRunner runner = new LocalSpawnRunner(
        logger, execCount, fs.getPath("/execroot"), ActionInputPrefetcher.NONE, options,
        resourceManager, USE_WRAPPER);

    outErr = new FileOutErr();
    SpawnResult reply = runner.exec(SIMPLE_SPAWN, policy);
    assertThat(reply.status()).isEqualTo(SpawnResult.Status.LOCAL_ACTION_NOT_ALLOWED);
    assertThat(reply.exitCode()).isEqualTo(-1);
    assertThat(reply.setupSuccess()).isFalse();
    assertThat(reply.getWallTimeMillis()).isEqualTo(0);
    assertThat(reply.getExecutorHostName()).isEqualTo(NetUtil.findShortHostName());

    // TODO(ulfjack): Maybe we should only lock after checking?
    assertThat(calledLockOutputFiles).isTrue();
  }

  @Test
  public void interruptedException() throws Exception {
    Subprocess.Factory factory = mock(Subprocess.Factory.class);
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
    SubprocessBuilder.setSubprocessFactory(factory);

    LocalExecutionOptions options = Options.getDefaults(LocalExecutionOptions.class);
    LocalSpawnRunner runner = new LocalSpawnRunner(
        logger, execCount, fs.getPath("/execroot"), ActionInputPrefetcher.NONE, options,
        resourceManager, USE_WRAPPER);

    outErr = new FileOutErr(fs.getPath("/out/stdout"), fs.getPath("/out/stderr"));
    try {
      runner.exec(SIMPLE_SPAWN, policy);
      fail();
    } catch (InterruptedException expected) {
      // Clear the interrupted status or subsequent tests in the same process will fail.
      Thread.interrupted();
    }
    assertThat(calledLockOutputFiles).isTrue();
  }

  @Test
  public void checkPrefetchCalled() throws Exception {
    Subprocess.Factory factory = mock(Subprocess.Factory.class);
    when(factory.create(any())).thenReturn(new FinishedSubprocess(0));
    SubprocessBuilder.setSubprocessFactory(factory);
    ActionInputPrefetcher mockPrefetcher = mock(ActionInputPrefetcher.class);

    LocalExecutionOptions options = Options.getDefaults(LocalExecutionOptions.class);
    LocalSpawnRunner runner = new LocalSpawnRunner(
        logger, execCount, fs.getPath("/execroot"), mockPrefetcher, options, resourceManager,
        USE_WRAPPER);

    timeoutMillis = 123 * 1000L;
    outErr = new FileOutErr(fs.getPath("/out/stdout"), fs.getPath("/out/stderr"));
    runner.exec(SIMPLE_SPAWN, policy);
    verify(mockPrefetcher).prefetchFiles(any());
  }

  @Test
  public void checkNoPrefetchCalled() throws Exception {
    Subprocess.Factory factory = mock(Subprocess.Factory.class);
    when(factory.create(any())).thenReturn(new FinishedSubprocess(0));
    SubprocessBuilder.setSubprocessFactory(factory);
    ActionInputPrefetcher mockPrefetcher = mock(ActionInputPrefetcher.class);
    doThrow(new RuntimeException("Called prefetch!")).when(mockPrefetcher).prefetchFiles(any());

    LocalExecutionOptions options = Options.getDefaults(LocalExecutionOptions.class);
    LocalSpawnRunner runner = new LocalSpawnRunner(
        logger, execCount, fs.getPath("/execroot"), mockPrefetcher, options, resourceManager,
        USE_WRAPPER);

    timeoutMillis = 123 * 1000L;
    outErr = new FileOutErr(fs.getPath("/out/stdout"), fs.getPath("/out/stderr"));

    Spawn spawn = new SpawnBuilder("/bin/echo", "Hi!")
        .withExecutionInfo(ExecutionRequirements.DISABLE_LOCAL_PREFETCH, "").build();
    // This would throw if the runner called prefetchFiles().
    runner.exec(spawn, policy);
  }
}
