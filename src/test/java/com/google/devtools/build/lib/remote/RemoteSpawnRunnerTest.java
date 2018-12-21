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
package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static org.junit.Assert.fail;
import static org.mockito.Matchers.any;
import static org.mockito.Matchers.eq;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyZeroInteractions;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.ExecuteRequest;
import build.bazel.remote.execution.v2.ExecuteResponse;
import build.bazel.remote.execution.v2.LogFile;
import build.bazel.remote.execution.v2.Platform;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.eventbus.EventBus;
import com.google.common.io.ByteStreams;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.CommandLines.ParamFileActionInput;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SimpleSpawn;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.SpawnExecException;
import com.google.devtools.build.lib.exec.SpawnInputExpander;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.exec.SpawnRunner.ProgressStatus;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.exec.util.FakeOwner;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.DigestUtil.ActionKey;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import com.google.rpc.Code;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Collection;
import java.util.SortedMap;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

/** Tests for {@link com.google.devtools.build.lib.remote.RemoteSpawnRunner} */
@RunWith(JUnit4.class)
public class RemoteSpawnRunnerTest {

  private static final ImmutableMap<String, String> NO_CACHE =
      ImmutableMap.of(ExecutionRequirements.NO_CACHE, "");
  private static ListeningScheduledExecutorService retryService;

  private Path execRoot;
  private Path logDir;
  private DigestUtil digestUtil;
  private FakeActionInputFileCache fakeFileCache;
  private FileOutErr outErr;

  private RemoteOptions options;
  private RemoteRetrier retrier;

  @Mock private AbstractRemoteActionCache cache;

  @Mock
  private GrpcRemoteExecutor executor;

  @Mock
  private SpawnRunner localRunner;

  // The action key of the Spawn returned by newSimpleSpawn().
  private final String simpleActionId =
      "eb45b20cc979d504f96b9efc9a08c48103c6f017afa09c0df5c70a5f92a98ea8";

  @BeforeClass
  public static void beforeEverything() {
    retryService = MoreExecutors.listeningDecorator(Executors.newScheduledThreadPool(1));
  }

  @Before
  public final void setUp() throws Exception {
    MockitoAnnotations.initMocks(this);
    digestUtil = new DigestUtil(DigestHashFunction.SHA256);
    FileSystem fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);
    execRoot = fs.getPath("/exec/root");
    logDir = fs.getPath("/server-logs");
    FileSystemUtils.createDirectoryAndParents(execRoot);
    fakeFileCache = new FakeActionInputFileCache(execRoot);

    Path stdout = fs.getPath("/tmp/stdout");
    Path stderr = fs.getPath("/tmp/stderr");
    FileSystemUtils.createDirectoryAndParents(stdout.getParentDirectory());
    FileSystemUtils.createDirectoryAndParents(stderr.getParentDirectory());
    outErr = new FileOutErr(stdout, stderr);

    options = Options.getDefaults(RemoteOptions.class);
    retrier = RemoteModule.createExecuteRetrier(options, retryService);
  }

  @AfterClass
  public static void afterEverything() {
    retryService.shutdownNow();
  }

  @Test
  @SuppressWarnings("unchecked")
  public void nonCachableSpawnsShouldNotBeCached_remote() throws Exception {
    // Test that if a spawn is marked "NO_CACHE" then it's not fetched from a remote cache.
    // It should be executed remotely, but marked non-cacheable to remote execution, so that
    // the action result is not saved in the remote cache.

    options.remoteAcceptCached = true;
    options.remoteLocalFallback = false;
    options.remoteUploadLocalResults = true;
    options.remoteResultCachePriority = 1;
    options.remoteExecutionPriority = 2;

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            options,
            Options.getDefaults(ExecutionOptions.class),
            new AtomicReference<>(localRunner),
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
            retrier,
            digestUtil,
            logDir);

    ExecuteResponse succeeded = ExecuteResponse.newBuilder().setResult(
        ActionResult.newBuilder().setExitCode(0).build()).build();
    when(executor.executeRemotely(any(ExecuteRequest.class))).thenReturn(succeeded);

    Spawn spawn = new SimpleSpawn(
        new FakeOwner("foo", "bar"),
        /*arguments=*/ ImmutableList.of(),
        /*environment=*/ ImmutableMap.of(),
        NO_CACHE,
        /*inputs=*/ ImmutableList.of(),
        /*outputs=*/ ImmutableList.<ActionInput>of(),
        ResourceSet.ZERO);

    SpawnExecutionContext policy = new FakeSpawnExecutionContext(spawn);

    runner.exec(spawn, policy);

    ArgumentCaptor<ExecuteRequest> requestCaptor = ArgumentCaptor.forClass(ExecuteRequest.class);
    verify(executor).executeRemotely(requestCaptor.capture());
    assertThat(requestCaptor.getValue().getSkipCacheLookup()).isTrue();
    assertThat(requestCaptor.getValue().getResultsCachePolicy().getPriority()).isEqualTo(1);
    assertThat(requestCaptor.getValue().getExecutionPolicy().getPriority()).isEqualTo(2);
    // TODO(olaola): verify that the uploaded action has the doNotCache set.

    verify(cache, never())
        .getCachedActionResult(any(ActionKey.class));
    verify(cache, never())
        .upload(
            any(ActionKey.class),
            any(Action.class),
            any(Command.class),
            any(Path.class),
            any(Collection.class),
            any(FileOutErr.class),
            any(Boolean.class));
    verifyZeroInteractions(localRunner);
  }

  @Test
  @SuppressWarnings("unchecked")
  public void nonCachableSpawnsShouldNotBeCached_local() throws Exception {
    // Test that if a spawn is executed locally, due to the local fallback, that its result is not
    // uploaded to the remote cache. However, the artifacts should still be uploaded.

    options.remoteAcceptCached = true;
    options.remoteLocalFallback = true;
    options.remoteUploadLocalResults = true;

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            options,
            Options.getDefaults(ExecutionOptions.class),
            new AtomicReference<>(localRunner),
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            null,
            retrier,
            digestUtil,
            logDir);

    // Throw an IOException to trigger the local fallback.
    when(executor.executeRemotely(any(ExecuteRequest.class))).thenThrow(IOException.class);

    Spawn spawn = new SimpleSpawn(
        new FakeOwner("foo", "bar"),
        /*arguments=*/ ImmutableList.of(),
        /*environment=*/ ImmutableMap.of(),
        NO_CACHE,
        /*inputs=*/ ImmutableList.of(),
        /*outputs=*/ ImmutableList.<ActionInput>of(),
        ResourceSet.ZERO);

    SpawnExecutionContext policy = new FakeSpawnExecutionContext(spawn);

    runner.exec(spawn, policy);

    verify(localRunner).exec(spawn, policy);

    verify(cache, never())
        .getCachedActionResult(any(ActionKey.class));
    verify(cache)
        .upload(
            any(ActionKey.class),
            any(Action.class),
            any(Command.class),
            any(Path.class),
            any(Collection.class),
            any(FileOutErr.class),
            eq(false));
  }

  @Test
  @SuppressWarnings("unchecked")
  public void failedActionShouldOnlyUploadOutputs() throws Exception {
    // Test that the outputs of a failed locally executed action are uploaded to a remote cache,
    // but the action result itself is not.

    options.remoteUploadLocalResults = true;

    RemoteSpawnRunner runner =
        spy(
            new RemoteSpawnRunner(
                execRoot,
                options,
                Options.getDefaults(ExecutionOptions.class),
                new AtomicReference<>(localRunner),
                true,
                /*cmdlineReporter=*/ null,
                "build-req-id",
                "command-id",
                cache,
                null,
                retrier,
                digestUtil,
                logDir));

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = new FakeSpawnExecutionContext(spawn);

    SpawnResult res = Mockito.mock(SpawnResult.class);
    when(res.exitCode()).thenReturn(1);
    when(res.status()).thenReturn(Status.EXECUTION_FAILED);
    when(localRunner.exec(eq(spawn), eq(policy))).thenReturn(res);

    assertThat(runner.exec(spawn, policy)).isSameAs(res);

    verify(localRunner).exec(eq(spawn), eq(policy));
    verify(runner)
        .execLocallyAndUpload(
            eq(spawn),
            eq(policy),
            any(SortedMap.class),
            eq(cache),
            any(ActionKey.class),
            any(Action.class),
            any(Command.class),
            eq(true));
    verify(cache)
        .upload(
            any(ActionKey.class),
            any(Action.class),
            any(Command.class),
            any(Path.class),
            any(Collection.class),
            any(FileOutErr.class),
            eq(false));
  }

  @Test
  public void dontAcceptFailedCachedAction() throws Exception {
    // Test that bazel fails if the remote cache serves a failed action.

    RemoteOptions options = Options.getDefaults(RemoteOptions.class);

    ActionResult failedAction = ActionResult.newBuilder().setExitCode(1).build();
    when(cache.getCachedActionResult(any(ActionKey.class))).thenReturn(failedAction);

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = new FakeSpawnExecutionContext(spawn);

    RemoteSpawnRunner runner =
        spy(
            new RemoteSpawnRunner(
                execRoot,
                options,
                Options.getDefaults(ExecutionOptions.class),
                new AtomicReference<>(localRunner),
                true,
                /*cmdlineReporter=*/ null,
                "build-req-id",
                "command-id",
                cache,
                null,
                retrier,
                digestUtil,
                logDir));

    try {
      runner.exec(spawn, policy);
      fail("Expected exception");
    } catch (EnvironmentalExecException expected) {
      // Intentionally left empty.
    }
  }

  @Test
  @SuppressWarnings("unchecked")
  public void printWarningIfCacheIsDown() throws Exception {
    // If we try to upload to a local cache, that is down a warning should be printed.

    options.remoteUploadLocalResults = true;
    options.remoteLocalFallback = true;

    Reporter reporter = new Reporter(new EventBus());
    StoredEventHandler eventHandler = new StoredEventHandler();
    reporter.addHandler(eventHandler);

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            options,
            Options.getDefaults(ExecutionOptions.class),
            new AtomicReference<>(localRunner),
            false,
            reporter,
            "build-req-id",
            "command-id",
            cache,
            null,
            retrier,
            digestUtil,
            logDir);

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = new FakeSpawnExecutionContext(spawn);

    when(cache.getCachedActionResult(any(ActionKey.class)))
        .thenThrow(new IOException("cache down"));

    doThrow(new IOException("cache down"))
        .when(cache)
        .upload(
            any(ActionKey.class),
            any(Action.class),
            any(Command.class),
            any(Path.class),
            any(Collection.class),
            any(FileOutErr.class),
            eq(true));

    SpawnResult res =
        new SpawnResult.Builder()
            .setStatus(Status.SUCCESS)
            .setExitCode(0)
            .setRunnerName("test")
            .build();
    when(localRunner.exec(eq(spawn), eq(policy))).thenReturn(res);

    assertThat(runner.exec(spawn, policy)).isEqualTo(res);

    verify(localRunner).exec(eq(spawn), eq(policy));

    assertThat(eventHandler.getEvents()).hasSize(1);

    Event evt = eventHandler.getEvents().get(0);
    assertThat(evt.getKind()).isEqualTo(EventKind.WARNING);
    assertThat(evt.getMessage()).contains("fail");
    assertThat(evt.getMessage()).contains("upload");
  }

  @Test
  public void noRemoteExecutorFallbackFails() throws Exception {
    // Errors from the fallback runner should be propogated out of the remote runner.

    options.remoteUploadLocalResults = true;
    options.remoteLocalFallback = true;

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            options,
            Options.getDefaults(ExecutionOptions.class),
            new AtomicReference<>(localRunner),
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            null,
            retrier,
            digestUtil,
            logDir);

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = new FakeSpawnExecutionContext(spawn);

    when(cache.getCachedActionResult(any(ActionKey.class))).thenReturn(null);

    IOException err = new IOException("local execution error");
    when(localRunner.exec(eq(spawn), eq(policy))).thenThrow(err);

    try {
      runner.exec(spawn, policy);
      fail("expected IOException to be raised");
    } catch (IOException e) {
      assertThat(e).isSameAs(err);
    }

    verify(localRunner).exec(eq(spawn), eq(policy));
  }

  @Test
  public void remoteCacheErrorFallbackFails() throws Exception {
    // Errors from the fallback runner should be propogated out of the remote runner.

    options.remoteUploadLocalResults = true;
    options.remoteLocalFallback = true;

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            options,
            Options.getDefaults(ExecutionOptions.class),
            new AtomicReference<>(localRunner),
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            null,
            retrier,
            digestUtil,
            logDir);

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = new FakeSpawnExecutionContext(spawn);

    when(cache.getCachedActionResult(any(ActionKey.class))).thenThrow(new IOException());

    IOException err = new IOException("local execution error");
    when(localRunner.exec(eq(spawn), eq(policy))).thenThrow(err);

    try {
      runner.exec(spawn, policy);
      fail("expected IOException to be raised");
    } catch (IOException e) {
      assertThat(e).isSameAs(err);
    }

    verify(localRunner).exec(eq(spawn), eq(policy));
  }

  @Test
  public void testLocalFallbackFailureRemoteExecutorFailure() throws Exception {
    options.remoteLocalFallback = true;

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            options,
            Options.getDefaults(ExecutionOptions.class),
            new AtomicReference<>(localRunner),
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
            retrier,
            digestUtil,
            logDir);

    when(cache.getCachedActionResult(any(ActionKey.class))).thenReturn(null);
    when(executor.executeRemotely(any(ExecuteRequest.class))).thenThrow(new IOException());

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = new FakeSpawnExecutionContext(spawn);

    IOException err = new IOException("local execution error");
    when(localRunner.exec(eq(spawn), eq(policy))).thenThrow(err);

    try {
      runner.exec(spawn, policy);
      fail("expected IOException to be raised");
    } catch (IOException e) {
      assertThat(e).isSameAs(err);
    }

    verify(localRunner).exec(eq(spawn), eq(policy));
  }

  @Test
  public void testHumanReadableServerLogsSavedForFailingAction() throws Exception {
    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            Options.getDefaults(RemoteOptions.class),
            Options.getDefaults(ExecutionOptions.class),
            new AtomicReference<>(localRunner),
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
            retrier,
            digestUtil,
            logDir);

    Digest logDigest = digestUtil.computeAsUtf8("bla");
    Path logPath = logDir.getRelative(simpleActionId).getRelative("logname");
    when(executor.executeRemotely(any(ExecuteRequest.class)))
        .thenReturn(
            ExecuteResponse.newBuilder()
                .putServerLogs(
                    "logname",
                    LogFile.newBuilder().setHumanReadable(true).setDigest(logDigest).build())
                .setResult(ActionResult.newBuilder().setExitCode(31).build())
                .build());
    SettableFuture<Void> completed = SettableFuture.create();
    completed.set(null);
    when(cache.downloadFile(eq(logPath), eq(logDigest))).thenReturn(completed);

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = new FakeSpawnExecutionContext(spawn);

    SpawnResult res = runner.exec(spawn, policy);
    assertThat(res.status()).isEqualTo(Status.NON_ZERO_EXIT);

    verify(executor).executeRemotely(any(ExecuteRequest.class));
    verify(cache).downloadFile(eq(logPath), eq(logDigest));
  }

  @Test
  public void testHumanReadableServerLogsSavedForFailingActionWithStatus() throws Exception {
    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            Options.getDefaults(RemoteOptions.class),
            Options.getDefaults(ExecutionOptions.class),
            new AtomicReference<>(localRunner),
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
            retrier,
            digestUtil,
            logDir);

    Digest logDigest = digestUtil.computeAsUtf8("bla");
    Path logPath = logDir.getRelative(simpleActionId).getRelative("logname");
    com.google.rpc.Status timeoutStatus =
        com.google.rpc.Status.newBuilder().setCode(Code.DEADLINE_EXCEEDED.getNumber()).build();
    ExecuteResponse resp =
        ExecuteResponse.newBuilder()
            .putServerLogs(
                "logname", LogFile.newBuilder().setHumanReadable(true).setDigest(logDigest).build())
            .setStatus(timeoutStatus)
            .build();
    when(executor.executeRemotely(any(ExecuteRequest.class)))
        .thenThrow(new Retrier.RetryException(
                "", 1, new ExecutionStatusException(resp.getStatus(), resp)));
    SettableFuture<Void> completed = SettableFuture.create();
    completed.set(null);
    when(cache.downloadFile(eq(logPath), eq(logDigest))).thenReturn(completed);

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = new FakeSpawnExecutionContext(spawn);

    SpawnResult res = runner.exec(spawn, policy);
    assertThat(res.status()).isEqualTo(Status.TIMEOUT);

    verify(executor).executeRemotely(any(ExecuteRequest.class));
    verify(cache).downloadFile(eq(logPath), eq(logDigest));
  }

  @Test
  public void testNonHumanReadableServerLogsNotSaved() throws Exception {
    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            Options.getDefaults(RemoteOptions.class),
            Options.getDefaults(ExecutionOptions.class),
            new AtomicReference<>(localRunner),
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
            retrier,
            digestUtil,
            logDir);

    Digest logDigest = digestUtil.computeAsUtf8("bla");
    ActionResult result = ActionResult.newBuilder().setExitCode(31).build();
    when(executor.executeRemotely(any(ExecuteRequest.class)))
        .thenReturn(
            ExecuteResponse.newBuilder()
                .putServerLogs(
                    "logname",
                    LogFile.newBuilder().setDigest(logDigest).build())
                .setResult(result)
                .build());

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = new FakeSpawnExecutionContext(spawn);

    SpawnResult res = runner.exec(spawn, policy);
    assertThat(res.status()).isEqualTo(Status.NON_ZERO_EXIT);

    verify(executor).executeRemotely(any(ExecuteRequest.class));
    verify(cache).download(eq(result), eq(execRoot), any(FileOutErr.class));
    verify(cache, never()).downloadFile(any(Path.class), any(Digest.class));
  }

  @Test
  public void testServerLogsNotSavedForSuccessfulAction() throws Exception {
    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            Options.getDefaults(RemoteOptions.class),
            Options.getDefaults(ExecutionOptions.class),
            new AtomicReference<>(localRunner),
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
            retrier,
            digestUtil,
            logDir);

    Digest logDigest = digestUtil.computeAsUtf8("bla");
    ActionResult result = ActionResult.newBuilder().setExitCode(0).build();
    when(executor.executeRemotely(any(ExecuteRequest.class)))
        .thenReturn(
            ExecuteResponse.newBuilder()
                .putServerLogs(
                    "logname",
                    LogFile.newBuilder().setHumanReadable(true).setDigest(logDigest).build())
                .setResult(result)
                .build());

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = new FakeSpawnExecutionContext(spawn);

    SpawnResult res = runner.exec(spawn, policy);
    assertThat(res.status()).isEqualTo(Status.SUCCESS);

    verify(executor).executeRemotely(any(ExecuteRequest.class));
    verify(cache).download(eq(result), eq(execRoot), any(FileOutErr.class));
    verify(cache, never()).downloadFile(any(Path.class), any(Digest.class));
  }

  @Test
  public void cacheDownloadFailureTriggersRemoteExecution() throws Exception {
    // If downloading a cached action fails, remote execution should be tried.

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            options,
            Options.getDefaults(ExecutionOptions.class),
            new AtomicReference<>(localRunner),
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
            retrier,
            digestUtil,
            logDir);

    ActionResult cachedResult = ActionResult.newBuilder().setExitCode(0).build();
    when(cache.getCachedActionResult(any(ActionKey.class))).thenReturn(cachedResult);
    Retrier.RetryException downloadFailure =
        new Retrier.RetryException(
            "", 1, new CacheNotFoundException(Digest.getDefaultInstance(), digestUtil));
    doThrow(downloadFailure)
        .when(cache)
        .download(eq(cachedResult), any(Path.class), any(FileOutErr.class));
    ActionResult execResult = ActionResult.newBuilder().setExitCode(31).build();
    ExecuteResponse succeeded = ExecuteResponse.newBuilder().setResult(execResult).build();
    when(executor.executeRemotely(any(ExecuteRequest.class))).thenReturn(succeeded);
    doNothing().when(cache).download(eq(execResult), any(Path.class), any(FileOutErr.class));

    Spawn spawn = newSimpleSpawn();

    SpawnExecutionContext policy = new FakeSpawnExecutionContext(spawn);

    SpawnResult res = runner.exec(spawn, policy);
    assertThat(res.status()).isEqualTo(Status.NON_ZERO_EXIT);
    assertThat(res.exitCode()).isEqualTo(31);

    verify(executor).executeRemotely(any(ExecuteRequest.class));
  }

  @Test
  public void testRemoteExecutionTimeout() throws Exception {
    // If remote execution times out the SpawnResult status should be TIMEOUT.

    options.remoteLocalFallback = false;

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            options,
            Options.getDefaults(ExecutionOptions.class),
            new AtomicReference<>(localRunner),
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
            retrier,
            digestUtil,
            logDir);

    ActionResult cachedResult = ActionResult.newBuilder().setExitCode(0).build();
    when(cache.getCachedActionResult(any(ActionKey.class))).thenReturn(null);
    ExecuteResponse resp =
        ExecuteResponse.newBuilder()
            .setResult(cachedResult)
            .setStatus(
                com.google.rpc.Status.newBuilder()
                    .setCode(Code.DEADLINE_EXCEEDED.getNumber())
                    .build())
            .build();
    when(executor.executeRemotely(any(ExecuteRequest.class)))
        .thenThrow(
            new Retrier.RetryException(
                "", 1, new ExecutionStatusException(resp.getStatus(), resp)));

    Spawn spawn = newSimpleSpawn();

    SpawnExecutionContext policy = new FakeSpawnExecutionContext(spawn);

    SpawnResult res = runner.exec(spawn, policy);
    assertThat(res.status()).isEqualTo(Status.TIMEOUT);

    verify(executor).executeRemotely(any(ExecuteRequest.class));
    verify(cache).download(eq(cachedResult), eq(execRoot), any(FileOutErr.class));
  }

  @Test
  public void testRemoteExecutionTimeoutDoesNotTriggerFallback() throws Exception {
    // If remote execution times out the SpawnResult status should be TIMEOUT, regardess of local
    // fallback option.

    options.remoteLocalFallback = true;

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            options,
            Options.getDefaults(ExecutionOptions.class),
            new AtomicReference<>(localRunner),
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
            retrier,
            digestUtil,
            logDir);

    ActionResult cachedResult = ActionResult.newBuilder().setExitCode(0).build();
    when(cache.getCachedActionResult(any(ActionKey.class))).thenReturn(null);
    ExecuteResponse resp =
        ExecuteResponse.newBuilder()
            .setResult(cachedResult)
            .setStatus(
                com.google.rpc.Status.newBuilder()
                    .setCode(Code.DEADLINE_EXCEEDED.getNumber())
                    .build())
            .build();
    when(executor.executeRemotely(any(ExecuteRequest.class)))
        .thenThrow(
            new Retrier.RetryException(
                "", 1, new ExecutionStatusException(resp.getStatus(), resp)));

    Spawn spawn = newSimpleSpawn();

    SpawnExecutionContext policy = new FakeSpawnExecutionContext(spawn);

    SpawnResult res = runner.exec(spawn, policy);
    assertThat(res.status()).isEqualTo(Status.TIMEOUT);

    verify(executor).executeRemotely(any(ExecuteRequest.class));
    verify(cache).download(eq(cachedResult), eq(execRoot), any(FileOutErr.class));
    verify(localRunner, never()).exec(eq(spawn), eq(policy));
  }

  @Test
  public void testRemoteExecutionCommandFailureDoesNotTriggerFallback() throws Exception {
    options.remoteLocalFallback = true;

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            options,
            Options.getDefaults(ExecutionOptions.class),
            new AtomicReference<>(localRunner),
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
            retrier,
            digestUtil,
            logDir);

    ActionResult cachedResult = ActionResult.newBuilder().setExitCode(0).build();
    when(cache.getCachedActionResult(any(ActionKey.class))).thenReturn(null);
    ExecuteResponse failed = ExecuteResponse.newBuilder().setResult(
        ActionResult.newBuilder().setExitCode(33).build()).build();
    when(executor.executeRemotely(any(ExecuteRequest.class))).thenReturn(failed);

    Spawn spawn = newSimpleSpawn();

    SpawnExecutionContext policy = new FakeSpawnExecutionContext(spawn);

    SpawnResult res = runner.exec(spawn, policy);
    assertThat(res.status()).isEqualTo(Status.NON_ZERO_EXIT);
    assertThat(res.exitCode()).isEqualTo(33);

    verify(executor).executeRemotely(any(ExecuteRequest.class));
    verify(cache, never()).download(eq(cachedResult), eq(execRoot), any(FileOutErr.class));
    verify(localRunner, never()).exec(eq(spawn), eq(policy));
  }

  @Test
  public void testExitCode_executorfailure() throws Exception {
    // If we get a failure due to the remote cache not working, the exit code should be
    // ExitCode.REMOTE_ERROR.

    options.remoteLocalFallback = false;

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            options,
            Options.getDefaults(ExecutionOptions.class),
            new AtomicReference<>(localRunner),
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
            retrier,
            digestUtil,
            logDir);

    when(cache.getCachedActionResult(any(ActionKey.class))).thenReturn(null);
    when(executor.executeRemotely(any(ExecuteRequest.class))).thenThrow(new IOException("reasons"));

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = new FakeSpawnExecutionContext(spawn);

    try {
      runner.exec(spawn, policy);
      fail("Exception expected");
    } catch (SpawnExecException e) {
      assertThat(e.getSpawnResult().exitCode())
          .isEqualTo(ExitCode.REMOTE_ERROR.getNumericExitCode());
      assertThat(e.getSpawnResult().getDetailMessage("", "", false, false)).contains("reasons");
    }
  }

  @Test
  public void testExitCode_executionfailure() throws Exception {
    // If we get a failure due to the remote executor not working, the exit code should be
    // ExitCode.REMOTE_ERROR.

    options.remoteLocalFallback = false;

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            options,
            Options.getDefaults(ExecutionOptions.class),
            new AtomicReference<>(localRunner),
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
            retrier,
            digestUtil,
            logDir);

    when(cache.getCachedActionResult(any(ActionKey.class))).thenThrow(new IOException("reasons"));

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = new FakeSpawnExecutionContext(spawn);

    try {
      runner.exec(spawn, policy);
      fail("Exception expected");
    } catch (SpawnExecException e) {
      assertThat(e.getSpawnResult().exitCode())
          .isEqualTo(ExitCode.REMOTE_ERROR.getNumericExitCode());
      assertThat(e.getSpawnResult().getDetailMessage("", "", false, false)).contains("reasons");
    }
  }

  @Test
  public void testMaterializeParamFiles() throws Exception {
    ExecutionOptions executionOptions =
        Options.parse(ExecutionOptions.class, "--materialize_param_files").getOptions();
    executionOptions.materializeParamFiles = true;
    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            Options.getDefaults(RemoteOptions.class),
            executionOptions,
            new AtomicReference<>(localRunner),
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
            retrier,
            digestUtil,
            logDir);

    ExecuteResponse succeeded =
        ExecuteResponse.newBuilder()
            .setResult(ActionResult.newBuilder().setExitCode(0).build())
            .build();
    when(executor.executeRemotely(any(ExecuteRequest.class))).thenReturn(succeeded);

    ImmutableList<String> args = ImmutableList.of("--foo", "--bar");
    ParamFileActionInput input =
        new ParamFileActionInput(
            PathFragment.create("out/param_file"), args, ParameterFileType.UNQUOTED, ISO_8859_1);
    Spawn spawn =
        new SimpleSpawn(
            new FakeOwner("foo", "bar"),
            /*arguments=*/ ImmutableList.of(),
            /*environment=*/ ImmutableMap.of(),
            /*executionInfo=*/ ImmutableMap.of(),
            ImmutableList.of(input),
            /*outputs=*/ ImmutableList.<ActionInput>of(),
            ResourceSet.ZERO);
    SpawnExecutionContext policy = new FakeSpawnExecutionContext(spawn);
    SpawnResult res = runner.exec(spawn, policy);
    assertThat(res.status()).isEqualTo(Status.SUCCESS);
    Path paramFile = execRoot.getRelative("out/param_file");
    assertThat(paramFile.exists()).isTrue();
    try (InputStream inputStream = paramFile.getInputStream()) {
      assertThat(
              new String(ByteStreams.toByteArray(inputStream), StandardCharsets.UTF_8).split("\n"))
          .asList()
          .containsExactly("--foo", "--bar");
    }
  }

  @Test
  public void testParsePlatformSortsProperties() throws Exception {
    String s =
        String.join(
            "\n",
            "properties: {",
            " name: \"b\"",
            " value: \"2\"",
            "}",
            "properties: {",
            " name: \"a\"",
            " value: \"1\"",
            "}");
    Platform expected =
        Platform.newBuilder()
            .addProperties(Platform.Property.newBuilder().setName("a").setValue("1"))
            .addProperties(Platform.Property.newBuilder().setName("b").setValue("2"))
            .build();
    assertThat(RemoteSpawnRunner.parsePlatform(null, s)).isEqualTo(expected);
  }

  private static Spawn newSimpleSpawn() {
    return new SimpleSpawn(
        new FakeOwner("foo", "bar"),
            /*arguments=*/ ImmutableList.of(),
            /*environment=*/ ImmutableMap.of(),
            /*executionInfo=*/ ImmutableMap.of(),
            /*inputs=*/ ImmutableList.of(),
            /*outputs=*/ ImmutableList.<ActionInput>of(),
        ResourceSet.ZERO);
  }

  // TODO(buchgr): Extract a common class to be used for testing.
  class FakeSpawnExecutionContext implements SpawnExecutionContext {

    private final ArtifactExpander artifactExpander =
        (artifact, output) -> output.add(artifact);

    private final Spawn spawn;

    FakeSpawnExecutionContext(Spawn spawn) {
      this.spawn = spawn;
    }

    @Override
    public int getId() {
      return 0;
    }

    @Override
    public void prefetchInputs() throws IOException {
      throw new UnsupportedOperationException();
    }

    @Override
    public void lockOutputFiles() throws InterruptedException {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean speculating() {
      return false;
    }

    @Override
    public MetadataProvider getMetadataProvider() {
      return fakeFileCache;
    }

    @Override
    public ArtifactExpander getArtifactExpander() {
      throw new UnsupportedOperationException();
    }

    @Override
    public Duration getTimeout() {
      return Duration.ZERO;
    }

    @Override
    public FileOutErr getFileOutErr() {
      return outErr;
    }

    @Override
    public SortedMap<PathFragment, ActionInput> getInputMapping(
        boolean expandTreeArtifactsInRunfiles) throws IOException {
      return new SpawnInputExpander(execRoot, /*strict*/ false)
          .getInputMapping(spawn, artifactExpander, ArtifactPathResolver.IDENTITY, fakeFileCache,
              true);
    }

    @Override
    public void report(ProgressStatus state, String name) {
      assertThat(state).isEqualTo(ProgressStatus.EXECUTING);
    }
  }
}
