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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyCollection;
import static org.mockito.Matchers.eq;
import static org.mockito.Mockito.doNothing;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.verifyZeroInteractions;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.ExecuteRequest;
import build.bazel.remote.execution.v2.ExecuteResponse;
import build.bazel.remote.execution.v2.LogFile;
import com.google.common.collect.ClassToInstanceMap;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.EventBus;
import com.google.common.io.ByteStreams;
import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.CommandLines.ParamFileActionInput;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SimpleSpawn;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.exec.AbstractSpawnStrategy;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.RemoteLocalFallbackRegistry;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.exec.util.FakeOwner;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.ActionKey;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.options.RemoteOutputsMode;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.FakeSpawnExecutionContext;
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
import java.util.List;
import java.util.concurrent.Executors;
import javax.annotation.Nullable;
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

  private RemoteOptions remoteOptions;

  @Mock private RemoteExecutionCache cache;

  @Mock private GrpcRemoteExecutor executor;

  @Mock private SpawnRunner localRunner;

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

    remoteOptions = Options.getDefaults(RemoteOptions.class);
  }

  @AfterClass
  public static void afterEverything() {
    retryService.shutdownNow();
  }

  @Test
  public void nonCachableSpawnsShouldNotBeCached_remote() throws Exception {
    // Test that if a spawn is marked "NO_CACHE" then it's not fetched from a remote cache.
    // It should be executed remotely, but marked non-cacheable to remote execution, so that
    // the action result is not saved in the remote cache.

    remoteOptions.remoteAcceptCached = true;
    remoteOptions.remoteLocalFallback = false;
    remoteOptions.remoteUploadLocalResults = true;
    remoteOptions.remoteResultCachePriority = 1;
    remoteOptions.remoteExecutionPriority = 2;

    RemoteSpawnRunner runner = newSpawnRunner();

    ExecuteResponse succeeded =
        ExecuteResponse.newBuilder()
            .setResult(ActionResult.newBuilder().setExitCode(0).build())
            .build();
    when(executor.executeRemotely(any(ExecuteRequest.class))).thenReturn(succeeded);

    Spawn spawn = simpleSpawnWithExecutionInfo(NO_CACHE);
    SpawnExecutionContext policy = getSpawnContext(spawn);

    runner.exec(spawn, policy);

    ArgumentCaptor<ExecuteRequest> requestCaptor = ArgumentCaptor.forClass(ExecuteRequest.class);
    verify(executor).executeRemotely(requestCaptor.capture());
    assertThat(requestCaptor.getValue().getSkipCacheLookup()).isTrue();
    assertThat(requestCaptor.getValue().getResultsCachePolicy().getPriority()).isEqualTo(1);
    assertThat(requestCaptor.getValue().getExecutionPolicy().getPriority()).isEqualTo(2);
    // TODO(olaola): verify that the uploaded action has the doNotCache set.

    verify(cache, never()).downloadActionResult(any(ActionKey.class));
    verify(cache, never()).upload(any(), any(), any(), any(), any(), any());
    verifyZeroInteractions(localRunner);
  }

  private FakeSpawnExecutionContext getSpawnContext(Spawn spawn) {
    AbstractSpawnStrategy fakeLocalStrategy = new AbstractSpawnStrategy(execRoot, localRunner) {};
    ClassToInstanceMap<ActionContext> actionContextRegistry =
        ImmutableClassToInstanceMap.of(RemoteLocalFallbackRegistry.class, () -> fakeLocalStrategy);
    return new FakeSpawnExecutionContext(
        spawn, fakeFileCache, execRoot, outErr, actionContextRegistry);
  }

  @Test
  public void nonCachableSpawnsShouldNotBeCached_localFallback() throws Exception {
    // Test that if a non-cachable spawn is executed locally due to the local fallback,
    // that its result is not uploaded to the remote cache.

    remoteOptions.remoteAcceptCached = true;
    remoteOptions.remoteLocalFallback = true;
    remoteOptions.remoteUploadLocalResults = true;

    RemoteSpawnRunner runner = newSpawnRunner();

    // Throw an IOException to trigger the local fallback.
    when(executor.executeRemotely(any(ExecuteRequest.class))).thenThrow(IOException.class);

    Spawn spawn = simpleSpawnWithExecutionInfo(NO_CACHE);
    SpawnExecutionContext policy = getSpawnContext(spawn);

    runner.exec(spawn, policy);

    verify(localRunner).exec(spawn, policy);
    verify(cache).ensureInputsPresent(any(), any());
    verifyNoMoreInteractions(cache);
  }

  @Test
  public void cachableSpawnsShouldBeCached_localFallback() throws Exception {
    // Test that if a cachable spawn is executed locally due to the local fallback,
    // that its result is uploaded to the remote cache.

    remoteOptions.remoteAcceptCached = true;
    remoteOptions.remoteLocalFallback = true;
    remoteOptions.remoteUploadLocalResults = true;

    RemoteSpawnRunner runner = spy(newSpawnRunner());

    // Throw an IOException to trigger the local fallback.
    when(executor.executeRemotely(any(ExecuteRequest.class))).thenThrow(IOException.class);

    SpawnResult res =
        new SpawnResult.Builder()
            .setStatus(Status.SUCCESS)
            .setExitCode(0)
            .setRunnerName("test")
            .build();
    when(localRunner.exec(any(Spawn.class), any(SpawnExecutionContext.class))).thenReturn(res);

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = getSpawnContext(spawn);

    SpawnResult result = runner.exec(spawn, policy);
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(result.status()).isEqualTo(Status.SUCCESS);
    verify(localRunner).exec(eq(spawn), eq(policy));
    verify(runner)
        .execLocallyAndUpload(
            eq(spawn), eq(policy), any(), any(), any(), any(), /* uploadLocalResults= */ eq(true));
    verify(cache).upload(any(), any(), any(), any(), any(), any());
  }

  @Test
  public void failedLocalActionShouldNotBeUploaded() throws Exception {
    // Test that the outputs of a locally executed action that failed are not uploaded.

    remoteOptions.remoteLocalFallback = true;
    remoteOptions.remoteUploadLocalResults = true;

    RemoteSpawnRunner runner = spy(newSpawnRunner());

    // Throw an IOException to trigger the local fallback.
    when(executor.executeRemotely(any(ExecuteRequest.class))).thenThrow(IOException.class);

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = getSpawnContext(spawn);

    SpawnResult res = Mockito.mock(SpawnResult.class);
    when(res.exitCode()).thenReturn(1);
    when(res.status()).thenReturn(Status.EXECUTION_FAILED);
    when(localRunner.exec(eq(spawn), eq(policy))).thenReturn(res);

    assertThat(runner.exec(spawn, policy)).isSameInstanceAs(res);

    verify(localRunner).exec(eq(spawn), eq(policy));
    verify(runner)
        .execLocallyAndUpload(
            eq(spawn),
            eq(policy),
            any(),
            any(),
            any(),
            any(),
            /* uploadLocalResults= */ eq(true));
    verify(cache, never()).upload(any(), any(), any(), any(), any(), any());
  }

  @Test
  public void treatFailedCachedActionAsCacheMiss_local() throws Exception {
    // Test that bazel treats failed cache action as a cache miss and attempts to execute action
    // locally

    remoteOptions.remoteLocalFallback = true;
    remoteOptions.remoteUploadLocalResults = true;

    ActionResult failedAction = ActionResult.newBuilder().setExitCode(1).build();
    when(cache.downloadActionResult(any(ActionKey.class))).thenReturn(failedAction);

    RemoteSpawnRunner runner = spy(newSpawnRunner());
    // Throw an IOException to trigger the local fallback.
    when(executor.executeRemotely(any(ExecuteRequest.class))).thenThrow(IOException.class);

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = getSpawnContext(spawn);

    SpawnResult succeeded =
        new SpawnResult.Builder()
            .setStatus(Status.SUCCESS)
            .setExitCode(0)
            .setRunnerName("test")
            .build();
    when(localRunner.exec(eq(spawn), eq(policy))).thenReturn(succeeded);

    runner.exec(spawn, policy);

    verify(localRunner).exec(eq(spawn), eq(policy));
    verify(runner)
        .execLocallyAndUpload(
            eq(spawn),
            eq(policy),
            any(),
            any(),
            any(),
            any(),
            /* uploadLocalResults= */ eq(true));
    verify(cache).upload(any(), any(), any(), any(), any(), any());
    verify(cache, never()).download(any(ActionResult.class), any(Path.class), eq(outErr), any());
  }

  @Test
  public void treatFailedCachedActionAsCacheMiss_remote() throws Exception {
    // Test that bazel treats failed cache action as a cache miss and attempts to execute action
    // remotely

    ActionResult failedAction = ActionResult.newBuilder().setExitCode(1).build();
    when(cache.downloadActionResult(any(ActionKey.class))).thenReturn(failedAction);

    RemoteSpawnRunner runner = newSpawnRunner();

    ExecuteResponse succeeded =
        ExecuteResponse.newBuilder()
            .setResult(ActionResult.newBuilder().setExitCode(0).build())
            .build();
    when(executor.executeRemotely(any(ExecuteRequest.class))).thenReturn(succeeded);
    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = getSpawnContext(spawn);

    runner.exec(spawn, policy);

    ArgumentCaptor<ExecuteRequest> requestCaptor = ArgumentCaptor.forClass(ExecuteRequest.class);
    verify(executor).executeRemotely(requestCaptor.capture());
    assertThat(requestCaptor.getValue().getSkipCacheLookup()).isTrue();
  }

  @Test
  public void printWarningIfCacheIsDown() throws Exception {
    // If we try to upload to a local cache, that is down a warning should be printed.

    remoteOptions.remoteUploadLocalResults = true;
    remoteOptions.remoteLocalFallback = true;

    Reporter reporter = new Reporter(new EventBus());
    StoredEventHandler eventHandler = new StoredEventHandler();
    reporter.addHandler(eventHandler);

    RemoteSpawnRunner runner = newSpawnRunner(reporter);
    // Trigger local fallback
    when(executor.executeRemotely(any(ExecuteRequest.class))).thenThrow(new IOException());

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = getSpawnContext(spawn);

    when(cache.downloadActionResult(any(ActionKey.class))).thenThrow(new IOException("cache down"));

    doThrow(new IOException("cache down"))
        .when(cache)
        .upload(any(), any(), any(), any(), any(), any());

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
    // Errors from the fallback runner should be propagated out of the remote runner.

    remoteOptions.remoteUploadLocalResults = true;
    remoteOptions.remoteLocalFallback = true;

    RemoteSpawnRunner runner = newSpawnRunner();
    // Trigger local fallback
    when(executor.executeRemotely(any(ExecuteRequest.class))).thenThrow(new IOException());

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = getSpawnContext(spawn);

    when(cache.downloadActionResult(any(ActionKey.class))).thenReturn(null);

    IOException err = new IOException("local execution error");
    when(localRunner.exec(eq(spawn), eq(policy))).thenThrow(err);

    IOException e = assertThrows(IOException.class, () -> runner.exec(spawn, policy));
    assertThat(e).isSameInstanceAs(err);

    verify(localRunner).exec(eq(spawn), eq(policy));
  }

  @Test
  public void remoteCacheErrorFallbackFails() throws Exception {
    // Errors from the fallback runner should be propagated out of the remote runner.

    remoteOptions.remoteUploadLocalResults = true;
    remoteOptions.remoteLocalFallback = true;

    RemoteSpawnRunner runner = newSpawnRunner();
    // Trigger local fallback
    when(executor.executeRemotely(any(ExecuteRequest.class))).thenThrow(new IOException());

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = getSpawnContext(spawn);

    when(cache.downloadActionResult(any(ActionKey.class))).thenThrow(new IOException());

    IOException err = new IOException("local execution error");
    when(localRunner.exec(eq(spawn), eq(policy))).thenThrow(err);

    IOException e = assertThrows(IOException.class, () -> runner.exec(spawn, policy));
    assertThat(e).isSameInstanceAs(err);

    verify(localRunner).exec(eq(spawn), eq(policy));
  }

  @Test
  public void testLocalFallbackFailureRemoteExecutorFailure() throws Exception {
    remoteOptions.remoteLocalFallback = true;

    RemoteSpawnRunner runner = newSpawnRunner();

    when(cache.downloadActionResult(any(ActionKey.class))).thenReturn(null);
    when(executor.executeRemotely(any(ExecuteRequest.class))).thenThrow(new IOException());

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = getSpawnContext(spawn);

    IOException err = new IOException("local execution error");
    when(localRunner.exec(eq(spawn), eq(policy))).thenThrow(err);

    IOException e = assertThrows(IOException.class, () -> runner.exec(spawn, policy));
    assertThat(e).isSameInstanceAs(err);

    verify(localRunner).exec(eq(spawn), eq(policy));
  }

  @Test
  public void testHumanReadableServerLogsSavedForFailingAction() throws Exception {
    RemoteSpawnRunner runner = newSpawnRunner();
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
    SpawnExecutionContext policy = getSpawnContext(spawn);

    SpawnResult res = runner.exec(spawn, policy);
    assertThat(res.status()).isEqualTo(Status.NON_ZERO_EXIT);

    verify(executor).executeRemotely(any(ExecuteRequest.class));
    verify(cache).downloadFile(eq(logPath), eq(logDigest));
  }

  @Test
  public void testHumanReadableServerLogsSavedForFailingActionWithStatus() throws Exception {
    RemoteSpawnRunner runner = newSpawnRunner();
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
        .thenThrow(new IOException(new ExecutionStatusException(resp.getStatus(), resp)));
    SettableFuture<Void> completed = SettableFuture.create();
    completed.set(null);
    when(cache.downloadFile(eq(logPath), eq(logDigest))).thenReturn(completed);

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = getSpawnContext(spawn);

    SpawnResult res = runner.exec(spawn, policy);
    assertThat(res.status()).isEqualTo(Status.TIMEOUT);

    verify(executor).executeRemotely(any(ExecuteRequest.class));
    verify(cache).downloadFile(eq(logPath), eq(logDigest));
  }

  @Test
  public void testNonHumanReadableServerLogsNotSaved() throws Exception {
    RemoteSpawnRunner runner = newSpawnRunner();

    Digest logDigest = digestUtil.computeAsUtf8("bla");
    ActionResult result = ActionResult.newBuilder().setExitCode(31).build();
    when(executor.executeRemotely(any(ExecuteRequest.class)))
        .thenReturn(
            ExecuteResponse.newBuilder()
                .putServerLogs("logname", LogFile.newBuilder().setDigest(logDigest).build())
                .setResult(result)
                .build());

    Spawn spawn = newSimpleSpawn();
    FakeSpawnExecutionContext policy = getSpawnContext(spawn);
    SpawnResult res = runner.exec(spawn, policy);
    assertThat(res.status()).isEqualTo(Status.NON_ZERO_EXIT);

    verify(executor).executeRemotely(any(ExecuteRequest.class));
    verify(cache).download(eq(result), eq(execRoot), any(FileOutErr.class), any());
    verify(cache, never()).downloadFile(any(Path.class), any(Digest.class));
  }

  @Test
  public void testServerLogsNotSavedForSuccessfulAction() throws Exception {
    RemoteSpawnRunner runner = newSpawnRunner();

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
    FakeSpawnExecutionContext policy = getSpawnContext(spawn);

    SpawnResult res = runner.exec(spawn, policy);
    assertThat(res.status()).isEqualTo(Status.SUCCESS);

    verify(executor).executeRemotely(any(ExecuteRequest.class));
    verify(cache).download(eq(result), eq(execRoot), any(FileOutErr.class), any());
    verify(cache, never()).downloadFile(any(Path.class), any(Digest.class));
  }

  @Test
  public void cacheDownloadFailureTriggersRemoteExecution() throws Exception {
    // If downloading a cached action fails, remote execution should be tried.

    RemoteSpawnRunner runner = newSpawnRunner();

    ActionResult cachedResult = ActionResult.newBuilder().setExitCode(0).build();
    when(cache.downloadActionResult(any(ActionKey.class))).thenReturn(cachedResult);
    Exception downloadFailure = new CacheNotFoundException(Digest.getDefaultInstance());
    doThrow(downloadFailure)
        .when(cache)
        .download(eq(cachedResult), any(Path.class), any(FileOutErr.class), any());
    ActionResult execResult = ActionResult.newBuilder().setExitCode(31).build();
    ExecuteResponse succeeded = ExecuteResponse.newBuilder().setResult(execResult).build();
    when(executor.executeRemotely(any(ExecuteRequest.class))).thenReturn(succeeded);
    doNothing().when(cache).download(eq(execResult), any(Path.class), any(FileOutErr.class), any());

    Spawn spawn = newSimpleSpawn();

    SpawnExecutionContext policy = getSpawnContext(spawn);

    SpawnResult res = runner.exec(spawn, policy);
    assertThat(res.status()).isEqualTo(Status.NON_ZERO_EXIT);
    assertThat(res.exitCode()).isEqualTo(31);

    verify(executor).executeRemotely(any(ExecuteRequest.class));
  }

  @Test
  public void resultsDownloadFailureTriggersRemoteExecutionWithSkipCacheLookup() throws Exception {
    // If downloading an action result fails, remote execution should be retried
    // with skip cache lookup enabled

    RemoteSpawnRunner runner = newSpawnRunner();

    when(cache.downloadActionResult(any(ActionKey.class))).thenReturn(null);
    ActionResult cachedResult = ActionResult.newBuilder().setExitCode(0).build();
    ActionResult execResult = ActionResult.newBuilder().setExitCode(31).build();
    ExecuteResponse cachedResponse =
        ExecuteResponse.newBuilder().setResult(cachedResult).setCachedResult(true).build();
    ExecuteResponse executedResponse = ExecuteResponse.newBuilder().setResult(execResult).build();
    when(executor.executeRemotely(any(ExecuteRequest.class)))
        .thenReturn(cachedResponse)
        .thenReturn(executedResponse);
    Exception downloadFailure = new CacheNotFoundException(Digest.getDefaultInstance());
    doThrow(downloadFailure)
        .when(cache)
        .download(eq(cachedResult), any(Path.class), any(FileOutErr.class), any());
    doNothing().when(cache).download(eq(execResult), any(Path.class), any(FileOutErr.class), any());

    Spawn spawn = newSimpleSpawn();

    SpawnExecutionContext policy = getSpawnContext(spawn);

    SpawnResult res = runner.exec(spawn, policy);
    assertThat(res.status()).isEqualTo(Status.NON_ZERO_EXIT);
    assertThat(res.exitCode()).isEqualTo(31);

    ArgumentCaptor<ExecuteRequest> requestCaptor = ArgumentCaptor.forClass(ExecuteRequest.class);
    verify(executor, times(2)).executeRemotely(requestCaptor.capture());
    List<ExecuteRequest> requests = requestCaptor.getAllValues();
    // first request should have been executed without skip cache lookup
    assertThat(requests.get(0).getSkipCacheLookup()).isFalse();
    // second should have been executed with skip cache lookup
    assertThat(requests.get(1).getSkipCacheLookup()).isTrue();
  }

  @Test
  public void testRemoteExecutionTimeout() throws Exception {
    // If remote execution times out the SpawnResult status should be TIMEOUT.

    remoteOptions.remoteLocalFallback = false;

    RemoteSpawnRunner runner = newSpawnRunner();

    ActionResult cachedResult = ActionResult.newBuilder().setExitCode(0).build();
    when(cache.downloadActionResult(any(ActionKey.class))).thenReturn(null);
    ExecuteResponse resp =
        ExecuteResponse.newBuilder()
            .setResult(cachedResult)
            .setStatus(
                com.google.rpc.Status.newBuilder()
                    .setCode(Code.DEADLINE_EXCEEDED.getNumber())
                    .build())
            .build();
    when(executor.executeRemotely(any(ExecuteRequest.class)))
        .thenThrow(new IOException(new ExecutionStatusException(resp.getStatus(), resp)));

    Spawn spawn = newSimpleSpawn();

    SpawnExecutionContext policy = getSpawnContext(spawn);

    SpawnResult res = runner.exec(spawn, policy);
    assertThat(res.status()).isEqualTo(Status.TIMEOUT);

    verify(executor).executeRemotely(any(ExecuteRequest.class));
    verify(cache).download(eq(cachedResult), eq(execRoot), any(FileOutErr.class), any());
  }

  @Test
  public void testRemoteExecutionTimeoutDoesNotTriggerFallback() throws Exception {
    // If remote execution times out the SpawnResult status should be TIMEOUT, regardess of local
    // fallback option.

    remoteOptions.remoteLocalFallback = true;

    RemoteSpawnRunner runner = newSpawnRunner();

    ActionResult cachedResult = ActionResult.newBuilder().setExitCode(0).build();
    when(cache.downloadActionResult(any(ActionKey.class))).thenReturn(null);
    ExecuteResponse resp =
        ExecuteResponse.newBuilder()
            .setResult(cachedResult)
            .setStatus(
                com.google.rpc.Status.newBuilder()
                    .setCode(Code.DEADLINE_EXCEEDED.getNumber())
                    .build())
            .build();
    when(executor.executeRemotely(any(ExecuteRequest.class)))
        .thenThrow(new IOException(new ExecutionStatusException(resp.getStatus(), resp)));

    Spawn spawn = newSimpleSpawn();

    SpawnExecutionContext policy = getSpawnContext(spawn);

    SpawnResult res = runner.exec(spawn, policy);
    assertThat(res.status()).isEqualTo(Status.TIMEOUT);

    verify(executor).executeRemotely(any(ExecuteRequest.class));
    verify(cache).download(eq(cachedResult), eq(execRoot), any(FileOutErr.class), any());
    verify(localRunner, never()).exec(eq(spawn), eq(policy));
  }

  @Test
  public void testRemoteExecutionCommandFailureDoesNotTriggerFallback() throws Exception {
    remoteOptions.remoteLocalFallback = true;

    RemoteSpawnRunner runner = newSpawnRunner();

    ActionResult cachedResult = ActionResult.newBuilder().setExitCode(0).build();
    when(cache.downloadActionResult(any(ActionKey.class))).thenReturn(null);
    ExecuteResponse failed =
        ExecuteResponse.newBuilder()
            .setResult(ActionResult.newBuilder().setExitCode(33).build())
            .build();
    when(executor.executeRemotely(any(ExecuteRequest.class))).thenReturn(failed);

    Spawn spawn = newSimpleSpawn();

    SpawnExecutionContext policy = getSpawnContext(spawn);

    SpawnResult res = runner.exec(spawn, policy);
    assertThat(res.status()).isEqualTo(Status.NON_ZERO_EXIT);
    assertThat(res.exitCode()).isEqualTo(33);

    verify(executor).executeRemotely(any(ExecuteRequest.class));
    verify(cache, never()).download(eq(cachedResult), eq(execRoot), any(FileOutErr.class), any());
    verify(localRunner, never()).exec(eq(spawn), eq(policy));
  }

  @Test
  public void testExitCode_executorfailure() throws Exception {
    // If we get a failure due to the remote cache not working, the exit code should be
    // ExitCode.REMOTE_ERROR.

    remoteOptions.remoteLocalFallback = false;

    RemoteSpawnRunner runner = newSpawnRunner();

    when(cache.downloadActionResult(any(ActionKey.class))).thenReturn(null);
    when(executor.executeRemotely(any(ExecuteRequest.class))).thenThrow(new IOException("reasons"));

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = getSpawnContext(spawn);

    SpawnResult result = runner.exec(spawn, policy);
    assertThat(result.exitCode()).isEqualTo(ExitCode.REMOTE_ERROR.getNumericExitCode());
    assertThat(result.getDetailMessage("", "", false, false)).contains("reasons");
  }

  @Test
  public void testExitCode_executionfailure() throws Exception {
    // If we get a failure due to the remote executor not working, the exit code should be
    // ExitCode.REMOTE_ERROR.

    remoteOptions.remoteLocalFallback = false;

    RemoteSpawnRunner runner = newSpawnRunner();

    when(cache.downloadActionResult(any(ActionKey.class))).thenThrow(new IOException("reasons"));

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = getSpawnContext(spawn);

    SpawnResult result = runner.exec(spawn, policy);
    assertThat(result.exitCode()).isEqualTo(ExitCode.REMOTE_ERROR.getNumericExitCode());
    assertThat(result.getDetailMessage("", "", false, false)).contains("reasons");
  }

  @Test
  public void testMaterializeParamFiles() throws Exception {
    testParamFilesAreMaterializedForFlag("--materialize_param_files");
  }

  @Test
  public void testMaterializeParamFilesIsImpliedBySubcommands() throws Exception {
    testParamFilesAreMaterializedForFlag("--subcommands");
  }

  private void testParamFilesAreMaterializedForFlag(String flag) throws Exception {
    ExecutionOptions executionOptions = Options.parse(ExecutionOptions.class, flag).getOptions();
    executionOptions.materializeParamFiles = true;
    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            Options.getDefaults(RemoteOptions.class),
            executionOptions,
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
            retryService,
            digestUtil,
            logDir,
            /* filesToDownload= */ ImmutableSet.of());

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
            /*inputs=*/ NestedSetBuilder.create(Order.STABLE_ORDER, input),
            /*outputs=*/ ImmutableSet.<ActionInput>of(),
            ResourceSet.ZERO);
    SpawnExecutionContext policy = getSpawnContext(spawn);
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
  public void testDownloadMinimalOnCacheHit() throws Exception {
    // arrange
    remoteOptions.remoteOutputsMode = RemoteOutputsMode.MINIMAL;

    ActionResult succeededAction = ActionResult.newBuilder().setExitCode(0).build();
    when(cache.downloadActionResult(any(ActionKey.class))).thenReturn(succeededAction);

    RemoteSpawnRunner runner = newSpawnRunner();

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = getSpawnContext(spawn);

    // act
    SpawnResult result = runner.exec(spawn, policy);
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(result.status()).isEqualTo(Status.SUCCESS);

    // assert
    verify(cache)
        .downloadMinimal(eq(succeededAction), anyCollection(), any(), any(), any(), any(), any());
    verify(cache, never()).download(any(ActionResult.class), any(Path.class), eq(outErr), any());
  }

  @Test
  public void testDownloadMinimalOnCacheMiss() throws Exception {
    // arrange
    remoteOptions.remoteOutputsMode = RemoteOutputsMode.MINIMAL;

    ActionResult succeededAction = ActionResult.newBuilder().setExitCode(0).build();
    ExecuteResponse succeeded = ExecuteResponse.newBuilder().setResult(succeededAction).build();
    when(executor.executeRemotely(any(ExecuteRequest.class))).thenReturn(succeeded);

    RemoteSpawnRunner runner = newSpawnRunner();

    Spawn spawn = newSimpleSpawn();
    FakeSpawnExecutionContext policy = getSpawnContext(spawn);

    // act
    SpawnResult result = runner.exec(spawn, policy);
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(result.status()).isEqualTo(Status.SUCCESS);

    // assert
    verify(executor).executeRemotely(any());
    verify(cache)
        .downloadMinimal(eq(succeededAction), anyCollection(), any(), any(), any(), any(), any());
    verify(cache, never()).download(any(ActionResult.class), any(Path.class), eq(outErr), any());
  }

  @Test
  public void testDownloadMinimalIoError() throws Exception {
    // arrange
    remoteOptions.remoteOutputsMode = RemoteOutputsMode.MINIMAL;

    ActionResult succeededAction = ActionResult.newBuilder().setExitCode(0).build();
    when(cache.downloadActionResult(any(ActionKey.class))).thenReturn(succeededAction);
    IOException downloadFailure = new IOException("downloadMinimal failed");
    when(cache.downloadMinimal(any(), anyCollection(), any(), any(), any(), any(), any()))
        .thenThrow(downloadFailure);

    RemoteSpawnRunner runner = newSpawnRunner();

    Spawn spawn = newSimpleSpawn();
    FakeSpawnExecutionContext policy = getSpawnContext(spawn);

    // act
    SpawnResult result = runner.exec(spawn, policy);
    assertThat(result.getFailureMessage()).isEqualTo(downloadFailure.getMessage());

    // assert
    verify(cache)
        .downloadMinimal(eq(succeededAction), anyCollection(), any(), any(), any(), any(), any());
    verify(cache, never()).download(any(ActionResult.class), any(Path.class), eq(outErr), any());
  }

  @Test
  public void testDownloadTopLevel() throws Exception {
    // arrange
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteOutputsMode = RemoteOutputsMode.TOPLEVEL;

    ArtifactRoot outputRoot = ArtifactRoot.asDerivedRoot(execRoot, execRoot.getRelative("outs"));
    Artifact topLevelOutput =
        ActionsTestUtil.createArtifact(outputRoot, outputRoot.getRoot().getRelative("foo.bin"));

    ActionResult succeededAction = ActionResult.newBuilder().setExitCode(0).build();
    when(cache.downloadActionResult(any(ActionKey.class))).thenReturn(succeededAction);

    RemoteSpawnRunner runner = newSpawnRunner(ImmutableSet.of(topLevelOutput));

    Spawn spawn = newSimpleSpawn(topLevelOutput);
    FakeSpawnExecutionContext policy = getSpawnContext(spawn);

    // act
    SpawnResult result = runner.exec(spawn, policy);
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(result.status()).isEqualTo(Status.SUCCESS);

    // assert
    verify(cache).download(eq(succeededAction), any(Path.class), eq(outErr), any());
    verify(cache, never())
        .downloadMinimal(eq(succeededAction), anyCollection(), any(), any(), any(), any(), any());
  }

  private static Spawn newSimpleSpawn(Artifact... outputs) {
    return simpleSpawnWithExecutionInfo(ImmutableMap.of(), outputs);
  }

  private static SimpleSpawn simpleSpawnWithExecutionInfo(
      ImmutableMap<String, String> executionInfo, Artifact... outputs) {
    return new SimpleSpawn(
        new FakeOwner("foo", "bar"),
        /*arguments=*/ ImmutableList.of(),
        /*environment=*/ ImmutableMap.of(),
        /*executionInfo=*/ executionInfo,
        /*inputs=*/ NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        /*outputs=*/ ImmutableSet.copyOf(outputs),
        ResourceSet.ZERO);
  }

  private RemoteSpawnRunner newSpawnRunner() {
    return newSpawnRunner(
        /* verboseFailures= */ false,
        executor,
        /* reporter= */ null,
        /* topLevelOutputs= */ ImmutableSet.of());
  }

  private RemoteSpawnRunner newSpawnRunner(Reporter reporter) {
    return newSpawnRunner(
        /* verboseFailures= */ false, executor, reporter, /* topLevelOutputs= */ ImmutableSet.of());
  }

  private RemoteSpawnRunner newSpawnRunner(ImmutableSet<ActionInput> topLevelOutputs) {
    return newSpawnRunner(
        /* verboseFailures= */ false, executor, /* reporter= */ null, topLevelOutputs);
  }

  private RemoteSpawnRunner newSpawnRunner(
      boolean verboseFailures,
      @Nullable GrpcRemoteExecutor executor,
      @Nullable Reporter reporter,
      ImmutableSet<ActionInput> topLevelOutputs) {
    return new RemoteSpawnRunner(
        execRoot,
        remoteOptions,
        Options.getDefaults(ExecutionOptions.class),
        verboseFailures,
        reporter,
        "build-req-id",
        "command-id",
        cache,
        executor,
        retryService,
        digestUtil,
        logDir,
        topLevelOutputs);
  }
}
