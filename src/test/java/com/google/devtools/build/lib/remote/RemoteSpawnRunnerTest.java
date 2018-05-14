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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
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
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystem.HashFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import com.google.devtools.remoteexecution.v1test.ActionResult;
import com.google.devtools.remoteexecution.v1test.Digest;
import com.google.devtools.remoteexecution.v1test.ExecuteRequest;
import com.google.devtools.remoteexecution.v1test.ExecuteResponse;
import com.google.devtools.remoteexecution.v1test.LogFile;
import com.google.protobuf.ByteString;
import com.google.rpc.Code;
import java.io.IOException;
import java.time.Duration;
import java.util.Collection;
import java.util.SortedMap;
import org.junit.Before;
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

  private Path execRoot;
  private Path logDir;
  private DigestUtil digestUtil;
  private FakeActionInputFileCache fakeFileCache;
  private FileOutErr outErr;

  @Mock private AbstractRemoteActionCache cache;

  @Mock
  private GrpcRemoteExecutor executor;

  @Mock
  private SpawnRunner localRunner;

  // The action key of the Spawn returned by newSimpleSpawn().
  private final String simpleActionId =
      "eb45b20cc979d504f96b9efc9a08c48103c6f017afa09c0df5c70a5f92a98ea8";

  @Before
  public final void setUp() throws Exception {
    MockitoAnnotations.initMocks(this);
    digestUtil = new DigestUtil(HashFunction.SHA256);
    FileSystem fs = new InMemoryFileSystem(new JavaClock(), HashFunction.SHA256);
    execRoot = fs.getPath("/exec/root");
    logDir = fs.getPath("/server-logs");
    FileSystemUtils.createDirectoryAndParents(execRoot);
    fakeFileCache = new FakeActionInputFileCache(execRoot);

    Path stdout = fs.getPath("/tmp/stdout");
    Path stderr = fs.getPath("/tmp/stderr");
    FileSystemUtils.createDirectoryAndParents(stdout.getParentDirectory());
    FileSystemUtils.createDirectoryAndParents(stderr.getParentDirectory());
    outErr = new FileOutErr(stdout, stderr);
  }

  @Test
  @SuppressWarnings("unchecked")
  public void nonCachableSpawnsShouldNotBeCached_remote() throws Exception {
    // Test that if a spawn is marked "NO_CACHE" then it's not fetched from a remote cache.
    // It should be executed remotely, but marked non-cacheable to remote execution, so that
    // the action result is not saved in the remote cache.

    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteAcceptCached = true;
    options.remoteLocalFallback = false;
    options.remoteUploadLocalResults = true;

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            options,
            localRunner,
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
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
    assertThat(requestCaptor.getValue().getAction().getDoNotCache()).isTrue();

    verify(cache, never())
        .getCachedActionResult(any(ActionKey.class));
    verify(cache, never())
        .upload(
            any(ActionKey.class),
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

    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteAcceptCached = true;
    options.remoteLocalFallback = true;
    options.remoteUploadLocalResults = true;

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            options,
            localRunner,
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            null,
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

    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteUploadLocalResults = true;

    RemoteSpawnRunner runner =
        spy(
            new RemoteSpawnRunner(
                execRoot,
                options,
                localRunner,
                true,
                /*cmdlineReporter=*/ null,
                "build-req-id",
                "command-id",
                cache,
                null,
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
    verify(runner).execLocallyAndUpload(eq(spawn), eq(policy), any(SortedMap.class), eq(cache),
        any(ActionKey.class));
    verify(cache)
        .upload(
            any(ActionKey.class),
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
                localRunner,
                true,
                /*cmdlineReporter=*/ null,
                "build-req-id",
                "command-id",
                cache,
                null,
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

    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteUploadLocalResults = true;
    options.remoteLocalFallback = true;

    Reporter reporter = new Reporter(new EventBus());
    StoredEventHandler eventHandler = new StoredEventHandler();
    reporter.addHandler(eventHandler);

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            options,
            localRunner,
            false,
            reporter,
            "build-req-id",
            "command-id",
            cache,
            null,
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

    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteUploadLocalResults = true;
    options.remoteLocalFallback = true;

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            options,
            localRunner,
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            null,
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

    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteUploadLocalResults = true;
    options.remoteLocalFallback = true;

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            options,
            localRunner,
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            null,
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
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteLocalFallback = true;

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            options,
            localRunner,
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
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
            localRunner,
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
            digestUtil,
            logDir);

    Digest logDigest = digestUtil.computeAsUtf8("bla");
    when(executor.executeRemotely(any(ExecuteRequest.class)))
        .thenReturn(
            ExecuteResponse.newBuilder()
                .putServerLogs(
                    "logname",
                    LogFile.newBuilder().setHumanReadable(true).setDigest(logDigest).build())
                .setResult(ActionResult.newBuilder().setExitCode(31).build())
                .build());

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = new FakeSpawnExecutionContext(spawn);

    SpawnResult res = runner.exec(spawn, policy);
    assertThat(res.status()).isEqualTo(Status.NON_ZERO_EXIT);

    verify(executor).executeRemotely(any(ExecuteRequest.class));
    Path logPath = logDir.getRelative(simpleActionId).getRelative("logname");
    verify(cache).downloadFile(eq(logPath), eq(logDigest), eq(false), eq(null));
  }

  @Test
  public void testHumanReadableServerLogsSavedForFailingActionWithStatus() throws Exception {
    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            Options.getDefaults(RemoteOptions.class),
            localRunner,
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
            digestUtil,
            logDir);

    Digest logDigest = digestUtil.computeAsUtf8("bla");
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

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = new FakeSpawnExecutionContext(spawn);

    SpawnResult res = runner.exec(spawn, policy);
    assertThat(res.status()).isEqualTo(Status.TIMEOUT);

    verify(executor).executeRemotely(any(ExecuteRequest.class));
    Path logPath = logDir.getRelative(simpleActionId).getRelative("logname");
    verify(cache).downloadFile(eq(logPath), eq(logDigest), eq(false), eq(null));
  }

  @Test
  public void testNonHumanReadableServerLogsNotSaved() throws Exception {
    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            Options.getDefaults(RemoteOptions.class),
            localRunner,
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
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
    verify(cache, never())
        .downloadFile(
            any(Path.class), any(Digest.class), any(Boolean.class), any(ByteString.class));
  }

  @Test
  public void testServerLogsNotSavedForSuccessfulAction() throws Exception {
    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            Options.getDefaults(RemoteOptions.class),
            localRunner,
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
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
    verify(cache, never())
        .downloadFile(
            any(Path.class), any(Digest.class), any(Boolean.class), any(ByteString.class));
  }

  @Test
  public void cacheDownloadFailureTriggersRemoteExecution() throws Exception {
    // If downloading a cached action fails, remote execution should be tried.

    RemoteOptions options = Options.getDefaults(RemoteOptions.class);

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            options,
            localRunner,
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
            digestUtil,
            logDir);

    ActionResult cachedResult = ActionResult.newBuilder().setExitCode(0).build();
    when(cache.getCachedActionResult(any(ActionKey.class))).thenReturn(cachedResult);
    doThrow(CacheNotFoundException.class)
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

    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteLocalFallback = false;

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            options,
            localRunner,
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
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

    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteLocalFallback = true;

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            options,
            localRunner,
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
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
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteLocalFallback = true;

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            options,
            localRunner,
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
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

    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteLocalFallback = false;

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            options,
            localRunner,
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
            digestUtil,
            logDir);

    when(cache.getCachedActionResult(any(ActionKey.class))).thenReturn(null);
    when(executor.executeRemotely(any(ExecuteRequest.class))).thenThrow(new IOException());

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = new FakeSpawnExecutionContext(spawn);

    try {
      runner.exec(spawn, policy);
      fail("Exception expected");
    } catch (SpawnExecException e) {
      assertThat(e.getSpawnResult().exitCode())
          .isEqualTo(ExitCode.REMOTE_ERROR.getNumericExitCode());
    }
  }

  @Test
  public void testExitCode_executionfailure() throws Exception {
    // If we get a failure due to the remote executor not working, the exit code should be
    // ExitCode.REMOTE_ERROR.

    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    options.remoteLocalFallback = false;

    RemoteSpawnRunner runner =
        new RemoteSpawnRunner(
            execRoot,
            options,
            localRunner,
            true,
            /*cmdlineReporter=*/ null,
            "build-req-id",
            "command-id",
            cache,
            executor,
            digestUtil,
            logDir);

    when(cache.getCachedActionResult(any(ActionKey.class))).thenThrow(new IOException());

    Spawn spawn = newSimpleSpawn();
    SpawnExecutionContext policy = new FakeSpawnExecutionContext(spawn);

    try {
      runner.exec(spawn, policy);
      fail("Exception expected");
    } catch (SpawnExecException e) {
      assertThat(e.getSpawnResult().exitCode())
          .isEqualTo(ExitCode.REMOTE_ERROR.getNumericExitCode());
    }
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
    public ActionInputFileCache getActionInputFileCache() {
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
    public SortedMap<PathFragment, ActionInput> getInputMapping() throws IOException {
      return new SpawnInputExpander(execRoot, /*strict*/ false)
          .getInputMapping(spawn, artifactExpander, fakeFileCache);
    }

    @Override
    public void report(ProgressStatus state, String name) {
      assertThat(state).isEqualTo(ProgressStatus.EXECUTING);
    }
  }
}
