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
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyCollection;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.OutputFile;
import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SimpleSpawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.actions.cache.MetadataInjector;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.exec.SpawnCache.CacheHandle;
import com.google.devtools.build.lib.exec.SpawnInputExpander;
import com.google.devtools.build.lib.exec.SpawnRunner.ProgressStatus;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.exec.util.FakeOwner;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.common.RemoteCacheClient.ActionKey;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.options.RemoteOutputsMode;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;
import org.mockito.invocation.InvocationOnMock;
import org.mockito.stubbing.Answer;

/** Tests for {@link RemoteSpawnCache}. */
@RunWith(JUnit4.class)
public class RemoteSpawnCacheTest {
  private static final ArtifactExpander SIMPLE_ARTIFACT_EXPANDER =
      new ArtifactExpander() {
        @Override
        public void expand(Artifact artifact, Collection<? super Artifact> output) {
          output.add(artifact);
        }
      };

  private FileSystem fs;
  private DigestUtil digestUtil;
  private Path execRoot;
  private SimpleSpawn simpleSpawn;
  private FakeActionInputFileCache fakeFileCache;
  @Mock private RemoteCache remoteCache;
  private RemoteSpawnCache cache;
  private FileOutErr outErr;
  private final List<Pair<ProgressStatus, String>> progressUpdates = new ArrayList<>();

  private StoredEventHandler eventHandler = new StoredEventHandler();

  private Reporter reporter;

  private final SpawnExecutionContext simplePolicy =
      new SpawnExecutionContext() {
        @Override
        public int getId() {
          return 0;
        }

        @Override
        public void prefetchInputs() {}

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
              .getInputMapping(
                  simpleSpawn,
                  SIMPLE_ARTIFACT_EXPANDER,
                  ArtifactPathResolver.IDENTITY,
                  fakeFileCache,
                  true);
        }

        @Override
        public void report(ProgressStatus state, String name) {
          progressUpdates.add(Pair.of(state, name));
        }

        @Override
        public MetadataInjector getMetadataInjector() {
          return new MetadataInjector() {
            @Override
            public void injectRemoteFile(
                Artifact output, byte[] digest, long size, int locationIndex, String actionId) {
              throw new UnsupportedOperationException();
            }

            @Override
            public void injectRemoteDirectory(
                Artifact.SpecialArtifact output,
                Map<TreeFileArtifact, RemoteFileArtifactValue> children) {
              throw new UnsupportedOperationException();
            }

            @Override
            public void markOmitted(ActionInput output) {
              throw new UnsupportedOperationException();
            }

            @Override
            public void injectDigest(ActionInput output, FileStatus statNoFollow, byte[] digest) {
              throw new UnsupportedOperationException();
            }
          };
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
      };

  private static SimpleSpawn simpleSpawnWithExecutionInfo(
      ImmutableMap<String, String> executionInfo) {
    return new SimpleSpawn(
        new FakeOwner("Mnemonic", "Progress Message"),
        ImmutableList.of("/bin/echo", "Hi!"),
        ImmutableMap.of("VARIABLE", "value"),
        executionInfo,
        /* inputs= */ NestedSetBuilder.create(
            Order.STABLE_ORDER, ActionInputHelper.fromPath("input")),
        /* outputs= */ ImmutableSet.of(ActionInputHelper.fromPath("/random/file")),
        ResourceSet.ZERO);
  }

  private RemoteSpawnCache remoteSpawnCacheWithOptions(RemoteOptions options) {
    return new RemoteSpawnCache(
        execRoot,
        options,
        remoteCache,
        "build-req-id",
        "command-id",
        reporter,
        digestUtil,
        /* filesToDownload= */ ImmutableSet.of());
  }

  @Before
  public final void setUp() throws Exception {
    MockitoAnnotations.initMocks(this);
    fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);
    digestUtil = new DigestUtil(DigestHashFunction.SHA256);
    execRoot = fs.getPath("/exec/root");
    FileSystemUtils.createDirectoryAndParents(execRoot);
    fakeFileCache = new FakeActionInputFileCache(execRoot);
    simpleSpawn = simpleSpawnWithExecutionInfo(ImmutableMap.of());

    Path stdout = fs.getPath("/tmp/stdout");
    Path stderr = fs.getPath("/tmp/stderr");
    FileSystemUtils.createDirectoryAndParents(stdout.getParentDirectory());
    FileSystemUtils.createDirectoryAndParents(stderr.getParentDirectory());
    outErr = new FileOutErr(stdout, stderr);
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    reporter = new Reporter(new EventBus());
    eventHandler = new StoredEventHandler();
    reporter.addHandler(eventHandler);
    cache = remoteSpawnCacheWithOptions(options);

    fakeFileCache.createScratchInput(simpleSpawn.getInputFiles().getSingleton(), "xyz");
  }

  @SuppressWarnings("unchecked")
  @Test
  public void cacheHit() throws Exception {
    ActionResult actionResult = ActionResult.getDefaultInstance();
    when(remoteCache.downloadActionResult(any(ActionKey.class), /* inlineOutErr= */ eq(false)))
        .thenAnswer(
            new Answer<ActionResult>() {
              @Override
              public ActionResult answer(InvocationOnMock invocation) {
                RequestMetadata meta = TracingMetadataUtils.fromCurrentContext();
                assertThat(meta.getCorrelatedInvocationsId()).isEqualTo("build-req-id");
                assertThat(meta.getToolInvocationId()).isEqualTo("command-id");
                return actionResult;
              }
            });
    Mockito.doAnswer(
            new Answer<Void>() {
              @Override
              public Void answer(InvocationOnMock invocation) {
                RequestMetadata meta = TracingMetadataUtils.fromCurrentContext();
                assertThat(meta.getCorrelatedInvocationsId()).isEqualTo("build-req-id");
                assertThat(meta.getToolInvocationId()).isEqualTo("command-id");
                return null;
              }
            })
        .when(remoteCache)
        .download(eq(actionResult), eq(execRoot), eq(outErr), any());

    CacheHandle entry = cache.lookup(simpleSpawn, simplePolicy);
    assertThat(entry.hasResult()).isTrue();
    SpawnResult result = entry.getResult();
    // All other methods on RemoteActionCache have side effects, so we verify all of them.
    verify(remoteCache).download(eq(actionResult), eq(execRoot), eq(outErr), any());
    verify(remoteCache, never())
        .upload(
            any(ActionKey.class),
            any(Action.class),
            any(Command.class),
            any(Path.class),
            any(Collection.class),
            any(FileOutErr.class));
    assertThat(result.setupSuccess()).isTrue();
    assertThat(result.exitCode()).isEqualTo(0);
    assertThat(result.isCacheHit()).isTrue();
    // We expect the CachedLocalSpawnRunner to _not_ write to outErr at all.
    assertThat(outErr.hasRecordedOutput()).isFalse();
    assertThat(outErr.hasRecordedStderr()).isFalse();
    assertThat(progressUpdates)
        .containsExactly(Pair.of(ProgressStatus.CHECKING_CACHE, "remote-cache"));
  }

  @Test
  public void cacheMiss() throws Exception {
    CacheHandle entry = cache.lookup(simpleSpawn, simplePolicy);
    assertThat(entry.hasResult()).isFalse();
    SpawnResult result =
        new SpawnResult.Builder()
            .setExitCode(0)
            .setStatus(Status.SUCCESS)
            .setRunnerName("test")
            .build();
    ImmutableList<Path> outputFiles = ImmutableList.of(fs.getPath("/random/file"));
    Mockito.doAnswer(
            new Answer<Void>() {
              @Override
              public Void answer(InvocationOnMock invocation) {
                RequestMetadata meta = TracingMetadataUtils.fromCurrentContext();
                assertThat(meta.getCorrelatedInvocationsId()).isEqualTo("build-req-id");
                assertThat(meta.getToolInvocationId()).isEqualTo("command-id");
                return null;
              }
            })
        .when(remoteCache)
        .upload(
            any(ActionKey.class),
            any(Action.class),
            any(Command.class),
            any(Path.class),
            eq(outputFiles),
            eq(outErr));
    entry.store(result);
    verify(remoteCache)
        .upload(
            any(ActionKey.class),
            any(Action.class),
            any(Command.class),
            any(Path.class),
            eq(outputFiles),
            eq(outErr));
    assertThat(progressUpdates)
        .containsExactly(Pair.of(ProgressStatus.CHECKING_CACHE, "remote-cache"));
  }

  @Test
  public void noCacheSpawns() throws Exception {
    // Checks that spawns satisfying Spawns.mayBeCached=false are not looked up in the cache
    // (even if it is a local cache) and that the results/artifacts are not uploaded to the cache.

    RemoteOptions withLocalCache = Options.getDefaults(RemoteOptions.class);
    withLocalCache.diskCache = PathFragment.create("/etc/something/cache/here");
    for (RemoteSpawnCache remoteSpawnCache :
        ImmutableList.of(cache, remoteSpawnCacheWithOptions(withLocalCache))) {
      for (String requirement :
          ImmutableList.of(ExecutionRequirements.NO_CACHE, ExecutionRequirements.LOCAL)) {
        SimpleSpawn uncacheableSpawn =
            simpleSpawnWithExecutionInfo(ImmutableMap.of(requirement, ""));
        CacheHandle entry = remoteSpawnCache.lookup(uncacheableSpawn, simplePolicy);
        verify(remoteCache, never())
            .downloadActionResult(any(ActionKey.class), /* inlineOutErr= */ eq(false));
        assertThat(entry.hasResult()).isFalse();
        SpawnResult result =
            new SpawnResult.Builder()
                .setExitCode(0)
                .setStatus(Status.SUCCESS)
                .setRunnerName("test")
                .build();
        entry.store(result);
        verifyNoMoreInteractions(remoteCache);
        assertThat(progressUpdates).isEmpty();
      }
    }
  }

  @Test
  public void noRemoteCacheSpawns_remoteCache() throws Exception {
    // Checks that spawns satisfying Spawns.mayBeCachedRemotely=false are not looked up in the
    // remote cache, and that the results/artifacts are not uploaded to the remote cache.

    RemoteOptions remoteCacheOptions = Options.getDefaults(RemoteOptions.class);
    remoteCacheOptions.remoteCache = "https://somecache.com";
    RemoteSpawnCache remoteSpawnCache = remoteSpawnCacheWithOptions(remoteCacheOptions);

    for (String requirement :
        ImmutableList.of(
            ExecutionRequirements.NO_CACHE,
            ExecutionRequirements.LOCAL,
            ExecutionRequirements.NO_REMOTE_CACHE,
            ExecutionRequirements.NO_REMOTE)) {
      SimpleSpawn uncacheableSpawn = simpleSpawnWithExecutionInfo(ImmutableMap.of(requirement, ""));
      CacheHandle entry = remoteSpawnCache.lookup(uncacheableSpawn, simplePolicy);
      verify(remoteCache, never())
          .downloadActionResult(any(ActionKey.class), /* inlineOutErr= */ eq(false));
      assertThat(entry.hasResult()).isFalse();
      SpawnResult result =
          new SpawnResult.Builder()
              .setExitCode(0)
              .setStatus(Status.SUCCESS)
              .setRunnerName("test")
              .build();
      entry.store(result);
      verifyNoMoreInteractions(remoteCache);
      assertThat(progressUpdates).isEmpty();
    }
  }

  @Test
  public void noRemoteCacheSpawns_combinedCache() throws Exception {
    // Checks that spawns satisfying Spawns.mayBeCachedRemotely=false are not looked up in the
    // remote cache, and that the results/artifacts are not uploaded to the remote cache.
    // For the purposes of execution requirements, a combined cache with a remote component
    // is considered a remote cache
    RemoteOptions combinedCacheOptions = Options.getDefaults(RemoteOptions.class);
    combinedCacheOptions.remoteCache = "https://somecache.com";
    combinedCacheOptions.diskCache = PathFragment.create("/etc/something/cache/here");
    RemoteSpawnCache remoteSpawnCache = remoteSpawnCacheWithOptions(combinedCacheOptions);

    for (String requirement :
        ImmutableList.of(
            ExecutionRequirements.NO_CACHE,
            ExecutionRequirements.LOCAL,
            ExecutionRequirements.NO_REMOTE_CACHE,
            ExecutionRequirements.NO_REMOTE)) {
      SimpleSpawn uncacheableSpawn = simpleSpawnWithExecutionInfo(ImmutableMap.of(requirement, ""));
      CacheHandle entry = remoteSpawnCache.lookup(uncacheableSpawn, simplePolicy);
      verify(remoteCache, never())
          .downloadActionResult(any(ActionKey.class), /* inlineOutErr= */ eq(false));
      assertThat(entry.hasResult()).isFalse();
      SpawnResult result =
          new SpawnResult.Builder()
              .setExitCode(0)
              .setStatus(Status.SUCCESS)
              .setRunnerName("test")
              .build();
      entry.store(result);
      verifyNoMoreInteractions(remoteCache);
      assertThat(progressUpdates).isEmpty();
    }
  }

  @Test
  public void noRemoteCacheStillUsesLocalCache() throws Exception {
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    remoteOptions.diskCache = PathFragment.create("/etc/something/cache/here");
    cache = remoteSpawnCacheWithOptions(remoteOptions);

    SimpleSpawn cacheableSpawn =
        simpleSpawnWithExecutionInfo(ImmutableMap.of(ExecutionRequirements.NO_REMOTE_CACHE, ""));
    cache.lookup(cacheableSpawn, simplePolicy);
    verify(remoteCache).downloadActionResult(any(ActionKey.class), /* inlineOutErr= */ eq(false));
  }

  @Test
  public void noRemoteExecStillUsesCache() throws Exception {
    SimpleSpawn cacheableSpawn =
        simpleSpawnWithExecutionInfo(ImmutableMap.of(ExecutionRequirements.NO_REMOTE_EXEC, ""));
    cache.lookup(cacheableSpawn, simplePolicy);
    verify(remoteCache).downloadActionResult(any(ActionKey.class), /* inlineOutErr= */ eq(false));
  }

  @Test
  public void failedActionsAreNotUploaded() throws Exception {
    // Only successful action results are uploaded to the remote cache.
    CacheHandle entry = cache.lookup(simpleSpawn, simplePolicy);
    verify(remoteCache).downloadActionResult(any(ActionKey.class), /* inlineOutErr= */ eq(false));
    assertThat(entry.hasResult()).isFalse();
    SpawnResult result =
        new SpawnResult.Builder()
            .setExitCode(1)
            .setStatus(Status.NON_ZERO_EXIT)
            .setRunnerName("test")
            .build();
    ImmutableList<Path> outputFiles = ImmutableList.of(fs.getPath("/random/file"));
    entry.store(result);
    verify(remoteCache, never())
        .upload(
            any(ActionKey.class),
            any(Action.class),
            any(Command.class),
            any(Path.class),
            eq(outputFiles),
            eq(outErr));
    assertThat(progressUpdates)
        .containsExactly(Pair.of(ProgressStatus.CHECKING_CACHE, "remote-cache"));
  }

  @Test
  public void printWarningIfUploadFails() throws Exception {
    CacheHandle entry = cache.lookup(simpleSpawn, simplePolicy);
    assertThat(entry.hasResult()).isFalse();
    SpawnResult result =
        new SpawnResult.Builder()
            .setExitCode(0)
            .setStatus(Status.SUCCESS)
            .setRunnerName("test")
            .build();
    ImmutableList<Path> outputFiles = ImmutableList.of(fs.getPath("/random/file"));

    doThrow(new IOException("cache down"))
        .when(remoteCache)
        .upload(
            any(ActionKey.class),
            any(Action.class),
            any(Command.class),
            any(Path.class),
            eq(outputFiles),
            eq(outErr));

    entry.store(result);
    verify(remoteCache)
        .upload(
            any(ActionKey.class),
            any(Action.class),
            any(Command.class),
            any(Path.class),
            eq(outputFiles),
            eq(outErr));

    assertThat(eventHandler.getEvents()).hasSize(1);
    Event evt = eventHandler.getEvents().get(0);
    assertThat(evt.getKind()).isEqualTo(EventKind.WARNING);
    assertThat(evt.getMessage()).contains("cache down");
    assertThat(progressUpdates)
        .containsExactly(Pair.of(ProgressStatus.CHECKING_CACHE, "remote-cache"));
  }

  @Test
  public void printWarningIfDownloadFails() throws Exception {
    doThrow(new IOException(io.grpc.Status.UNAVAILABLE.asRuntimeException()))
        .when(remoteCache)
        .downloadActionResult(any(ActionKey.class), /* inlineOutErr= */ eq(false));

    CacheHandle entry = cache.lookup(simpleSpawn, simplePolicy);
    assertThat(entry.hasResult()).isFalse();
    SpawnResult result =
        new SpawnResult.Builder()
            .setExitCode(0)
            .setStatus(Status.SUCCESS)
            .setRunnerName("test")
            .build();
    ImmutableList<Path> outputFiles = ImmutableList.of(fs.getPath("/random/file"));

    Mockito.doAnswer(
            new Answer<Void>() {
              @Override
              public Void answer(InvocationOnMock invocation) {
                RequestMetadata meta = TracingMetadataUtils.fromCurrentContext();
                assertThat(meta.getCorrelatedInvocationsId()).isEqualTo("build-req-id");
                assertThat(meta.getToolInvocationId()).isEqualTo("command-id");
                return null;
              }
            })
        .when(remoteCache)
        .upload(
            any(ActionKey.class),
            any(Action.class),
            any(Command.class),
            any(Path.class),
            eq(outputFiles),
            eq(outErr));
    entry.store(result);
    verify(remoteCache)
        .upload(
            any(ActionKey.class),
            any(Action.class),
            any(Command.class),
            any(Path.class),
            eq(outputFiles),
            eq(outErr));

    assertThat(eventHandler.getEvents()).hasSize(1);
    Event evt = eventHandler.getEvents().get(0);
    assertThat(evt.getKind()).isEqualTo(EventKind.WARNING);
    assertThat(evt.getMessage()).contains("UNAVAILABLE");
    assertThat(progressUpdates)
        .containsExactly(Pair.of(ProgressStatus.CHECKING_CACHE, "remote-cache"));
  }

  @Test
  public void orphanedCachedResultIgnored() throws Exception {
    Digest digest = digestUtil.computeAsUtf8("bla");
    ActionResult actionResult =
        ActionResult.newBuilder()
            .addOutputFiles(OutputFile.newBuilder().setPath("/random/file").setDigest(digest))
            .build();
    when(remoteCache.downloadActionResult(any(ActionKey.class), /* inlineOutErr= */ eq(false)))
        .thenAnswer(
            new Answer<ActionResult>() {
              @Override
              public ActionResult answer(InvocationOnMock invocation) {
                RequestMetadata meta = TracingMetadataUtils.fromCurrentContext();
                assertThat(meta.getCorrelatedInvocationsId()).isEqualTo("build-req-id");
                assertThat(meta.getToolInvocationId()).isEqualTo("command-id");
                return actionResult;
              }
            });
    doThrow(new CacheNotFoundException(digest))
        .when(remoteCache)
        .download(eq(actionResult), eq(execRoot), eq(outErr), any());

    CacheHandle entry = cache.lookup(simpleSpawn, simplePolicy);
    assertThat(entry.hasResult()).isFalse();
    SpawnResult result =
        new SpawnResult.Builder()
            .setExitCode(0)
            .setStatus(Status.SUCCESS)
            .setRunnerName("test")
            .build();
    ImmutableList<Path> outputFiles = ImmutableList.of(fs.getPath("/random/file"));

    Mockito.doAnswer(
            new Answer<Void>() {
              @Override
              public Void answer(InvocationOnMock invocation) {
                RequestMetadata meta = TracingMetadataUtils.fromCurrentContext();
                assertThat(meta.getCorrelatedInvocationsId()).isEqualTo("build-req-id");
                assertThat(meta.getToolInvocationId()).isEqualTo("command-id");
                return null;
              }
            })
        .when(remoteCache)
        .upload(
            any(ActionKey.class),
            any(Action.class),
            any(Command.class),
            any(Path.class),
            eq(outputFiles),
            eq(outErr));
    entry.store(result);
    verify(remoteCache)
        .upload(
            any(ActionKey.class),
            any(Action.class),
            any(Command.class),
            any(Path.class),
            eq(outputFiles),
            eq(outErr));
    assertThat(progressUpdates)
        .containsExactly(Pair.of(ProgressStatus.CHECKING_CACHE, "remote-cache"));
    assertThat(eventHandler.getEvents()).isEmpty(); // no warning is printed.
  }

  @Test
  public void failedCacheActionAsCacheMiss() throws Exception {
    ActionResult actionResult = ActionResult.newBuilder().setExitCode(1).build();
    when(remoteCache.downloadActionResult(any(ActionKey.class), /* inlineOutErr= */ eq(false)))
        .thenReturn(actionResult);

    CacheHandle entry = cache.lookup(simpleSpawn, simplePolicy);

    assertThat(entry.hasResult()).isFalse();
  }

  @Test
  public void testDownloadMinimal() throws Exception {
    // arrange
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    remoteOptions.remoteOutputsMode = RemoteOutputsMode.MINIMAL;
    cache = remoteSpawnCacheWithOptions(remoteOptions);

    ActionResult success = ActionResult.newBuilder().setExitCode(0).build();
    when(remoteCache.downloadActionResult(any(), /* inlineOutErr= */ eq(false)))
        .thenReturn(success);

    // act
    CacheHandle cacheHandle = cache.lookup(simpleSpawn, simplePolicy);

    // assert
    assertThat(cacheHandle.hasResult()).isTrue();
    assertThat(cacheHandle.getResult().exitCode()).isEqualTo(0);
    verify(remoteCache)
        .downloadMinimal(any(), any(), anyCollection(), any(), any(), any(), any(), any());
  }

  @Test
  public void testDownloadMinimalIoError() throws Exception {
    // arrange
    RemoteOptions remoteOptions = Options.getDefaults(RemoteOptions.class);
    remoteOptions.remoteOutputsMode = RemoteOutputsMode.MINIMAL;
    cache = remoteSpawnCacheWithOptions(remoteOptions);

    IOException downloadFailure = new IOException("downloadMinimal failed");

    ActionResult success = ActionResult.newBuilder().setExitCode(0).build();
    when(remoteCache.downloadActionResult(any(), /* inlineOutErr= */ eq(false)))
        .thenReturn(success);
    when(remoteCache.downloadMinimal(
            any(), any(), anyCollection(), any(), any(), any(), any(), any()))
        .thenThrow(downloadFailure);

    // act
    CacheHandle cacheHandle = cache.lookup(simpleSpawn, simplePolicy);

    // assert
    assertThat(cacheHandle.hasResult()).isFalse();
    verify(remoteCache)
        .downloadMinimal(any(), any(), anyCollection(), any(), any(), any(), any(), any());
    assertThat(eventHandler.getEvents().size()).isEqualTo(1);
    Event evt = eventHandler.getEvents().get(0);
    assertThat(evt.getKind()).isEqualTo(EventKind.WARNING);
    assertThat(evt.getMessage()).contains(downloadFailure.getMessage());
  }
}
