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
import static org.mockito.Matchers.any;
import static org.mockito.Matchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.OutputFile;
import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SimpleSpawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.exec.SpawnCache.CacheHandle;
import com.google.devtools.build.lib.exec.SpawnInputExpander;
import com.google.devtools.build.lib.exec.SpawnRunner.ProgressStatus;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.exec.util.FakeOwner;
import com.google.devtools.build.lib.remote.Retrier.RetryException;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.DigestUtil.ActionKey;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
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
  @Mock private AbstractRemoteActionCache remoteCache;
  private RemoteSpawnCache cache;
  private FileOutErr outErr;
  private final List<Pair<ProgressStatus, String>> progressUpdates = new ArrayList();

  private StoredEventHandler eventHandler = new StoredEventHandler();

  private final SpawnExecutionContext simplePolicy =
      new SpawnExecutionContext() {
        @Override
        public int getId() {
          return 0;
        }

        @Override
        public void prefetchInputs() {
          // CachedLocalSpawnRunner should never prefetch itself, though the nested SpawnRunner may.
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
              .getInputMapping(simpleSpawn, SIMPLE_ARTIFACT_EXPANDER, ArtifactPathResolver.IDENTITY,
                  fakeFileCache, true);
        }

        @Override
        public void report(ProgressStatus state, String name) {
          progressUpdates.add(Pair.of(state, name));
        }
      };

  @Before
  public final void setUp() throws Exception {
    MockitoAnnotations.initMocks(this);
    fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);
    digestUtil = new DigestUtil(DigestHashFunction.SHA256);
    execRoot = fs.getPath("/exec/root");
    FileSystemUtils.createDirectoryAndParents(execRoot);
    fakeFileCache = new FakeActionInputFileCache(execRoot);
    simpleSpawn =
        new SimpleSpawn(
            new FakeOwner("Mnemonic", "Progress Message"),
            ImmutableList.of("/bin/echo", "Hi!"),
            ImmutableMap.of("VARIABLE", "value"),
            /*executionInfo=*/ ImmutableMap.<String, String>of(),
            /*inputs=*/ ImmutableList.of(ActionInputHelper.fromPath("input")),
            /*outputs=*/ ImmutableList.of(ActionInputHelper.fromPath("/random/file")),
            ResourceSet.ZERO);

    Path stdout = fs.getPath("/tmp/stdout");
    Path stderr = fs.getPath("/tmp/stderr");
    FileSystemUtils.createDirectoryAndParents(stdout.getParentDirectory());
    FileSystemUtils.createDirectoryAndParents(stderr.getParentDirectory());
    outErr = new FileOutErr(stdout, stderr);
    RemoteOptions options = Options.getDefaults(RemoteOptions.class);
    Reporter reporter = new Reporter(new EventBus());
    eventHandler = new StoredEventHandler();
    reporter.addHandler(eventHandler);
    cache =
        new RemoteSpawnCache(
            execRoot,
            options,
            remoteCache,
            "build-req-id",
            "command-id",
            reporter,
            digestUtil);
    fakeFileCache.createScratchInput(simpleSpawn.getInputFiles().get(0), "xyz");
  }

  @SuppressWarnings("unchecked")
  @Test
  public void cacheHit() throws Exception {
    ActionResult actionResult = ActionResult.getDefaultInstance();
    when(remoteCache.getCachedActionResult(any(ActionKey.class)))
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
        .download(actionResult, execRoot, outErr);

    CacheHandle entry = cache.lookup(simpleSpawn, simplePolicy);
    assertThat(entry.hasResult()).isTrue();
    SpawnResult result = entry.getResult();
    // All other methods on RemoteActionCache have side effects, so we verify all of them.
    verify(remoteCache).download(actionResult, execRoot, outErr);
    verify(remoteCache, never())
        .ensureInputsPresent(
            any(TreeNodeRepository.class),
            any(Path.class),
            any(TreeNode.class),
            any(Action.class),
            any(Command.class));
    verify(remoteCache, never())
        .upload(
            any(ActionKey.class),
            any(Action.class),
            any(Command.class),
            any(Path.class),
            any(Collection.class),
            any(FileOutErr.class),
            any(Boolean.class));
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
            eq(outErr),
            eq(true));
    entry.store(result);
    verify(remoteCache)
        .upload(
            any(ActionKey.class),
            any(Action.class),
            any(Command.class),
            any(Path.class),
            eq(outputFiles),
            eq(outErr),
            eq(true));
    assertThat(progressUpdates)
        .containsExactly(Pair.of(ProgressStatus.CHECKING_CACHE, "remote-cache"));
  }

  @Test
  public void noCacheSpawns() throws Exception {
    // Checks that spawns that have mayBeCached false are not looked up in the remote cache,
    // and also that their result is not uploaded to the remote cache. The artifacts, however,
    // are uploaded.
    SimpleSpawn uncacheableSpawn =
        new SimpleSpawn(
            new FakeOwner("foo", "bar"),
            /*arguments=*/ ImmutableList.of(),
            /*environment=*/ ImmutableMap.of(),
            ImmutableMap.of(ExecutionRequirements.NO_CACHE, ""),
            /*inputs=*/ ImmutableList.of(),
            /*outputs=*/ ImmutableList.of(ActionInputHelper.fromPath("/random/file")),
            ResourceSet.ZERO);
    CacheHandle entry = cache.lookup(uncacheableSpawn, simplePolicy);
    verify(remoteCache, never())
        .getCachedActionResult(any(ActionKey.class));
    assertThat(entry.hasResult()).isFalse();
    SpawnResult result =
        new SpawnResult.Builder()
            .setExitCode(0)
            .setStatus(Status.SUCCESS)
            .setRunnerName("test")
            .build();
    entry.store(result);
    ImmutableList<Path> outputFiles = ImmutableList.of(fs.getPath("/random/file"));
    verify(remoteCache)
        .upload(
            any(ActionKey.class),
            any(Action.class),
            any(Command.class),
            any(Path.class),
            eq(outputFiles),
            eq(outErr),
            eq(false));
    assertThat(progressUpdates).containsExactly();
  }

  @Test
  public void noCacheSpawnsNoResultStore() throws Exception {
    // Only successful action results are uploaded to the remote cache. The artifacts, however,
    // are uploaded regardless.
    CacheHandle entry = cache.lookup(simpleSpawn, simplePolicy);
    verify(remoteCache).getCachedActionResult(any(ActionKey.class));
    assertThat(entry.hasResult()).isFalse();
    SpawnResult result =
        new SpawnResult.Builder()
            .setExitCode(1)
            .setStatus(Status.NON_ZERO_EXIT)
            .setRunnerName("test")
            .build();
    ImmutableList<Path> outputFiles = ImmutableList.of(fs.getPath("/random/file"));
    entry.store(result);
    verify(remoteCache)
        .upload(
            any(ActionKey.class),
            any(Action.class),
            any(Command.class),
            any(Path.class),
            eq(outputFiles),
            eq(outErr),
            eq(false));
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
            eq(outErr),
            eq(true));

    entry.store(result);
    verify(remoteCache)
        .upload(
            any(ActionKey.class),
            any(Action.class),
            any(Command.class),
            any(Path.class),
            eq(outputFiles),
            eq(outErr),
            eq(true));

    assertThat(eventHandler.getEvents()).hasSize(1);
    Event evt = eventHandler.getEvents().get(0);
    assertThat(evt.getKind()).isEqualTo(EventKind.WARNING);
    assertThat(evt.getMessage()).contains("Error");
    assertThat(evt.getMessage()).contains("writing");
    assertThat(evt.getMessage()).contains("cache down");
    assertThat(progressUpdates)
        .containsExactly(Pair.of(ProgressStatus.CHECKING_CACHE, "remote-cache"));
  }

  @Test
  public void printWarningIfDownloadFails() throws Exception {
    doThrow(new RetryException("reason", 0, io.grpc.Status.UNAVAILABLE.asRuntimeException()))
        .when(remoteCache)
        .getCachedActionResult(any(ActionKey.class));

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
            eq(outErr),
            eq(true));
    entry.store(result);
    verify(remoteCache)
        .upload(
            any(ActionKey.class),
            any(Action.class),
            any(Command.class),
            any(Path.class),
            eq(outputFiles),
            eq(outErr),
            eq(true));

    assertThat(eventHandler.getEvents()).hasSize(1);
    Event evt = eventHandler.getEvents().get(0);
    assertThat(evt.getKind()).isEqualTo(EventKind.WARNING);
    assertThat(evt.getMessage()).contains("Error");
    assertThat(evt.getMessage()).contains("reading");
    assertThat(evt.getMessage()).contains("reason");
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
    when(remoteCache.getCachedActionResult(any(ActionKey.class)))
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
    doThrow(new RetryException("cache miss", 0, new CacheNotFoundException(digest, digestUtil)))
        .when(remoteCache)
        .download(actionResult, execRoot, outErr);

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
            eq(outErr),
            eq(true));
    entry.store(result);
    verify(remoteCache)
        .upload(
            any(ActionKey.class),
            any(Action.class),
            any(Command.class),
            any(Path.class),
            eq(outputFiles),
            eq(outErr),
            eq(true));
    assertThat(progressUpdates)
        .containsExactly(Pair.of(ProgressStatus.CHECKING_CACHE, "remote-cache"));
    assertThat(eventHandler.getEvents()).isEmpty(); // no warning is printed.
  }
}
