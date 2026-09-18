// Copyright 2026 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.remote.util.Futures.getFromFuture;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.RequestMetadata;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.Futures;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.DiscoveredModulesPruner;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileStatusWithMetadata;
import com.google.devtools.build.lib.actions.ThreadStateReceiver;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil.FakeInputMetadataHandlerBase;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil.NullAction;
import com.google.devtools.build.lib.actions.util.DummyExecutor;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.exec.FileWriteStrategy;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.options.RemoteOutputsMode;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.InMemoryCacheClient;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.util.DeterministicWriter;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.time.Duration;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RemoteFileWriteStrategy}. */
@RunWith(JUnit4.class)
public final class RemoteFileWriteStrategyTest {
  private static final DigestUtil DIGEST_UTIL =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
  private static final String CONTENT = "hello";
  private static final Digest CONTENT_DIGEST = DIGEST_UTIL.compute(CONTENT.getBytes(UTF_8));

  private final FileSystem fileSystem = new InMemoryFileSystem(DigestHashFunction.SHA256);
  private final Scratch scratch = new Scratch(fileSystem);
  private final InMemoryCacheClient cacheClient = spy(new InMemoryCacheClient());
  private final StoredEventHandler eventHandler = new StoredEventHandler();
  private Path execRoot;
  private RemoteActionFileSystem actionFileSystem;
  private AbstractAction action;
  private Artifact output;

  @Before
  public void setUp() throws IOException {
    execRoot = scratch.dir("/execroot");
    var outputRoot = ArtifactRoot.asDerivedRoot(execRoot, RootType.OUTPUT, "bazel-out");
    outputRoot.getRoot().asPath().createDirectory();
    output =
        ActionsTestUtil.createArtifactWithRootRelativePath(
            outputRoot, PathFragment.create("pkg/file"));
    action = new NullAction(output);
    actionFileSystem =
        new RemoteActionFileSystem(
            fileSystem,
            execRoot.asFragment(),
            outputRoot.getExecPathString(),
            new ActionInputMap(0),
            mock(RemoteActionInputFetcher.class));
    actionFileSystem.createDirectoryAndParents(
        output.getPath().getParentDirectory().asFragment());
  }

  @Test
  public void remotable_storesContentsRemotely() throws Exception {
    var strategy = createStrategy(RemoteOutputsMode.MINIMAL, cacheClient);

    var unused =
        strategy.writeOutputToFile(
            action,
            createActionExecutionContext(actionFileSystem),
            writer(CONTENT),
            /* makeExecutable= */ false,
            /* isRemotable= */ true);

    assertStoredRemotely(cacheClient);
    var metadata = getRemoteMetadata();
    assertThat(metadata.isRemote()).isTrue();
    assertThat(metadata.getDigest()).isEqualTo(DigestUtil.toBinaryDigest(CONTENT_DIGEST));
    assertThat(metadata.getSize()).isEqualTo(CONTENT.length());
    assertThat(output.getPath().exists()).isFalse();
    assertThat(eventHandler.getEvents()).isEmpty();
  }

  @Test
  public void alreadyStoredRemotely_skipsUpload() throws Exception {
    var prepopulatedCacheClient =
        spy(new InMemoryCacheClient(ImmutableMap.of(CONTENT_DIGEST, CONTENT.getBytes(UTF_8))));
    var strategy = createStrategy(RemoteOutputsMode.MINIMAL, prepopulatedCacheClient);

    var unused =
        strategy.writeOutputToFile(
            action,
            createActionExecutionContext(actionFileSystem),
            writer(CONTENT),
            /* makeExecutable= */ false,
            /* isRemotable= */ true);

    verify(prepopulatedCacheClient, never()).uploadBlobImpl(any(), any(), any());
    assertThat(getRemoteMetadata().isRemote()).isTrue();
    assertThat(output.getPath().exists()).isFalse();
  }

  @Test
  public void nonRemotable_writesLocally() throws Exception {
    var strategy = createStrategy(RemoteOutputsMode.MINIMAL, cacheClient);

    var unused =
        strategy.writeOutputToFile(
            action,
            createActionExecutionContext(actionFileSystem),
            writer(CONTENT),
            /* makeExecutable= */ false,
            /* isRemotable= */ false);

    assertWrittenLocally();
    assertNotStoredRemotely(cacheClient);
  }

  @Test
  public void outputRequestedForDownload_writesLocally() throws Exception {
    var strategy = createStrategy(RemoteOutputsMode.ALL, cacheClient);

    var unused =
        strategy.writeOutputToFile(
            action,
            createActionExecutionContext(actionFileSystem),
            writer(CONTENT),
            /* makeExecutable= */ false,
            /* isRemotable= */ true);

    assertWrittenLocally();
    assertNotStoredRemotely(cacheClient);
  }

  @Test
  public void uploadDisabled_writesLocally() throws Exception {
    var strategy =
        createStrategy(RemoteOutputsMode.MINIMAL, cacheClient, /* uploadEnabled= */ false);

    var unused =
        strategy.writeOutputToFile(
            action,
            createActionExecutionContext(actionFileSystem),
            writer(CONTENT),
            /* makeExecutable= */ false,
            /* isRemotable= */ true);

    assertWrittenLocally();
    assertNotStoredRemotely(cacheClient);
  }

  @Test
  public void noActionFileSystem_writesLocally() throws Exception {
    var strategy = createStrategy(RemoteOutputsMode.MINIMAL, cacheClient);

    var unused =
        strategy.writeOutputToFile(
            action,
            createActionExecutionContext(/* actionFileSystem= */ null),
            writer(CONTENT),
            /* makeExecutable= */ false,
            /* isRemotable= */ true);

    assertWrittenLocally();
    assertNotStoredRemotely(cacheClient);
  }

  @Test
  public void uploadFails_writesLocallyAndWarns() throws Exception {
    doReturn(Futures.immediateFailedFuture(new IOException("upload failed")))
        .when(cacheClient)
        .uploadBlobImpl(any(), any(), any());
    var strategy = createStrategy(RemoteOutputsMode.MINIMAL, cacheClient);

    var unused =
        strategy.writeOutputToFile(
            action,
            createActionExecutionContext(actionFileSystem),
            writer(CONTENT),
            /* makeExecutable= */ false,
            /* isRemotable= */ true);

    assertWrittenLocally();
    var warning = eventHandler.getEvents().stream().findFirst().orElseThrow();
    assertThat(warning.getKind()).isEqualTo(EventKind.WARNING);
    assertThat(warning.getMessage()).contains("Remote Cache:");
    assertThat(warning.getMessage()).contains("upload failed");
  }

  private RemoteFileWriteStrategy createStrategy(
      RemoteOutputsMode outputsMode, InMemoryCacheClient cacheClient) {
    return createStrategy(outputsMode, cacheClient, /* uploadEnabled= */ true);
  }

  private RemoteFileWriteStrategy createStrategy(
      RemoteOutputsMode outputsMode, InMemoryCacheClient cacheClient, boolean uploadEnabled) {
    return new RemoteFileWriteStrategy(
        new FileWriteStrategy(),
        new CombinedCache(
            cacheClient,
            /* diskCacheClient= */ null,
            /* symlinkTemplate= */ null,
            DIGEST_UTIL,
            /* chunkingFunction= */ null,
            new ChunkLocationMap()),
        new RemoteOutputChecker("build", outputsMode, ImmutableList.of()),
        DIGEST_UTIL,
        "build-request-id",
        "command-id",
        Duration.ofHours(1),
        uploadEnabled,
        /* verboseFailures= */ false);
  }

  private ActionExecutionContext createActionExecutionContext(
      @Nullable FileSystem actionFileSystem) {
    var metadataHandler = new FakeInputMetadataHandlerBase();
    return new ActionExecutionContext(
        new DummyExecutor(fileSystem, execRoot),
        /* inputMetadataProvider= */ metadataHandler,
        ActionInputPrefetcher.NONE,
        new ActionKeyContext(),
        /* outputMetadataStore= */ metadataHandler,
        /* rewindingEnabled= */ false,
        ActionExecutionContext.LostInputsCheck.NONE,
        /* fileOutErr= */ null,
        eventHandler,
        /* clientEnv= */ ImmutableMap.of(),
        actionFileSystem,
        DiscoveredModulesPruner.DEFAULT,
        SyscallCache.NO_CACHE,
        ThreadStateReceiver.NULL_INSTANCE);
  }

  private static DeterministicWriter writer(String content) {
    return out -> out.write(content.getBytes(UTF_8));
  }

  private FileArtifactValue getRemoteMetadata() throws IOException {
    var status = actionFileSystem.getPath(output.getPath().asFragment()).stat();
    assertThat(status).isInstanceOf(FileStatusWithMetadata.class);
    return ((FileStatusWithMetadata) status).getMetadata();
  }

  private void assertWrittenLocally() throws IOException {
    assertThat(FileSystemUtils.readContent(output.getPath(), UTF_8)).isEqualTo(CONTENT);
  }

  private static void assertStoredRemotely(InMemoryCacheClient cacheClient) throws Exception {
    assertThat(
            getFromFuture(
                cacheClient.findMissingDigests(
                    RemoteActionExecutionContext.create(RequestMetadata.getDefaultInstance()),
                    ImmutableList.of(CONTENT_DIGEST))))
        .isEmpty();
  }

  private static void assertNotStoredRemotely(InMemoryCacheClient cacheClient) throws Exception {
    verify(cacheClient, never()).uploadBlobImpl(any(), any(), any());
    assertThat(
            getFromFuture(
                cacheClient.findMissingDigests(
                    RemoteActionExecutionContext.create(RequestMetadata.getDefaultInstance()),
                    ImmutableList.of(CONTENT_DIGEST))))
        .containsExactly(CONTENT_DIGEST);
  }
}
