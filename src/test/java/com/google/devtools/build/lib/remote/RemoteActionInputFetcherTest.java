// Copyright 2019 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.FileNode;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import com.google.common.eventbus.EventBus;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher.Priority;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher.Reason;
import com.google.devtools.build.lib.actions.ActionOutputDirectoryHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.remote.common.BulkTransferException;
import com.google.devtools.build.lib.remote.options.RemoteOutputsMode;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.InMemoryCacheClient;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.OutputPermissions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RemoteActionInputFetcher}. */
@RunWith(JUnit4.class)
public class RemoteActionInputFetcherTest extends ActionInputPrefetcherTestBase {
  private static final RemoteOutputChecker DUMMY_REMOTE_OUTPUT_CHECKER =
      new RemoteOutputChecker("build", RemoteOutputsMode.MINIMAL, ImmutableList.of());

  private DigestUtil digestUtil;

  @Override
  public void setUp() throws IOException {
    super.setUp();
    Path dev = fs.getPath("/dev");
    dev.createDirectory();
    dev.setWritable(false);
    digestUtil = new DigestUtil(SyscallCache.NO_CACHE, HASH_FUNCTION);
  }

  @Override
  protected AbstractActionInputPrefetcher createPrefetcher(Map<HashCode, byte[]> cas) {
    CombinedCache combinedCache = newCombinedCache(digestUtil, cas);
    return new RemoteActionInputFetcher(
        new Reporter(eventBus),
        "none",
        "none",
        combinedCache,
        execRoot,
        tempPathGenerator,
        DUMMY_REMOTE_OUTPUT_CHECKER,
        ActionOutputDirectoryHelper.createForTesting(),
        OutputPermissions.READONLY);
  }

  @Test
  public void testStagingVirtualActionInput() throws Exception {
    // arrange
    CombinedCache combinedCache = newCombinedCache(digestUtil, new HashMap<>());
    RemoteActionInputFetcher actionInputFetcher =
        new RemoteActionInputFetcher(
            new Reporter(new EventBus()),
            "none",
            "none",
            combinedCache,
            execRoot,
            tempPathGenerator,
            DUMMY_REMOTE_OUTPUT_CHECKER,
            ActionOutputDirectoryHelper.createForTesting(),
            OutputPermissions.READONLY);
    VirtualActionInput a = ActionsTestUtil.createVirtualActionInput("file1", "hello world");

    // act
    wait(
        actionInputFetcher.prefetchFilesInterruptibly(
            action,
            ImmutableList.of(a),
            (ActionInput unused) -> null,
            Priority.MEDIUM,
            Reason.INPUTS));

    // assert
    Path p = execRoot.getRelative(a.getExecPath());
    assertThat(FileSystemUtils.readContent(p, StandardCharsets.UTF_8)).isEqualTo("hello world");
    assertThat(p.isExecutable()).isTrue();
    assertThat(actionInputFetcher.downloadedFiles()).isEmpty();
    assertThat(actionInputFetcher.downloadsInProgress()).isEmpty();
  }

  @Test
  public void testStagingEmptyVirtualActionInput() throws Exception {
    // arrange
    CombinedCache combinedCache = newCombinedCache(digestUtil, new HashMap<>());
    RemoteActionInputFetcher actionInputFetcher =
        new RemoteActionInputFetcher(
            new Reporter(new EventBus()),
            "none",
            "none",
            combinedCache,
            execRoot,
            tempPathGenerator,
            DUMMY_REMOTE_OUTPUT_CHECKER,
            ActionOutputDirectoryHelper.createForTesting(),
            OutputPermissions.READONLY);

    // act
    wait(
        actionInputFetcher.prefetchFilesInterruptibly(
            action,
            ImmutableList.of(VirtualActionInput.EMPTY_MARKER),
            (ActionInput unused) -> null,
            Priority.MEDIUM,
            Reason.INPUTS));

    // assert that nothing happened
    assertThat(actionInputFetcher.downloadedFiles()).isEmpty();
    assertThat(actionInputFetcher.downloadsInProgress()).isEmpty();
  }

  @Test
  public void prefetchFiles_missingFiles_failsWithSpecificMessage() throws Exception {
    Map<ActionInput, FileArtifactValue> metadata = new HashMap<>();
    Artifact a = createRemoteArtifact("file1", "hello world", metadata, /* cas= */ new HashMap<>());
    AbstractActionInputPrefetcher prefetcher = createPrefetcher(new HashMap<>());

    var error =
        assertThrows(
            BulkTransferException.class,
            () ->
                wait(
                    prefetcher.prefetchFilesInterruptibly(
                        action,
                        ImmutableList.of(a),
                        metadata::get,
                        Priority.MEDIUM,
                        Reason.INPUTS)));

    assertThat(prefetcher.downloadedFiles()).isEmpty();
    assertThat(prefetcher.downloadsInProgress()).isEmpty();
    var m = metadata.get(a);
    var digest = DigestUtil.buildDigest(m.getDigest(), m.getSize());
    assertThat(error)
        .hasMessageThat()
        .contains(String.format("%s/%s", digest.getHash(), digest.getSizeBytes()));
  }

  @Test
  public void injectRemoteRepo_invalidPath_throwsIOException() {
    // Tests that RemoteExternalOverlayFileSystem and RemoteActionInputFetcher
    // maintain path containment within the repository directory.
    PathFragment externalDir = PathFragment.create("/output_base/external");
    InMemoryFileSystem hostFs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    RemoteExternalOverlayFileSystem overlayFs =
        new RemoteExternalOverlayFileSystem(externalDir, hostFs);
    overlayFs.beforeCommand(
        newCombinedCache(digestUtil, new HashMap<>()),
        /* inputPrefetcher= */ null,
        new Reporter(new EventBus()),
        "none",
        "none",
        /* evaluator= */ null,
        /* remoteCacheTtl= */ Duration.ofHours(1));
    for (String invalidPath :
        ImmutableList.of("../../../../../.bashrc.bzl", "/etc/cron.d/evil.bzl")) {
      Tree tree =
          Tree.newBuilder()
              .setRoot(
                  Directory.newBuilder()
                      .addFiles(
                          FileNode.newBuilder()
                              .setName(invalidPath)
                              .setDigest(
                                  digestUtil.compute("payload\n".getBytes(StandardCharsets.UTF_8)))
                              .build())
                      .build())
              .build();
      assertThrows(
          IOException.class,
          () ->
              overlayFs.injectRemoteRepo(
                  RepositoryName.createUnvalidated("repo"), tree, "MARKER\n"));
    }
  }

  @Test
  public void rewoundActionOutput_execRootOnOverlayFileSystem_redownloaded() throws Exception {
    Map<ActionInput, FileArtifactValue> metadata = new HashMap<>();
    Map<HashCode, byte[]> cas = new HashMap<>();
    Artifact a = createRemoteArtifact("file", "hello world", metadata, cas);
    // When the remote repo contents cache is enabled, the exec root lies on the file system
    // overlaying the host file system that downloads are written to.
    RemoteExternalOverlayFileSystem overlayFs =
        new RemoteExternalOverlayFileSystem(PathFragment.create("/output_base/external"), fs);
    RemoteActionInputFetcher actionInputFetcher =
        new RemoteActionInputFetcher(
            new Reporter(eventBus),
            "none",
            "none",
            newCombinedCache(digestUtil, cas),
            overlayFs.getPath(execRoot.getPathString()),
            tempPathGenerator,
            DUMMY_REMOTE_OUTPUT_CHECKER,
            ActionOutputDirectoryHelper.createForTesting(),
            OutputPermissions.READONLY);

    wait(
        actionInputFetcher.prefetchFilesInterruptibly(
            action, metadata.keySet(), metadata::get, Priority.MEDIUM, Reason.INPUTS));
    assertThat(FileSystemUtils.readContent(a.getPath(), StandardCharsets.UTF_8))
        .isEqualTo("hello world");

    // Rewinding deletes the output and requires it to be downloaded again.
    a.getPath().delete();
    actionInputFetcher.handleRewoundActionOutputs(ImmutableList.of(a));

    wait(
        actionInputFetcher.prefetchFilesInterruptibly(
            action, metadata.keySet(), metadata::get, Priority.MEDIUM, Reason.INPUTS));
    assertThat(FileSystemUtils.readContent(a.getPath(), StandardCharsets.UTF_8))
        .isEqualTo("hello world");
  }

  private CombinedCache newCombinedCache(DigestUtil digestUtil, Map<HashCode, byte[]> cas) {
    Map<Digest, byte[]> cacheEntries = Maps.newHashMapWithExpectedSize(cas.size());
    for (Map.Entry<HashCode, byte[]> entry : cas.entrySet()) {
      cacheEntries.put(
          DigestUtil.buildDigest(entry.getKey().asBytes(), entry.getValue().length),
          entry.getValue());
    }
    return new CombinedCache(
        new InMemoryCacheClient(cacheEntries),
        /* diskCacheClient= */ null,
        /* symlinkTemplate= */ null,
        digestUtil,
        /* chunkingFunction= */ null);
  }
}
