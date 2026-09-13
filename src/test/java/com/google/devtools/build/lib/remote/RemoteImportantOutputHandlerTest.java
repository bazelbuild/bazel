// Copyright 2025 The Bazel Authors. All rights reserved.
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
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.RunfilesTree;
import com.google.devtools.build.lib.actions.StaticInputMetadataProvider;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.remote.options.RemoteOutputsMode;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.OutputService.RewoundActionSynchronizer;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RewindableRepoFileSystem;
import com.google.devtools.build.lib.vfs.RewindingSynchronizer;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests synchronization of top-level downloads with repository replacement. */
@RunWith(JUnit4.class)
public final class RemoteImportantOutputHandlerTest {
  private enum Download {
    SOURCE,
    RUNFILE,
    SYMLINK_TO_SOURCE
  }

  @Test
  public void sourceDownload_blocksRepoReplacement() throws Exception {
    checkDownloadLocks(Download.SOURCE);
  }

  @Test
  public void runfileDownload_blocksRepoReplacement() throws Exception {
    checkDownloadLocks(Download.RUNFILE);
  }

  @Test
  public void symlinkToSourceDownload_blocksRepoReplacement() throws Exception {
    checkDownloadLocks(Download.SYMLINK_TO_SOURCE);
  }

  private void checkDownloadLocks(Download download) throws Exception {
    var fs = new RepoFileSystem();
    var sourceRoot =
        ArtifactRoot.asExternalSourceRoot(Root.fromPath(fs.getPath("/output/external/repo")));
    Artifact source =
        new Artifact.SourceArtifact(
            sourceRoot,
            PathFragment.create("external/repo/input.txt"),
            () -> Label.parseCanonicalUnchecked("@@repo//:input.txt"));
    FileArtifactValue sourceMetadata = FileArtifactValue.createForRemoteFile(new byte[] {1}, 1, 1);
    var graph = mock(WalkableGraph.class);
    Artifact artifact;
    FileArtifactValue metadata;
    if (download == Download.SYMLINK_TO_SOURCE) {
      // The output of a symlink action whose target is the source file (see SymlinkAction): its
      // contents are downloaded to the resolved path, which lies in the repository.
      DerivedArtifact link =
          (DerivedArtifact)
              ActionsTestUtil.createArtifact(
                  ArtifactRoot.asDerivedRoot(
                      fs.getPath("/output/execroot/ws"), RootType.OUTPUT, "bazel-out"),
                  "link.txt");
      link.setGeneratingActionKey(ActionLookupData.create(ActionsTestUtil.NULL_ARTIFACT_OWNER, 0));
      ActionLookupValue ownerValue = mock(ActionLookupValue.class);
      when(ownerValue.getActions()).thenReturn(ImmutableList.of(mock(Action.class)));
      when(graph.getValue(ActionsTestUtil.NULL_ARTIFACT_OWNER)).thenReturn(ownerValue);
      artifact = link;
      metadata =
          FileArtifactValue.createFromExistingWithResolvedPath(
              sourceMetadata, source.getPath().asFragment());
    } else {
      artifact = source;
      metadata = sourceMetadata;
    }
    var metadataProvider =
        spy(new StaticInputMetadataProvider(ImmutableMap.of(artifact, metadata)));
    if (download == Download.RUNFILE) {
      RunfilesTree tree = mock(RunfilesTree.class);
      when(tree.getArtifacts()).thenReturn(NestedSetBuilder.create(Order.STABLE_ORDER, artifact));
      when(metadataProvider.getRunfilesTrees()).thenReturn(ImmutableList.of(tree));
    }
    var downloadFuture = SettableFuture.<Void>create();
    var scheduled = new CountDownLatch(1);
    var prefetcher = mock(ActionInputPrefetcher.class);
    when(prefetcher.prefetchFiles(any(), any(), any(), any(), any(), any()))
        .thenAnswer(
            invocation -> {
              scheduled.countDown();
              return downloadFuture;
            });
    RewindingSynchronizer synchronizer = fs.getRewindingSynchronizer();
    synchronizer.markReplacementsPossible();
    // Switch to per-key locks so that the consumer has to determine the repository of its download
    // and the producer waits for that repository's lock rather than for the shared one.
    synchronizer.acquireWriteLock("unrelated producer").close();
    var handler =
        new RemoteImportantOutputHandler(
            graph,
            new RemoteOutputChecker("build", RemoteOutputsMode.ALL, ImmutableList.of()),
            prefetcher,
            RewoundActionSynchronizer.NOOP,
            fs);
    ImmutableList<Artifact> outputs =
        download == Download.RUNFILE ? ImmutableList.of() : ImmutableList.of(artifact);
    var consumer =
        new TestThread(
            () ->
                assertThat(
                        handler
                            .processOutputsAndGetLostArtifacts(outputs, metadataProvider)
                            .isEmpty())
                    .isTrue());
    var replaced = new AtomicBoolean();
    var producer =
        new TestThread(
            () -> {
              try (var unused =
                  synchronizer.acquireWriteLock(sourceRoot.getExternalRepositoryName())) {
                replaced.set(true);
              }
            });
    consumer.start();
    try {
      assertThat(scheduled.await(10, TimeUnit.SECONDS)).isTrue();
      producer.start();
      awaitWaiting(producer);
      assertThat(replaced.get()).isFalse();
    } finally {
      downloadFuture.set(null);
    }
    consumer.joinAndAssertState(10_000);
    producer.joinAndAssertState(10_000);
    assertThat(replaced.get()).isTrue();
  }

  private static void awaitWaiting(Thread thread) throws InterruptedException {
    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
    while (thread.getState() != Thread.State.WAITING) {
      assertThat(thread.isAlive()).isTrue();
      assertThat(System.nanoTime()).isLessThan(deadline);
      Thread.sleep(1);
    }
  }

  private static final class RepoFileSystem extends InMemoryFileSystem
      implements RewindableRepoFileSystem {
    private static final PathFragment EXTERNAL_DIR = PathFragment.create("/output/external");

    private final RewindingSynchronizer synchronizer = new RewindingSynchronizer();

    RepoFileSystem() {
      super(DigestHashFunction.SHA256);
    }

    @Override
    public RewindingSynchronizer getRewindingSynchronizer() {
      return synchronizer;
    }

    @Override
    public boolean isRepoPath(PathFragment path) {
      return path.startsWith(EXTERNAL_DIR) && path.segmentCount() > EXTERNAL_DIR.segmentCount();
    }

    @Override
    public RepositoryName repoContaining(PathFragment path) {
      return RepositoryName.createUnvalidated(path.getSegment(EXTERNAL_DIR.segmentCount()));
    }
  }
}
