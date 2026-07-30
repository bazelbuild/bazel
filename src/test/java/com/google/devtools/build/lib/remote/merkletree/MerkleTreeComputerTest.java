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
package com.google.devtools.build.lib.remote.merkletree;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.MoreCollectors.onlyElement;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.DirectoryNode;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.DelegatingPairInputMetadataProvider;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetOutputTree;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.RunfilesArtifactValue;
import com.google.devtools.build.lib.actions.RunfilesTree;
import com.google.devtools.build.lib.actions.SimpleSpawn;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnInputs;
import com.google.devtools.build.lib.actions.VirtualActionInput;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.exec.util.FakeActionInputFileCache;
import com.google.devtools.build.lib.exec.util.FakeOwner;
import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.common.RemotePathResolver;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.FakeSpawnExecutionContext;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class MerkleTreeComputerTest {
  private Path execRoot;
  private ArtifactRoot artifactRoot;

  @Before
  public void setUp() throws IOException {
    var fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);
    execRoot = fs.getPath("/execroot/_main");
    execRoot.createDirectoryAndParents();
    artifactRoot = ArtifactRoot.asDerivedRoot(execRoot, ArtifactRoot.RootType.OUTPUT, "outputs");
    checkNotNull(artifactRoot.getRoot().asPath()).createDirectoryAndParents();
  }

  @Test
  public void testSubtreeComputationCancelled_subsequentReusingCallNotAffected() throws Exception {
    var fakeFileCache = new FakeActionInputFileCache();
    var treeArtifactInput =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(
            artifactRoot, "dir/subdir/tree_artifact");
    treeArtifactInput.getPath().createDirectoryAndParents();
    var treeArtifactBuilder = TreeArtifactValue.newBuilder(treeArtifactInput);
    var treeFileArtifact = Artifact.TreeFileArtifact.createTreeOutput(treeArtifactInput, "file");
    FileSystemUtils.writeContentAsLatin1(treeFileArtifact.getPath(), "file content");
    treeArtifactBuilder.putChild(
        treeFileArtifact, FileArtifactValue.createForTesting(treeFileArtifact));
    fakeFileCache.putTreeArtifact(treeArtifactInput, treeArtifactBuilder.build());
    var spawn = new SpawnBuilder().withInputs(treeArtifactInput).build();
    var merkleTreeComputer = createMerkleTreeComputer(/* uploader= */ null);

    var treeFileMetadataAccessed = new CountDownLatch(1);
    var delayedMetadataProvider =
        new DelegatingPairInputMetadataProvider(
            new InputMetadataProvider() {
              @Nullable
              @Override
              public FileArtifactValue getInputMetadataChecked(ActionInput input)
                  throws InterruptedException {
                if (input != treeFileArtifact || treeFileMetadataAccessed.getCount() == 0) {
                  return null;
                }
                treeFileMetadataAccessed.countDown();
                Thread.sleep(Long.MAX_VALUE);
                throw new IllegalStateException("not reached");
              }

              @Nullable
              @Override
              public TreeArtifactValue getTreeMetadata(ActionInput input) {
                return null;
              }

              @Nullable
              @Override
              public TreeArtifactValue getEnclosingTreeMetadata(PathFragment execPath) {
                return null;
              }

              @Nullable
              @Override
              public FilesetOutputTree getFileset(ActionInput input) {
                return null;
              }

              @Override
              public ImmutableMap<Artifact, FilesetOutputTree> getFilesets() {
                return ImmutableMap.of();
              }

              @Nullable
              @Override
              public RunfilesArtifactValue getRunfilesMetadata(ActionInput input) {
                return null;
              }

              @Nullable
              @Override
              public ImmutableList<RunfilesTree> getRunfilesTrees() {
                return null;
              }

              @Nullable
              @Override
              public ActionInput getInput(PathFragment execPath) {
                return null;
              }
            },
            fakeFileCache);
    var capturedThrowable = new AtomicReference<Throwable>();
    var buildThread =
        new Thread(
            () -> {
              try {
                var unused =
                    merkleTreeComputer.buildForSpawn(
                        spawn,
                        ImmutableSet.of(),
                        /* scrubber= */ null,
                        createSpawnExecutionContext(spawn, delayedMetadataProvider),
                        RemotePathResolver.createDefault(execRoot),
                        MerkleTreeComputer.BlobPolicy.KEEP);
              } catch (Throwable t) {
                if (t instanceof InterruptedException) {
                  Thread.currentThread().interrupt();
                }
                capturedThrowable.set(t);
              }
            });
    buildThread.start();
    // Wait until the Merkle subtree for the tree artifact started to build, then interrupt it.
    treeFileMetadataAccessed.await();
    buildThread.interrupt();
    buildThread.join();
    assertThat(capturedThrowable.get()).isInstanceOf(InterruptedException.class);

    // Expected to succeed despite the subtree computation having been canceled.
    var unused =
        merkleTreeComputer.buildForSpawn(
            spawn,
            ImmutableSet.of(),
            /* scrubber= */ null,
            createSpawnExecutionContext(spawn, fakeFileCache),
            RemotePathResolver.createDefault(execRoot),
            MerkleTreeComputer.BlobPolicy.KEEP);
  }

  @Test
  public void testSubtreeComputationCancelled_concurrentReusingCallNotAffected() throws Throwable {
    var fakeFileCache = new FakeActionInputFileCache();
    var treeArtifactInput =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(
            artifactRoot, "dir/subdir/tree_artifact");
    treeArtifactInput.getPath().createDirectoryAndParents();
    var treeArtifactBuilder = TreeArtifactValue.newBuilder(treeArtifactInput);
    var treeFileArtifact = Artifact.TreeFileArtifact.createTreeOutput(treeArtifactInput, "file");
    FileSystemUtils.writeContentAsLatin1(treeFileArtifact.getPath(), "file content");
    treeArtifactBuilder.putChild(
        treeFileArtifact, FileArtifactValue.createForTesting(treeFileArtifact));
    fakeFileCache.putTreeArtifact(treeArtifactInput, treeArtifactBuilder.build());
    var spawn = new SpawnBuilder().withInputs(treeArtifactInput).build();
    var ensureInputsPresentCount = new AtomicInteger();
    var merkleTreeComputer =
        createMerkleTreeComputer(
            new MerkleTreeUploader() {
              @Override
              public ListenableFuture<Void> uploadBlob(
                  RemoteActionExecutionContext context, Digest digest, byte[] data, boolean force) {
                return immediateVoidFuture();
              }

              @Override
              public ListenableFuture<Void> uploadFile(
                  RemoteActionExecutionContext context,
                  RemotePathResolver remotePathResolver,
                  Digest digest,
                  Path path,
                  boolean force) {
                return immediateVoidFuture();
              }

              @Override
              public ListenableFuture<Void> uploadVirtualActionInput(
                  RemoteActionExecutionContext context,
                  Digest digest,
                  VirtualActionInput virtualActionInput,
                  boolean force) {
                return immediateVoidFuture();
              }

              @Override
              public void ensureInputsPresent(
                  RemoteActionExecutionContext context,
                  MerkleTree.Uploadable merkleTree,
                  boolean force,
                  RemotePathResolver remotePathResolver) {
                ensureInputsPresentCount.incrementAndGet();
              }
            });

    var treeFileMetadataAccessed = new CountDownLatch(1);
    var treeFileMetadataContinue = new CountDownLatch(1);
    var delayedMetadataProvider =
        new DelegatingPairInputMetadataProvider(
            new InputMetadataProvider() {
              @Nullable
              @Override
              public FileArtifactValue getInputMetadataChecked(ActionInput input)
                  throws InterruptedException {
                if (input == treeFileArtifact && treeFileMetadataAccessed.getCount() >= 0) {
                  treeFileMetadataAccessed.countDown();
                  treeFileMetadataContinue.await();
                }
                return null;
              }

              @Nullable
              @Override
              public TreeArtifactValue getTreeMetadata(ActionInput input) {
                return null;
              }

              @Nullable
              @Override
              public TreeArtifactValue getEnclosingTreeMetadata(PathFragment execPath) {
                return null;
              }

              @Nullable
              @Override
              public FilesetOutputTree getFileset(ActionInput input) {
                return null;
              }

              @Override
              public ImmutableMap<Artifact, FilesetOutputTree> getFilesets() {
                return ImmutableMap.of();
              }

              @Nullable
              @Override
              public RunfilesArtifactValue getRunfilesMetadata(ActionInput input) {
                return null;
              }

              @Nullable
              @Override
              public ImmutableList<RunfilesTree> getRunfilesTrees() {
                return null;
              }

              @Nullable
              @Override
              public ActionInput getInput(PathFragment execPath) {
                return null;
              }
            },
            fakeFileCache);
    var interruptedThreadThrowable = new AtomicReference<Throwable>();
    var interruptedThread =
        new Thread(
            () -> {
              try {
                var unused =
                    merkleTreeComputer.buildForSpawn(
                        spawn,
                        ImmutableSet.of(),
                        /* scrubber= */ null,
                        createSpawnExecutionContext(spawn, delayedMetadataProvider),
                        RemotePathResolver.createDefault(execRoot),
                        MerkleTreeComputer.BlobPolicy.KEEP);
              } catch (Throwable t) {
                if (t instanceof InterruptedException) {
                  Thread.currentThread().interrupt();
                }
                interruptedThreadThrowable.set(t);
              }
            });
    interruptedThread.start();
    // Wait until the Merkle subtree for the tree artifact started to build.
    treeFileMetadataAccessed.await();

    var unrelatedThreads = new ArrayList<Thread>();
    var unrelatedThreadThrowables = new ArrayList<AtomicReference<Throwable>>();
    for (int i = 0; i < 10; i++) {
      var capturedThrowable = new AtomicReference<Throwable>();
      unrelatedThreadThrowables.add(capturedThrowable);
      var thread =
          new Thread(
              () -> {
                try {
                  var unused =
                      merkleTreeComputer.buildForSpawn(
                          spawn,
                          ImmutableSet.of(),
                          /* scrubber= */ null,
                          createSpawnExecutionContext(spawn, fakeFileCache),
                          RemotePathResolver.createDefault(execRoot),
                          MerkleTreeComputer.BlobPolicy.KEEP);
                } catch (Throwable t) {
                  if (t instanceof InterruptedException) {
                    Thread.currentThread().interrupt();
                  }
                  capturedThrowable.set(t);
                }
              });
      thread.start();
      unrelatedThreads.add(thread);
      // Wait for the new subtree build to block on the first one.
      while (thread.getState() == Thread.State.RUNNABLE || thread.getState() == Thread.State.NEW) {
        Thread.sleep(Duration.ofMillis(10));
      }
    }

    // Interrupting the first build does not result in its subtree build future being canceled since
    // other threads are also waiting for it.
    interruptedThread.interrupt();
    treeFileMetadataContinue.countDown();
    interruptedThread.join();
    for (var thread : unrelatedThreads) {
      thread.join();
    }

    assertThat(interruptedThreadThrowable.get()).isInstanceOf(InterruptedException.class);
    for (var capturedThrowable : unrelatedThreadThrowables) {
      if (capturedThrowable.get() != null) {
        throw capturedThrowable.get();
      }
    }

    // All threads share a single upload of the subtree.
    assertThat(ensureInputsPresentCount.get()).isEqualTo(1);
  }

  @Test
  public void duplicateMappedPath_sameDigest_succeeds() throws Exception {
    // Two artifacts from different configurations that map to the same path with identical content.
    var fastbuildRoot =
        ArtifactRoot.asDerivedRoot(
            execRoot, ArtifactRoot.RootType.OUTPUT, "outputs", "k8-fastbuild", "bin");
    checkNotNull(fastbuildRoot.getRoot().asPath()).createDirectoryAndParents();
    var stRoot =
        ArtifactRoot.asDerivedRoot(
            execRoot, ArtifactRoot.RootType.OUTPUT, "outputs", "k8-fastbuild-ST-1234", "bin");
    checkNotNull(stRoot.getRoot().asPath()).createDirectoryAndParents();

    Artifact artifact1 = ActionsTestUtil.createArtifact(fastbuildRoot, "pkg/foo.h");
    Artifact artifact2 = ActionsTestUtil.createArtifact(stRoot, "pkg/foo.h");
    artifact1.getPath().getParentDirectory().createDirectoryAndParents();
    artifact2.getPath().getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(artifact1.getPath(), "same content");
    FileSystemUtils.writeContentAsLatin1(artifact2.getPath(), "same content");

    FakeActionInputFileCache cache = new FakeActionInputFileCache();
    FileArtifactValue metadata1 = FileArtifactValue.createForTesting(artifact1);
    FileArtifactValue metadata2 = FileArtifactValue.createForTesting(artifact2);
    cache.put(artifact1, metadata1);
    cache.put(artifact2, metadata2);

    // PathMapper that strips the config segment
    PathMapper strippingMapper =
        new PathMapper() {
          @Override
          public PathFragment map(PathFragment execPath) {
            if (execPath.startsWith(PathFragment.create("outputs"))
                && execPath.segmentCount() >= 3) {
              return execPath
                  .subFragment(0, 1)
                  .getRelative("cfg")
                  .getRelative(execPath.subFragment(2));
            }
            return execPath;
          }
        };

    Spawn spawn =
        new SpawnBuilder().withInputs(artifact1, artifact2).setPathMapper(strippingMapper).build();
    var merkleTreeComputer = createMerkleTreeComputer(/* uploader= */ null);

    // Should succeed without error — same digest allows the collision
    var unused =
        merkleTreeComputer.buildForSpawn(
            spawn,
            ImmutableSet.of(),
            /* scrubber= */ null,
            createSpawnExecutionContext(spawn, cache),
            RemotePathResolver.createDefault(execRoot),
            MerkleTreeComputer.BlobPolicy.KEEP);
  }

  @Test
  public void duplicateTreeArtifactInput_stagedOnce() throws Exception {
    var fakeFileCache = new FakeActionInputFileCache();
    var treeArtifact =
        ActionsTestUtil.createTreeArtifactWithGeneratingAction(artifactRoot, "pkg/gen_src");
    treeArtifact.getPath().createDirectoryAndParents();
    var treeArtifactBuilder = TreeArtifactValue.newBuilder(treeArtifact);
    var treeFileArtifact = Artifact.TreeFileArtifact.createTreeOutput(treeArtifact, "gen.cc");
    FileSystemUtils.writeContentAsLatin1(treeFileArtifact.getPath(), "int main() {}");
    treeArtifactBuilder.putChild(
        treeFileArtifact, FileArtifactValue.createForTesting(treeFileArtifact));
    fakeFileCache.putTreeArtifact(treeArtifact, treeArtifactBuilder.build());

    // The same input can appear in both the mandatory and the discovered input set of an
    // input-discovering action, which are not deduplicated against each other.
    var spawn =
        new SimpleSpawn(
            new FakeOwner("Mnemonic", "progress message", "//dummy:label"),
            ImmutableList.of("cmd"),
            ImmutableMap.of(),
            ImmutableMap.of(),
            SpawnInputs.of(
                NestedSetBuilder.<ActionInput>create(Order.STABLE_ORDER, treeArtifact),
                NestedSetBuilder.<ActionInput>create(Order.STABLE_ORDER, treeArtifact),
                ImmutableList.of()),
            NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            ImmutableSet.of(),
            /* mandatoryOutputs= */ null,
            ResourceSet.ZERO);
    var merkleTree =
        (MerkleTree.Uploadable)
            createMerkleTreeComputer(/* uploader= */ null)
                .buildForSpawn(
                    spawn,
                    ImmutableSet.of(),
                    /* scrubber= */ null,
                    createSpawnExecutionContext(spawn, fakeFileCache),
                    RemotePathResolver.createDefault(execRoot),
                    MerkleTreeComputer.BlobPolicy.KEEP);

    Directory pkgDir = findDirectory(merkleTree, "outputs", "pkg");
    assertThat(pkgDir.getDirectoriesList().stream().map(DirectoryNode::getName))
        .containsExactly("gen_src");
    assertThat(pkgDir.getFilesList()).isEmpty();
  }

  private static Directory findDirectory(MerkleTree.Uploadable merkleTree, String... pathSegments)
      throws Exception {
    var blobs = merkleTree.blobs();
    Directory current = Directory.parseFrom((byte[]) blobs.get(merkleTree.digest()));
    for (String segment : pathSegments) {
      Digest childDigest =
          current.getDirectoriesList().stream()
              .filter(d -> d.getName().equals(segment))
              .collect(onlyElement())
              .getDigest();
      current = Directory.parseFrom((byte[]) blobs.get(childDigest));
    }
    return current;
  }

  private MerkleTreeComputer createMerkleTreeComputer(MerkleTreeUploader uploader) {
    return new MerkleTreeComputer(
        new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256),
        uploader,
        "buildRequestId",
        "commandId",
        "_main");
  }

  private FakeSpawnExecutionContext createSpawnExecutionContext(
      Spawn spawn, InputMetadataProvider inputMetadataProvider) {
    return new FakeSpawnExecutionContext(
        spawn,
        inputMetadataProvider,
        execRoot,
        new FileOutErr(),
        ImmutableClassToInstanceMap.of(),
        /* actionFileSystem= */ null);
  }
}
