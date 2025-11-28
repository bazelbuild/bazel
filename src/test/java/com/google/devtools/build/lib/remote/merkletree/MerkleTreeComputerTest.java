package com.google.devtools.build.lib.remote.merkletree;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.DelegatingPairInputMetadataProvider;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FilesetOutputTree;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.RunfilesArtifactValue;
import com.google.devtools.build.lib.actions.RunfilesTree;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.exec.util.FakeActionInputFileCache;
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
import java.util.Map;
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
        createDelayedInputMetadataProvider(
            fakeFileCache, treeFileArtifact, treeFileMetadataAccessed);
    var capturedThrowable = new AtomicReference<Throwable>();
    var buildThread =
        new Thread(
            () -> {
              try {
                merkleTreeComputer.buildForSpawn(
                    spawn,
                    ImmutableSet.of(),
                    /* scrubber= */ null,
                    createSpawnExecutionContext(spawn, delayedMetadataProvider),
                    RemotePathResolver.createDefault(execRoot),
                    MerkleTreeComputer.BlobPolicy.KEEP);
              } catch (Throwable t) {
                capturedThrowable.set(t);
              }
            });
    buildThread.start();
    // Wait until the Merkle subtree for the tree artifact started to build, then interrupt it.
    treeFileMetadataAccessed.await();
    buildThread.interrupt();
    buildThread.join();
    assertThat(capturedThrowable.get()).isInstanceOf(InterruptedException.class);

    // Expected to succeed despite the subtree computation being canceled before.
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
                  RemoteActionExecutionContext context, Digest digest, byte[] data) {
                return immediateVoidFuture();
              }

              @Override
              public ListenableFuture<Void> uploadFile(
                  RemoteActionExecutionContext context,
                  RemotePathResolver remotePathResolver,
                  Digest digest,
                  Path path) {
                return immediateVoidFuture();
              }

              @Override
              public ListenableFuture<Void> uploadVirtualActionInput(
                  RemoteActionExecutionContext context,
                  Digest digest,
                  VirtualActionInput virtualActionInput) {
                return immediateVoidFuture();
              }

              @Override
              public void ensureInputsPresent(
                  RemoteActionExecutionContext context,
                  MerkleTree.Uploadable merkleTree,
                  boolean force,
                  RemotePathResolver remotePathResolver)
                  throws IOException, InterruptedException {
                ensureInputsPresentCount.incrementAndGet();
              }
            });

    var treeFileMetadataAccessed = new CountDownLatch(1);
    var delayedMetadataProvider =
        createDelayedInputMetadataProvider(
            fakeFileCache, treeFileArtifact, treeFileMetadataAccessed);
    var interruptedThreadThrowable = new AtomicReference<Throwable>();
    var interruptedThread =
        new Thread(
            () -> {
              try {
                merkleTreeComputer.buildForSpawn(
                    spawn,
                    ImmutableSet.of(),
                    /* scrubber= */ null,
                    createSpawnExecutionContext(spawn, delayedMetadataProvider),
                    RemotePathResolver.createDefault(execRoot),
                    MerkleTreeComputer.BlobPolicy.KEEP);
              } catch (Throwable t) {
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
                  merkleTreeComputer.buildForSpawn(
                      spawn,
                      ImmutableSet.of(),
                      /* scrubber= */ null,
                      createSpawnExecutionContext(spawn, fakeFileCache),
                      RemotePathResolver.createDefault(execRoot),
                      MerkleTreeComputer.BlobPolicy.KEEP);
                } catch (Throwable t) {
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

    // Interrupting the first build results in its subtree build future being cancelled, but this
    // should not affect the second build.
    interruptedThread.interrupt();
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

    // The interrupted thread never made it to the upload part and all other threads should share a
    // single upload of the subtree.
    assertThat(ensureInputsPresentCount.get()).isEqualTo(1);
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

  private InputMetadataProvider createDelayedInputMetadataProvider(
      InputMetadataProvider base, ActionInput inputToDelay, CountDownLatch sleepSignal) {
    return new DelegatingPairInputMetadataProvider(
        new InputMetadataProvider() {
          @Nullable
          @Override
          public FileArtifactValue getInputMetadataChecked(ActionInput input)
              throws InterruptedException {
            if (input != inputToDelay || sleepSignal.getCount() == 0) {
              return null;
            }
            sleepSignal.countDown();
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
          public Map<Artifact, FilesetOutputTree> getFilesets() {
            return Map.of();
          }

          @Nullable
          @Override
          public RunfilesArtifactValue getRunfilesMetadata(ActionInput input) {
            return null;
          }

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
        base);
  }
}
