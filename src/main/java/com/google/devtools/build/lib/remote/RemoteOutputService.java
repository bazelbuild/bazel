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

import static com.google.common.base.Preconditions.checkNotNull;

import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.FilesetOutputTree;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.LostInputsActionExecutionException;
import com.google.devtools.build.lib.actions.OutputChecker;
import com.google.devtools.build.lib.actions.cache.OutputMetadataStore;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.buildtool.buildevent.ExecutionPhaseCompleteEvent;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.server.FailureDetails.Execution;
import com.google.devtools.build.lib.server.FailureDetails.Execution.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.vfs.BatchStat;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.OutputService;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.io.IOException;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.stream.Stream;
import javax.annotation.Nullable;

/** Output service implementation for the remote build without local output service daemon. */
public class RemoteOutputService implements OutputService {

  private final BlazeDirectories directories;
  private final ConcurrentHashMap<ActionExecutionMetadata, Cancellable> postExecutionTasks =
      new ConcurrentHashMap<>();

  @Nullable private RemoteOutputChecker remoteOutputChecker;
  @Nullable private RemoteActionInputFetcher actionInputFetcher;
  @Nullable private LeaseService leaseService;

  RemoteOutputService(BlazeDirectories directories) {
    this.directories = checkNotNull(directories);
  }

  void setRemoteOutputChecker(RemoteOutputChecker remoteOutputChecker) {
    this.remoteOutputChecker = remoteOutputChecker;
  }

  void setActionInputFetcher(RemoteActionInputFetcher actionInputFetcher) {
    this.actionInputFetcher = checkNotNull(actionInputFetcher, "actionInputFetcher");
  }

  void setLeaseService(LeaseService leaseService) {
    this.leaseService = leaseService;
  }

  @Override
  public ActionFileSystemType actionFileSystemType() {
    return actionInputFetcher != null
        ? ActionFileSystemType.REMOTE_FILE_SYSTEM
        : ActionFileSystemType.DISABLED;
  }

  @Nullable
  @Override
  public FileSystem createActionFileSystem(
      FileSystem delegateFileSystem,
      PathFragment execRootFragment,
      String relativeOutputPath,
      ImmutableList<Root> sourceRoots,
      InputMetadataProvider inputArtifactData,
      Iterable<Artifact> outputArtifacts,
      boolean rewindingEnabled) {
    checkNotNull(actionInputFetcher, "actionInputFetcher");
    return new RemoteActionFileSystem(
        delegateFileSystem,
        execRootFragment,
        relativeOutputPath,
        inputArtifactData,
        actionInputFetcher);
  }

  @Override
  public void updateActionFileSystemContext(
      ActionExecutionMetadata action,
      FileSystem actionFileSystem,
      OutputMetadataStore outputMetadataStore) {
    ((RemoteActionFileSystem) actionFileSystem).updateContext(action);
  }

  @Override
  public String getFileSystemName(String outputBaseFileSystemName) {
    return "remoteActionFS";
  }

  @Override
  public ModifiedFileSet startBuild(
      UUID buildId, String workspaceName, EventHandler eventHandler, boolean finalizeActions)
      throws AbruptExitException {
    // One of the responsibilities of OutputService.startBuild() is that it ensures the output path
    // is valid. If the previous OutputService redirected the output path to a remote location, we
    // must undo this.
    Path outputPath = directories.getOutputPath(workspaceName);
    if (outputPath.isSymbolicLink()) {
      try {
        outputPath.delete();
      } catch (IOException e) {
        throw new AbruptExitException(
            DetailedExitCode.of(
                FailureDetail.newBuilder()
                    .setMessage(
                        String.format("Couldn't remove output path symlink: %s", e.getMessage()))
                    .setExecution(
                        Execution.newBuilder().setCode(Code.LOCAL_OUTPUT_DIRECTORY_SYMLINK_FAILURE))
                    .build()),
            e);
      }
    }
    return ModifiedFileSet.EVERYTHING_MODIFIED;
  }

  @Override
  public void flushOutputTree() throws InterruptedException {
    if (actionInputFetcher != null) {
      actionInputFetcher.flushOutputTree();
    }
  }

  @Override
  public void finalizeBuild(boolean buildSuccessful) {
    // Intentionally left empty.
  }

  @Subscribe
  public void onExecutionPhaseCompleteEvent(ExecutionPhaseCompleteEvent event) {
    if (leaseService != null) {
      leaseService.finalizeExecution();
    }
  }

  @Override
  public void finalizeAction(Action action, OutputMetadataStore outputMetadataStore)
      throws IOException, InterruptedException {
    if (actionInputFetcher != null) {
      actionInputFetcher.finalizeAction(action, outputMetadataStore);
    }

    if (leaseService != null) {
      leaseService.finalizeAction();
    }
  }

  @Override
  public boolean shouldStoreRemoteOutputMetadataInActionCache() {
    return true;
  }

  @Override
  public OutputChecker getOutputChecker() {
    return checkNotNull(remoteOutputChecker, "remoteOutputChecker must not be null");
  }

  @Nullable
  @Override
  public BatchStat getBatchStatter() {
    return null;
  }

  @Override
  public boolean canCreateSymlinkTree() {
    /* TODO(buchgr): Optimize symlink creation for remote execution */
    return false;
  }

  @Override
  public void createSymlinkTree(
      Map<PathFragment, PathFragment> symlinks, PathFragment symlinkTreeRoot) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void clean() {
    // Intentionally left empty.
  }

  @Override
  public boolean supportsPathResolverForArtifactValues() {
    return actionFileSystemType() != ActionFileSystemType.DISABLED;
  }

  @Override
  public ArtifactPathResolver createPathResolverForArtifactValues(
      PathFragment execRoot,
      String relativeOutputPath,
      FileSystem fileSystem,
      ImmutableList<Root> pathEntries,
      ActionInputMap actionInputMap,
      Map<Artifact, ImmutableSortedSet<TreeFileArtifact>> treeArtifacts,
      Map<Artifact, FilesetOutputTree> filesets) {
    FileSystem remoteFileSystem =
        new RemoteActionFileSystem(
            fileSystem, execRoot, relativeOutputPath, actionInputMap, actionInputFetcher);
    return ArtifactPathResolver.createPathResolver(remoteFileSystem, fileSystem.getPath(execRoot));
  }

  @Override
  public void checkActionFileSystemForLostInputs(FileSystem actionFileSystem, Action action)
      throws LostInputsActionExecutionException {
    if (actionFileSystem instanceof RemoteActionFileSystem remoteFileSystem) {
      remoteFileSystem.checkForLostInputs(action);
    }
  }

  @Override
  public void registerPostExecutionTask(ActionExecutionMetadata action, Cancellable task) {
    // We don't expect to have multiple post-execution tasks for the same action registered at the
    // same time.
    postExecutionTasks.merge(
        action,
        task,
        (oldTask, newTask) -> {
          throw new IllegalStateException(
              "Attempted to register multiple post-execution tasks for %s: %s and %s"
                  .formatted(action, oldTask, newTask));
        });
  }

  @Override
  public void cancelPostExecutionTasks(ActionExecutionMetadata action) throws InterruptedException {
    Cancellable task = postExecutionTasks.remove(action);
    if (task != null) {
      task.cancel();
    }
  }

  @Override
  public RewoundActionSynchronizer createRewoundActionSynchronizer(boolean rewindingEnabled) {
    if (rewindingEnabled && actionInputFetcher != null) {
      return new RemoteRewoundActionSynchronizer();
    }
    return RewoundActionSynchronizer.NOOP;
  }

  /**
   * A {@link RewoundActionSynchronizer} implementation for Bazel's remote filesystem, which is
   * backed by actual files on disk and requires synchronization to ensure that action outputs
   * aren't deleted while they are being read.
   */
  final class RemoteRewoundActionSynchronizer implements RewoundActionSynchronizer {
    // A single coarse lock is used to synchronize rewound actions (writers) and both rewound and
    // non-rewound actions (readers) as long as no rewound action has attempted to prepare for its
    // execution.
    // This ensures high throughput and low memory footprint for the common case of no rewound
    // actions. In this case, there won't be any writers and the performance characteristics of a
    // ReentrantReadWriteLock are comparable to that of an atomic counter.
    // Note that it wouldn't be correct to only start using this lock once an action is rewound,
    // because a non-rewound actions consuming its non-lost outputs could have already started
    // executing.
    @Nullable private volatile ReadWriteLock coarseLock = new ReentrantReadWriteLock();

    // A fine-grained lock structure that is switched to when the first rewound action attempts to
    // prepare for its execution. This structure is used to ensure that rewound actions do not
    // delete their outputs while they are being read by other actions, while still allowing
    // rewound actions and non-rewound actions to run concurrently (i.e., not force the equivalent
    // of --jobs=1 for as long as a rewound action is running, as the coarse lock would).
    // A rewound action will acquire a write lock on its lookup data before it prepares for
    // execution, while any action will acquire a read lock on the lookup data of any generating
    // action of its inputs before it starts executing.
    // The values of this cache are weakly referenced to ensure that locks are cleaned up when they
    // are no longer needed.
    @Nullable
    private volatile LoadingCache<ActionLookupData, WeakSafeReentrantReadWriteLock> fineLocks;

    /*
    Proof of deadlock freedom:

    As long as the coarse lock is used, there can't be any deadlock because there is only a single
    read-write lock.

    Now assume that there is a deadlock while the fine locks are used. Consider the directed
    labeled "wait-for" graph defined as follows:

    * Nodes are given by the currently active Skyframe action execution threads, each of which is
      identified with the action it is (or will be) executing. Actions are in one-to-one
      correspondence with the ActionLookupData that is used as the key in the fine locks map.
    * For each pair of actions A_1 and A_2, there is an edge from A_1 to B_2 labeled with XY(A_3)
      if A_1 is waiting for the X lock of A_3 and B currently holds the Y lock of A_3, where X and Y
      are either R (for read) or W (for write). The resulting graph may have parallel edges with
      distinct labels.

    Let C be any directed cycle in the graph representing a deadlock, let A_1 -[XY(A_3)]-> A_2 be an
    edge in C and consider the following cases for the pair XY:

    * RR: Since a read-write lock whose read lock is held by at least one thread doesn't
          block any other thread from acquiring its read lock, this case doesn't occur.
    * WW: The write lock of A_3 is only ever (attempted to be) acquired by A_3 itself when it is
          rewound, which means that the edge would necessarily be of the shape A_3 -[WW(A_3)]-> A_3.
          But this isn't possible since the read-write locks are reentrant.
    * WR: In this case, A_1 attempts to acquire a write lock, which only happens when A_1 is a
          rewound action about to prepare for its (re-)execution. This means that the edge is
          necessarily of the shape A_1 -[WR(A_1)]-> A_2. While a rewound action is waiting for its
          own write lock in enterActionPeparation, it doesn't hold any locks since
          enterActionExecution hasn't been called yet in SkyframeActionExecutor and all past
          executions of the action have released all their locks due to use of try-with-resources.
          This means that A_1 can't have any incoming edges in the wait-for graph, which is a
          contradiction to the assumption that it is contained in the directed cycle C.

     We conclude that XY = RW. Since the write lock of A_3 is only ever acquired by A_3 itself, all
     edges in C are of the form A_1 -[RW(A_2)]-> A_2. But by construction of inputKeysFor, the
     action A_1 is attempting to acquire the read locks of all its inputs' generating actions, and
     thus the action A_1 depends on one of the outputs of A_2 (*).

     Applied to all edges of C, we conclude that there is a corresponding directed cycle in the
     action graph, which is a contradiction since Bazel disallows dependency cycles.

     Notes:
     * The proof would not go through at (*) if fineLocks was replaced by a Striped lock structure
       with a fixed number of locks. In fact, this gives rise to a deadlock if the number of stripes
       is at least 2, but low enough that distinct generating actions hash to the same stripe.
     */

    @Override
    public SilentCloseable enterActionPreparation(Action action, boolean wasRewound)
        throws InterruptedException {
      // Skyframe schedules non-rewound actions such that they never run concurrently with actions
      // that consume their outputs.
      if (!wasRewound) {
        return () -> {};
      }
      try (SilentCloseable c =
          Profiler.instance().profile(ProfilerTask.ACTION_LOCK, "action.enterActionPreparation")) {
        return enterActionPreparationInternal(action);
      }
    }

    private SilentCloseable enterActionPreparationInternal(Action action)
        throws InterruptedException {
      var localCoarseLock = coarseLock;
      Lock writeLock;

      if (localCoarseLock == null) {
        // Common case after some action has been rewound and thus inflated the fine locks, acquire
        // the single write lock representing this action and prepare its outputs for rewinding.
        writeLock = fineLocks.get(outputKeyFor(action)).writeLock();
        writeLock.lockInterruptibly();
        prepareOutputsForRewinding(action);
        return writeLock::unlock;
      } else {
        // This is the first time a rewound action has attempted to prepare for its execution.
        // Atomically switch to the fine locks structure.
        localCoarseLock.writeLock().lockInterruptibly();
        // At this point, all other actions are blocked on the read lock.
        var localFineLocks =
            Caffeine.newBuilder()
                .weakValues()
                // TODO: Investigate whether fair locks would be beneficial.
                .build((ActionLookupData unused) -> new WeakSafeReentrantReadWriteLock());
        // Lock the corresponding fine lock and publish the fine locks before releasing the coarse
        // lock.
        writeLock = localFineLocks.get(outputKeyFor(action)).writeLock();
        // We just created the lock, so locking it never blocks.
        writeLock.lock();
        fineLocks = localFineLocks;
        coarseLock = null;
        localCoarseLock.writeLock().unlock();
        // Safe to continue under the fine write lock only since blocked actions will acquire the
        // fine read lock after the write lock is released.
      }

      prepareOutputsForRewinding(action);
      return writeLock::unlock;
    }

    /**
     * Cancels all async tasks that operate on the action's outputs and resets any cached data about
     * their prefetching state.
     */
    private void prepareOutputsForRewinding(Action action) throws InterruptedException {
      cancelPostExecutionTasks(action);
      actionInputFetcher.handleRewoundActionOutputs(action.getOutputs());
    }

    @Override
    public SilentCloseable enterActionExecution(
        Action action, InputMetadataProvider metadataProvider) throws InterruptedException {
      try (SilentCloseable c =
          Profiler.instance().profile(ProfilerTask.ACTION_LOCK, "action.enterActionExecution")) {
        return enterActionExecutionInternal(action, metadataProvider);
      }
    }

    private SilentCloseable enterActionExecutionInternal(
        Action action, InputMetadataProvider metadataProvider) throws InterruptedException {
      var localCoarseLock = coarseLock;
      if (localCoarseLock != null) {
        // Common case for builds without any rewound actions: acquire the single lock that is never
        // acquired by a writer.
        localCoarseLock.readLock().lockInterruptibly();
      }
      // Read the fine locks after acquiring the coarse lock to allow the fine locks to be inflated
      // lazily.
      var localFineLocks = fineLocks;
      if (localFineLocks == null) {
        // Continuation of the common case for builds without any rewound actions: the fine locks
        // have not been inflated.
        return localCoarseLock.readLock()::unlock;
      }

      // At this point, there has been at least one rewound action that has inflated the fine locks.
      // We need to switch to it.
      if (localCoarseLock != null) {
        localCoarseLock.readLock().unlock();
      }
      var allReadWriteLocks =
          localFineLocks.getAll(inputKeysFor(action, metadataProvider)).values();
      var locksToUnlockBuilder =
          ImmutableList.<Lock>builderWithExpectedSize(allReadWriteLocks.size());
      try {
        for (var readWriteLock : allReadWriteLocks) {
          var readLock = readWriteLock.readLock();
          readLock.lockInterruptibly();
          locksToUnlockBuilder.add(readLock);
        }
      } catch (InterruptedException e) {
        for (var readLock : locksToUnlockBuilder.build()) {
          readLock.unlock();
        }
        throw e;
      }
      var locksToUnlock = locksToUnlockBuilder.build();
      return () -> locksToUnlock.forEach(Lock::unlock);
    }

    private static Iterable<ActionLookupData> inputKeysFor(
        Action action, InputMetadataProvider metadataProvider) {
      return () ->
          Stream.concat(
                  action.getInputs().toList().stream(),
                  metadataProvider.getRunfilesTrees().stream()
                      .flatMap(runfilesTree -> runfilesTree.getArtifacts().toList().stream()))
              .filter(artifact -> artifact instanceof DerivedArtifact)
              .map(artifact -> ((DerivedArtifact) artifact).getGeneratingActionKey())
              .iterator();
    }

    private static ActionLookupData outputKeyFor(Action action) {
      return ((DerivedArtifact) action.getPrimaryOutput()).getGeneratingActionKey();
    }

    // Classes below are based on Guava's Striped class, but optimized for memory usage by using
    // extension rather than delegation:
    // https://github.com/google/guava/blob/d25d62fc843ece1c3866859bc8639b815093eac8/guava/src/com/google/common/util/concurrent/Striped.java#L282-L326

    /**
     * ReadWriteLock implementation whose read and write locks retain a reference back to this lock.
     * Otherwise, a reference to just the read lock or just the write lock would not suffice to
     * ensure the {@code ReadWriteLock} is retained.
     */
    private static final class WeakSafeReentrantReadWriteLock extends ReentrantReadWriteLock {
      @Override
      public WeakSafeReadLock readLock() {
        return new WeakSafeReadLock(this);
      }

      @Override
      public WeakSafeWriteLock writeLock() {
        return new WeakSafeWriteLock(this);
      }
    }

    /**
     * A read lock that ensures a strong reference is retained to the owning {@link ReadWriteLock}.
     */
    private static final class WeakSafeReadLock extends ReentrantReadWriteLock.ReadLock {
      @SuppressWarnings({"unused", "FieldCanBeLocal"})
      private final WeakSafeReentrantReadWriteLock strongReference;

      WeakSafeReadLock(WeakSafeReentrantReadWriteLock readWriteLock) {
        super(readWriteLock);
        this.strongReference = readWriteLock;
      }
    }

    /**
     * A write lock that ensures a strong reference is retained to the owning {@link ReadWriteLock}.
     */
    private static final class WeakSafeWriteLock extends ReentrantReadWriteLock.WriteLock {
      @SuppressWarnings({"unused", "FieldCanBeLocal"})
      private final WeakSafeReentrantReadWriteLock strongReference;

      WeakSafeWriteLock(WeakSafeReentrantReadWriteLock readWriteLock) {
        super(readWriteLock);
        this.strongReference = readWriteLock;
      }

      @Override
      public Condition newCondition() {
        throw new UnsupportedOperationException();
      }
    }
  }
}
