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
import com.google.common.collect.ImmutableMap;
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
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import java.io.IOException;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.Supplier;
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
  @Nullable private Supplier<InputMetadataProvider> fileCacheSupplier;

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

  void setFileCacheSupplier(Supplier<InputMetadataProvider> fileCacheSupplier) {
    this.fileCacheSupplier = fileCacheSupplier;
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
      ActionInputMap inputArtifactData,
      Iterable<Artifact> outputArtifacts,
      boolean rewindingEnabled) {
    checkNotNull(actionInputFetcher, "actionInputFetcher");
    return new RemoteActionFileSystem(
        delegateFileSystem,
        execRootFragment,
        relativeOutputPath,
        inputArtifactData,
        outputArtifacts,
        fileCacheSupplier.get(),
        actionInputFetcher);
  }

  @Override
  public void updateActionFileSystemContext(
      ActionExecutionMetadata action,
      FileSystem actionFileSystem,
      Environment env,
      OutputMetadataStore outputMetadataStore,
      ImmutableMap<Artifact, FilesetOutputTree> filesets) {
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
            fileSystem,
            execRoot,
            relativeOutputPath,
            actionInputMap,
            ImmutableList.of(),
            fileCacheSupplier.get(),
            actionInputFetcher);
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

  final class RemoteRewoundActionSynchronizer implements RewoundActionSynchronizer {
    @Nullable private volatile ReadWriteLock coarseLock = new ReentrantReadWriteLock();
    @Nullable private volatile LoadingCache<ActionLookupData, ReadWriteLock> fineLocks = null;

    @Override
    public SilentCloseable enterActionPreparation(Action action, boolean wasRewound)
        throws InterruptedException {
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
      if (localCoarseLock != null) {
        localCoarseLock.writeLock().lockInterruptibly();
        System.err.println("Locked coarse write lock for " + action.prettyPrint());
        var localFineLocks =
            Caffeine.newBuilder()
                .<ActionLookupData, ReadWriteLock>build(artifact -> new ReentrantReadWriteLock());
        var fineWriteLock = localFineLocks.get(outputKeyFor(action)).writeLock();
        fineWriteLock.lock();
        System.err.println("Locked fine write lock for " + action.prettyPrint());
        fineLocks = localFineLocks;
        coarseLock = null;
        localCoarseLock.writeLock().unlock();
        System.err.println("Unlocked coarse write lock for " + action.prettyPrint());
        cancelPostExecutionTasks(action);
        return () -> {
          fineWriteLock.unlock();
          System.err.println("Unlocked fine write lock for " + action.prettyPrint());
        };
      }

      var writeLock = fineLocks.get(outputKeyFor(action)).writeLock();
      writeLock.lockInterruptibly();
      System.err.println("Locked fine write lock for " + action.prettyPrint());
      cancelPostExecutionTasks(action);
      System.err.println("Cancelled post-execution tasks for " + action);
      return () -> {
        writeLock.unlock();
        System.err.println("Unlocked fine write lock for " + action.prettyPrint());
      };
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
        localCoarseLock.readLock().lockInterruptibly();
        System.err.println("Locked coarse read lock for " + action.prettyPrint());
      }
      var localFineLocks = fineLocks;
      if (localFineLocks == null) {
        Lock lock = localCoarseLock.readLock();
        return () -> {
          lock.unlock();
          System.err.println("Unlocked coarse read lock for " + action.prettyPrint());
        };
      }
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
      System.err.println("Locked fine read locks for " + action.prettyPrint());
      return () -> {
        locksToUnlock.forEach(Lock::unlock);
        System.err.println("Unlocked fine read locks for " + action.prettyPrint());
      };
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
              .peek(key -> System.err.printf("Input key for %s: %s%n", action.prettyPrint(), key))
              .iterator();
    }

    private static ActionLookupData outputKeyFor(Action action) {
      var key = ((DerivedArtifact) action.getPrimaryOutput()).getGeneratingActionKey();
      System.err.printf("Output key for %s: %s%n", action.prettyPrint(), key);
      return key;
    }
  }
}
