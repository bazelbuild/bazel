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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.util.concurrent.Futures.addCallback;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.remote.util.RxFutures.toCompletable;
import static com.google.devtools.build.lib.remote.util.RxFutures.toListenableFuture;
import static com.google.devtools.build.lib.remote.util.RxUtils.mergeBulkTransfer;
import static com.google.devtools.build.lib.remote.util.RxUtils.toTransferResult;
import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.util.AsyncTaskCache;
import com.google.devtools.build.lib.remote.util.RxUtils.TransferResult;
import com.google.devtools.build.lib.remote.util.TempPathGenerator;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.OutputPermissions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.core.Flowable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.Stack;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/**
 * Abstract implementation of {@link ActionInputPrefetcher} which implements the orchestration of
 * prefeching multiple inputs so subclasses can focus on prefetching / downloading single input.
 */
public abstract class AbstractActionInputPrefetcher implements ActionInputPrefetcher {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final Reporter reporter;
  private final AsyncTaskCache.NoResult<Path> downloadCache = AsyncTaskCache.NoResult.create();
  private final TempPathGenerator tempPathGenerator;
  private final OutputPermissions outputPermissions;
  protected final Set<Artifact> outputsAreInputs = Sets.newConcurrentHashSet();

  protected final Path execRoot;
  protected final ImmutableList<Pattern> patternsToDownload;

  private final Set<ActionInput> missingActionInputs = Sets.newConcurrentHashSet();

  // Tracks directories temporarily made writable by prefetcher calls.
  // Concurrent calls may write to the same directories, so it's not safe to put back their
  // permissions until there are no ongoing calls, as tracked by numCallsInProgress below.
  private final Set<Path> temporarilyWritableDirectories = Sets.newConcurrentHashSet();

  // Tracks the number of currently ongoing prefetcher calls.
  @GuardedBy("this")
  private int numOngoingCalls = 0;

  /**
   * A symlink in the output tree.
   */

  @AutoValue
  static abstract class Symlink {

    abstract PathFragment getLinkExecPath();

    abstract PathFragment getTargetExecPath();

    static Symlink of(PathFragment linkExecPath, PathFragment targetExecPath) {
      checkArgument(!linkExecPath.equals(targetExecPath));
      return new AutoValue_AbstractActionInputPrefetcher_Symlink(linkExecPath,
          targetExecPath);
    }
  }

  /** Priority for the staging task. */
  protected enum Priority {
    /**
     * Critical priority tasks are tasks that are critical to the execution time e.g. staging files
     * for in-process actions.
     */
    CRITICAL,
    /**
     * High priority tasks are tasks that may have impact on the execution time e.g. staging outputs
     * that are inputs to local actions which will be executed later.
     */
    HIGH,
    /**
     * Medium priority tasks are tasks that may or may not have the impact on the execution time
     * e.g. staging inputs for local branch of dynamically scheduled actions.
     */
    MEDIUM,
    /**
     * Low priority tasks are tasks that don't have impact on the execution time e.g. staging
     * outputs of toplevel targets/aspects.
     */
    LOW,
  }

  protected AbstractActionInputPrefetcher(
      Reporter reporter,
      Path execRoot,
      TempPathGenerator tempPathGenerator,
      ImmutableList<Pattern> patternsToDownload,
      OutputPermissions outputPermissions) {
    this.reporter = reporter;
    this.execRoot = execRoot;
    this.tempPathGenerator = tempPathGenerator;
    this.patternsToDownload = patternsToDownload;
    this.outputPermissions = outputPermissions;
  }

  private boolean shouldDownloadFile(Path path, FileArtifactValue metadata) {
    if (!path.exists()) {
      return true;
    }

    // In the most cases, skyframe should be able to detect source files modifications and delete
    // staled outputs before action execution. However, there are some cases where outputs are not
    // tracked by skyframe. We compare the digest here to make sure we don't use staled files.
    try {
      byte[] digest = path.getFastDigest();
      if (digest == null) {
        digest = path.getDigest();
      }
      return !Arrays.equals(digest, metadata.getDigest());
    } catch (IOException ignored) {
      return true;
    }
  }

  protected abstract boolean canDownloadFile(Path path, FileArtifactValue metadata);

  /**
   * Downloads file to the given path via its metadata.
   *
   * @param tempPath the temporary path which the input should be written to.
   */
  protected abstract ListenableFuture<Void> doDownloadFile(
      Path tempPath, PathFragment execPath, FileArtifactValue metadata, Priority priority)
      throws IOException;

  protected void prefetchVirtualActionInput(VirtualActionInput input) throws IOException {}

  /** Transforms the error encountered during the prefetch . */
  protected Completable onErrorResumeNext(Throwable error) {
    return Completable.error(error);
  }

  /**
   * Fetches remotely stored action outputs, that are inputs to this spawn, and stores them under
   * their path in the output base.
   *
   * <p>The {@code inputs} may not contain any unexpanded directories.
   *
   * <p>This method is safe to be called concurrently from spawn runners before running any local
   * spawn.
   *
   * @return a future that is completed once all downloads have finished.
   */
  @Override
  public ListenableFuture<Void> prefetchFiles(
      Iterable<? extends ActionInput> inputs, MetadataProvider metadataProvider) {
    return prefetchFiles(inputs, metadataProvider, Priority.MEDIUM);
  }

  protected ListenableFuture<Void> prefetchFiles(
      Iterable<? extends ActionInput> inputs,
      MetadataProvider metadataProvider,
      Priority priority) {
    List<ActionInput> files = new ArrayList<>();

    for (ActionInput input : inputs) {
      // Source artifacts don't need to be fetched.
      if (input instanceof Artifact && ((Artifact) input).isSourceArtifact()) {
        continue;
      }

      // Skip empty tree artifacts (non-empty tree artifacts should have already been expanded).
      if (input.isDirectory()) {
        continue;
      }

      files.add(input);
    }

    Flowable<TransferResult> transfers =
        Flowable.fromIterable(files)
            .flatMapSingle(
                input ->
                    toTransferResult(
                        prefetchFile(metadataProvider, input, priority)));

    Completable prefetch = Completable.using(this::startPrefetching,
        ctx -> mergeBulkTransfer(transfers),
        this::stopPrefetching).onErrorResumeNext(this::onErrorResumeNext);

    return toListenableFuture(prefetch);
  }

  private synchronized AbstractActionInputPrefetcher startPrefetching() {
    numOngoingCalls++;
    return this;
  }

  private synchronized void stopPrefetching(AbstractActionInputPrefetcher unused)
      throws IOException {
    numOngoingCalls--;
    if (numOngoingCalls == 0) {
      // Set output permissions on directories (files are handled in finalizeDownload), matching the
      // behavior of SkyframeActionExecutor#checkOutputs when artifacts are produced by a locally
      // executed action.
      for (Path dir : temporarilyWritableDirectories) {
        dir.chmod(outputPermissions.getPermissionsMode());
      }
      temporarilyWritableDirectories.clear();
    }
  }

  private Completable prefetchFile(
      MetadataProvider metadataProvider, ActionInput input, Priority priority)
      throws IOException {
    if (input instanceof VirtualActionInput) {
      prefetchVirtualActionInput((VirtualActionInput) input);
      return Completable.complete();
    }

    PathFragment execPath = input.getExecPath();

    FileArtifactValue metadata = metadataProvider.getMetadata(input);
    if (metadata == null || !canDownloadFile(execRoot.getRelative(execPath), metadata)) {
      return Completable.complete();
    }

    @Nullable Symlink symlink = maybeGetSymlink(input, metadata, metadataProvider);

    if (symlink != null) {
      checkState(execPath.startsWith(symlink.getLinkExecPath()));
      execPath = symlink.getTargetExecPath()
          .getRelative(execPath.relativeTo(symlink.getLinkExecPath()));
    }

    @Nullable PathFragment treeRootExecPath = getTreeRoot(execPath, input);

    Completable result =
        downloadFileNoCheckRx(execRoot.getRelative(execPath),
            treeRootExecPath != null ? execRoot.getRelative(treeRootExecPath) : null, input,
            metadata,
            priority);

    if (symlink != null) {
      result = result.andThen(plantSymlink(symlink));
    }

    return result;
  }

  /**
   * For an input belonging to a tree artifact, returns the tree artifact root. Otherwise, returns
   * null.
   */
  @Nullable
  private PathFragment getTreeRoot(PathFragment execPath, ActionInput input) {
    if (input instanceof TreeFileArtifact) {
      // Derive root from execPath because we may prefetch to a different location.
      int numChildComponents = ((TreeFileArtifact) input).getParentRelativePath().segmentCount();
      return execPath.subFragment(0,
          execPath.segmentCount() - numChildComponents);
    }
    return null;
  }

  /**
   * Returns the symlink to be planted in the output tree for artifacts that are prefetched into a
   * different location.
   *
   * <p>Some artifacts (notably, those created by {@code ctx.actions.symlink}) are materialized in
   * the output tree as a symlink to another artifact, as indicated by the
   * {@link FileArtifactValue#getMaterializationExecPath()} field in their (or their parent tree
   * artifact's) metadata.</p>
   */
  @Nullable
  private Symlink maybeGetSymlink(ActionInput input,
      FileArtifactValue metadata, MetadataProvider metadataProvider) throws IOException {
    if (input instanceof TreeFileArtifact) {
      // Check whether the entire tree artifact should be prefetched into a separate location.
      SpecialArtifact treeArtifact = ((TreeFileArtifact) input).getParent();
      FileArtifactValue treeMetadata = checkNotNull(metadataProvider.getMetadata(treeArtifact));
      return maybeGetSymlink(treeArtifact, treeMetadata, metadataProvider);
    }
    PathFragment execPath = input.getExecPath();
    PathFragment materializationExecPath = metadata.getMaterializationExecPath().orElse(execPath);
    if (!materializationExecPath.equals(execPath)) {
      return Symlink.of(execPath, materializationExecPath);
    }
    return null;
  }

  /**
   * Downloads file into the {@code path} with its metadata.
   *
   * <p>The file will be written into a temporary file and moved to the final destination after the
   * download finished.
   */
  private Completable downloadFileRx(
      Path path,
      @Nullable Path treeRoot,
      @Nullable ActionInput actionInput,
      FileArtifactValue metadata,
      Priority priority) {
    if (!canDownloadFile(path, metadata)) {
      return Completable.complete();
    }
    return downloadFileNoCheckRx(path, treeRoot, actionInput, metadata,
        priority);
  }

  private Completable downloadFileNoCheckRx(
      Path path,
      @Nullable Path treeRoot,
      @Nullable ActionInput actionInput,
      FileArtifactValue metadata,
      Priority priority) {
    if (path.isSymbolicLink()) {
      try {
        path = path.getRelative(path.readSymbolicLink());
      } catch (IOException e) {
        return Completable.error(e);
      }
    }

    Path finalPath = path;

    AtomicBoolean completed = new AtomicBoolean(false);
    Completable download =
        Completable.using(
                tempPathGenerator::generateTempPath,
                tempPath ->
                    toCompletable(
                        () ->
                            doDownloadFile(
                                tempPath, finalPath.relativeTo(execRoot), metadata, priority),
                        directExecutor())
                        .doOnComplete(
                            () -> {
                              finalizeDownload(treeRoot, tempPath, finalPath);
                              completed.set(true);
                            }),
                tempPath -> {
                  if (!completed.get()) {
                    deletePartialDownload(tempPath);
                  }
                },
                // Set eager=false here because we want cleanup the download *after* upstream is
                // disposed.
                /* eager= */ false)
            .doOnError(
                error -> {
                  if (error instanceof CacheNotFoundException && actionInput != null) {
                    missingActionInputs.add(actionInput);
                  }
                });

    return downloadCache.executeIfNot(
        finalPath,
        Completable.defer(
            () -> {
              if (shouldDownloadFile(finalPath, metadata)) {
                return download;
              }
              return Completable.complete();
            }));
  }

  /**
   * Download file to the {@code path} with given metadata. Blocking await for the download to
   * complete.
   *
   * <p>The file will be written into a temporary file and moved to the final destination after the
   * download finished.
   */
  public void downloadFile(Path path, @Nullable ActionInput actionInput, FileArtifactValue metadata)
      throws IOException, InterruptedException {
    getFromFuture(downloadFileAsync(path.asFragment(), actionInput, metadata, Priority.CRITICAL));
  }

  protected ListenableFuture<Void> downloadFileAsync(
      PathFragment path,
      @Nullable ActionInput actionInput,
      FileArtifactValue metadata,
      Priority priority) {
    return toListenableFuture(
        Completable.using(this::startPrefetching,
            ctx -> downloadFileRx(
                execRoot.getFileSystem().getPath(path),
                /* treeRoot= */ null,
                actionInput,
                metadata,
                priority), this::stopPrefetching));
  }

  private void finalizeDownload(@Nullable Path treeRoot, Path tmpPath, Path finalPath)
      throws IOException {
    Path parentDir = checkNotNull(finalPath.getParentDirectory());

    if (treeRoot != null) {
      checkState(parentDir.startsWith(treeRoot));

      // Create intermediate tree artifact directories.
      // Be careful to minimize filesystem calls when prefetching multiple files into the same tree.
      Stack<Path> dirs = new Stack<>();
      for (Path dir = parentDir; ; dir = dir.getParentDirectory()) {
        dirs.push(dir);
        if (dir.exists() || dir.equals(treeRoot)) {
          break;
        }
      }
      while (!dirs.empty()) {
        Path dir = dirs.pop();
        dir.createWritableDirectory();  // create or make writable
        temporarilyWritableDirectories.add(dir);
      }
    } else {
      // If the parent directory is not writable, temporarily make it so.
      // This is needed when fetching a non-tree artifact nested inside a tree artifact, or a tree
      // artifact inside a fileset (see b/254844173 for the latter).
      if (!parentDir.isWritable()) {
        parentDir.setWritable(true);
        temporarilyWritableDirectories.add(parentDir);
      }
    }

    // Set output permissions, matching the behavior of SkyframeActionExecutor#checkOutputs when the
    // artifact is produced by a locally executed action.
    tmpPath.chmod(outputPermissions.getPermissionsMode());
    FileSystemUtils.moveFile(tmpPath, finalPath);
  }

  private void deletePartialDownload(Path path) {
    try {
      path.delete();
    } catch (IOException e) {
      logger.atWarning().withCause(e).log(
          "Failed to delete output file after incomplete download: %s", path);
    }
  }

  private Completable plantSymlink(Symlink symlink) {
    return downloadCache.executeIfNot(
        execRoot.getRelative(symlink.getLinkExecPath()),
        Completable.defer(() -> {
          Path link = execRoot.getRelative(symlink.getLinkExecPath());
          Path target = execRoot.getRelative(symlink.getTargetExecPath());
          // Delete the link path if it already exists. This is the case for tree artifacts,
          // whose root directory is created before the action runs.
          link.delete();
          link.createSymbolicLink(target);
          return Completable.complete();
        }));
  }

  public ImmutableSet<Path> downloadedFiles() {
    return downloadCache.getFinishedTasks();
  }

  public ImmutableSet<Path> downloadsInProgress() {
    return downloadCache.getInProgressTasks();
  }

  @VisibleForTesting
  public AsyncTaskCache.NoResult<Path> getDownloadCache() {
    return downloadCache;
  }

  public void shutdown() {
    downloadCache.shutdown();
    while (true) {
      try {
        downloadCache.awaitTermination();
        break;
      } catch (InterruptedException ignored) {
        downloadCache.shutdownNow();
      }
    }
  }

  /** Event which is fired when inputs for local action are eagerly prefetched. */
  public static class InputsEagerlyPrefetched implements Postable {
    private final List<Artifact> artifacts;

    public InputsEagerlyPrefetched(List<Artifact> artifacts) {
      this.artifacts = artifacts;
    }

    public List<Artifact> getArtifacts() {
      return artifacts;
    }
  }

  @SuppressWarnings({"CheckReturnValue", "FutureReturnValueIgnored"})
  public void finalizeAction(Action action, MetadataHandler metadataHandler) {
    List<Artifact> inputsToDownload = new ArrayList<>();
    List<Artifact> outputsToDownload = new ArrayList<>();

    for (Artifact output : action.getOutputs()) {
      if (outputsAreInputs.remove(output)) {
        if (output.isTreeArtifact()) {
          var children = metadataHandler.getTreeArtifactChildren((SpecialArtifact) output);
          inputsToDownload.addAll(children);
        } else {
          inputsToDownload.add(output);
        }
      } else if (output.isTreeArtifact()) {
        var children = metadataHandler.getTreeArtifactChildren((SpecialArtifact) output);
        for (var file : children) {
          if (outputMatchesPattern(file)) {
            outputsToDownload.add(file);
          }
        }
      } else if (outputMatchesPattern(output)) {
        outputsToDownload.add(output);
      }
    }

    if (!inputsToDownload.isEmpty()) {
      var future = prefetchFiles(inputsToDownload, metadataHandler, Priority.HIGH);
      addCallback(
          future,
          new FutureCallback<Void>() {
            @Override
            public void onSuccess(Void unused) {
              reporter.post(new InputsEagerlyPrefetched(inputsToDownload));
            }

            @Override
            public void onFailure(Throwable throwable) {
              reporter.handle(
                  Event.warn(
                      String.format(
                          "Failed to eagerly prefetch inputs: %s", throwable.getMessage())));
            }
          },
          directExecutor());
    }

    if (!outputsToDownload.isEmpty()) {
      var future = prefetchFiles(outputsToDownload, metadataHandler, Priority.LOW);
      addCallback(
          future,
          new FutureCallback<Void>() {
            @Override
            public void onSuccess(Void unused) {}

            @Override
            public void onFailure(Throwable throwable) {
              reporter.handle(
                  Event.warn(
                      String.format("Failed to download outputs: %s", throwable.getMessage())));
            }
          },
          directExecutor());
    }
  }

  private boolean outputMatchesPattern(Artifact output) {
    for (var pattern : patternsToDownload) {
      if (pattern.matcher(output.getExecPathString()).matches()) {
        return true;
      }
    }
    return false;
  }

  public void flushOutputTree() throws InterruptedException {
    downloadCache.awaitInProgressTasks();
  }

  public ImmutableSet<ActionInput> getMissingActionInputs() {
    return ImmutableSet.copyOf(missingActionInputs);
  }
}
