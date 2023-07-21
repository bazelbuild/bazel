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
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.remote.util.RxFutures.toCompletable;
import static com.google.devtools.build.lib.remote.util.RxFutures.toListenableFuture;
import static com.google.devtools.build.lib.remote.util.RxUtils.mergeBulkTransfer;
import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.cache.OutputMetadataStore;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.util.AsyncTaskCache;
import com.google.devtools.build.lib.remote.util.RxUtils;
import com.google.devtools.build.lib.remote.util.RxUtils.TransferResult;
import com.google.devtools.build.lib.remote.util.TempPathGenerator;
import com.google.devtools.build.lib.vfs.FileSymlinkLoopException;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.OutputPermissions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.core.Flowable;
import io.reactivex.rxjava3.core.Single;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;

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

  protected final Path execRoot;
  protected final RemoteOutputChecker remoteOutputChecker;

  private final Set<ActionInput> missingActionInputs = Sets.newConcurrentHashSet();

  private static final Object dummyValue = new Object();

  /**
   * Tracks output directories temporarily made writable for prefetching. Since concurrent calls may
   * write to the same directory, it's not safe to make it non-writable until no other ongoing
   * prefetcher calls are writing to it.
   */
  private final ConcurrentHashMap<Path, DirectoryState> temporarilyWritableDirectories =
      new ConcurrentHashMap<>();

  /** The state of a single temporarily writable directory. */
  private static final class DirectoryState {
    /** The number of ongoing prefetcher calls touching this directory. */
    int numCalls;
    /** Whether the output permissions must be set on the directory when prefetching completes. */
    boolean mustSetOutputPermissions;
  }

  /**
   * Tracks output directories written to by a single prefetcher call.
   *
   * <p>This makes it possible to set the output permissions on directories touched by the
   * prefetcher call all at once, so that files prefetched within the same call don't repeatedly set
   * output permissions on the same directory.
   */
  private final class DirectoryContext {
    private final ConcurrentHashMap<Path, Object> dirs = new ConcurrentHashMap<>();

    /**
     * Makes a directory temporarily writable for the remainder of the prefetcher call associated
     * with this context.
     *
     * @param isDefinitelyTreeDir Whether this directory definitely belongs to a tree artifact.
     *     Otherwise, whether it belongs to a tree artifact is inferred from its permissions.
     */
    void createOrSetWritable(Path dir, boolean isDefinitelyTreeDir) throws IOException {
      AtomicReference<IOException> caughtException = new AtomicReference<>();

      dirs.compute(
          dir,
          (outerUnused, previousValue) -> {
            if (previousValue != null) {
              return previousValue;
            }

            temporarilyWritableDirectories.compute(
                dir,
                (innerUnused, state) -> {
                  if (state == null) {
                    state = new DirectoryState();
                    state.numCalls = 0;

                    try {
                      if (isDefinitelyTreeDir) {
                        state.mustSetOutputPermissions = true;
                        var ignored = dir.createWritableDirectory();
                      } else {
                        // If the directory is writable, it's a package and should be kept writable.
                        // Otherwise, it must belong to a tree artifact, since the directory for a
                        // tree is created in a non-writable state before prefetching begins, and
                        // this is the first time the prefetcher is seeing it.
                        state.mustSetOutputPermissions = !dir.isWritable();
                        if (state.mustSetOutputPermissions) {
                          dir.setWritable(true);
                        }
                      }
                    } catch (IOException e) {
                      caughtException.set(e);
                      return null;
                    }
                  }

                  ++state.numCalls;

                  return state;
                });

            if (caughtException.get() != null) {
              return null;
            }

            return dummyValue;
          });

      if (caughtException.get() != null) {
        throw caughtException.get();
      }
    }

    /**
     * Signals that the prefetcher call associated with this context has finished.
     *
     * <p>The output permissions will be set on any directories temporarily made writable by this
     * call, if this is the last remaining call temporarily making them writable.
     */
    void close() throws IOException {
      AtomicReference<IOException> caughtException = new AtomicReference<>();

      for (Path dir : dirs.keySet()) {
        temporarilyWritableDirectories.compute(
            dir,
            (unused, state) -> {
              checkState(state != null);
              if (--state.numCalls == 0) {
                if (state.mustSetOutputPermissions) {
                  try {
                    dir.chmod(outputPermissions.getPermissionsMode());
                  } catch (IOException e) {
                    // Store caught exceptions, but keep cleaning up the map.
                    if (caughtException.get() == null) {
                      caughtException.set(e);
                    } else {
                      caughtException.get().addSuppressed(e);
                    }
                  }
                }
              }
              return state.numCalls > 0 ? state : null;
            });
      }
      dirs.clear();

      if (caughtException.get() != null) {
        throw caughtException.get();
      }
    }
  }

  /** A symlink in the output tree. */
  @AutoValue
  abstract static class Symlink {

    abstract PathFragment getLinkExecPath();

    abstract PathFragment getTargetExecPath();

    static Symlink of(PathFragment linkExecPath, PathFragment targetExecPath) {
      checkArgument(!linkExecPath.equals(targetExecPath));
      return new AutoValue_AbstractActionInputPrefetcher_Symlink(linkExecPath, targetExecPath);
    }
  }

  protected AbstractActionInputPrefetcher(
      Reporter reporter,
      Path execRoot,
      TempPathGenerator tempPathGenerator,
      RemoteOutputChecker remoteOutputChecker,
      OutputPermissions outputPermissions) {
    this.reporter = reporter;
    this.execRoot = execRoot;
    this.tempPathGenerator = tempPathGenerator;
    this.remoteOutputChecker = remoteOutputChecker;
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
      ActionExecutionMetadata action,
      Reporter reporter,
      Path tempPath,
      PathFragment execPath,
      FileArtifactValue metadata,
      Priority priority)
      throws IOException;

  protected void prefetchVirtualActionInput(VirtualActionInput input) throws IOException {}

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
      ActionExecutionMetadata action,
      Iterable<? extends ActionInput> inputs,
      MetadataSupplier metadataSupplier,
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

    DirectoryContext dirCtx = new DirectoryContext();

    Flowable<TransferResult> transfers =
        Flowable.fromIterable(files)
            .flatMapSingle(
                input -> prefetchFile(action, dirCtx, metadataSupplier, input, priority));

    Completable prefetch =
        Completable.using(
            () -> dirCtx, ctx -> mergeBulkTransfer(transfers), DirectoryContext::close);

    return toListenableFuture(prefetch);
  }

  private Single<TransferResult> prefetchFile(
      ActionExecutionMetadata action,
      DirectoryContext dirCtx,
      MetadataSupplier metadataSupplier,
      ActionInput input,
      Priority priority) {
    try {
      if (input instanceof VirtualActionInput) {
        prefetchVirtualActionInput((VirtualActionInput) input);
        return Single.just(TransferResult.ok());
      }

      PathFragment execPath = input.getExecPath();

      FileArtifactValue metadata = metadataSupplier.getMetadata(input);
      if (metadata == null || !canDownloadFile(execRoot.getRelative(execPath), metadata)) {
        return Single.just(TransferResult.ok());
      }

      @Nullable Symlink symlink = maybeGetSymlink(input, metadata, metadataSupplier);

      if (symlink != null) {
        checkState(execPath.startsWith(symlink.getLinkExecPath()));
        execPath =
            symlink.getTargetExecPath().getRelative(execPath.relativeTo(symlink.getLinkExecPath()));
      }

      @Nullable PathFragment treeRootExecPath = maybeGetTreeRoot(input, metadataSupplier);

      Completable result =
          downloadFileNoCheckRx(
              action,
              dirCtx,
              execRoot.getRelative(execPath),
              treeRootExecPath != null ? execRoot.getRelative(treeRootExecPath) : null,
              input,
              metadata,
              priority);

      if (symlink != null) {
        result = result.andThen(plantSymlink(symlink));
      }

      return RxUtils.toTransferResult(result);
    } catch (IOException e) {
      return Single.just(TransferResult.error(e));
    } catch (InterruptedException e) {
      return Single.just(TransferResult.interrupted());
    }
  }

  /**
   * For an input belonging to a tree artifact, returns the prefetch exec path of the tree artifact
   * root. Otherwise, returns null.
   *
   * <p>Some artifacts (notably, those created by {@code ctx.actions.symlink}) are materialized in
   * the output tree as a symlink to another artifact, as indicated by the {@link
   * FileArtifactValue#getMaterializationExecPath()} field in their metadata.
   */
  @Nullable
  private PathFragment maybeGetTreeRoot(ActionInput input, MetadataSupplier metadataSupplier)
      throws IOException, InterruptedException {
    if (!(input instanceof TreeFileArtifact)) {
      return null;
    }
    SpecialArtifact treeArtifact = ((TreeFileArtifact) input).getParent();
    FileArtifactValue treeMetadata =
        checkNotNull(
            metadataSupplier.getMetadata(treeArtifact),
            "input %s belongs to a tree artifact whose metadata is missing",
            input);
    return treeMetadata.getMaterializationExecPath().orElse(treeArtifact.getExecPath());
  }

  /**
   * Returns the symlink to be planted in the output tree for artifacts that are prefetched into a
   * different location.
   *
   * <p>Some artifacts (notably, those created by {@code ctx.actions.symlink}) are materialized in
   * the output tree as a symlink to another artifact, as indicated by the {@link
   * FileArtifactValue#getMaterializationExecPath()} field in their (or their parent tree
   * artifact's) metadata.
   */
  @Nullable
  private Symlink maybeGetSymlink(
      ActionInput input, FileArtifactValue metadata, MetadataSupplier metadataSupplier)
      throws IOException, InterruptedException {
    if (input instanceof TreeFileArtifact) {
      // Check whether the entire tree artifact should be prefetched into a separate location.
      SpecialArtifact treeArtifact = ((TreeFileArtifact) input).getParent();
      FileArtifactValue treeMetadata =
          checkNotNull(
              metadataSupplier.getMetadata(treeArtifact),
              "input %s belongs to a tree artifact whose metadata is missing",
              input);
      return maybeGetSymlink(treeArtifact, treeMetadata, metadataSupplier);
    }
    PathFragment execPath = input.getExecPath();
    PathFragment materializationExecPath = metadata.getMaterializationExecPath().orElse(execPath);
    if (!materializationExecPath.equals(execPath)) {
      return Symlink.of(execPath, materializationExecPath);
    }
    return null;
  }

  private Path resolveOneSymlink(Path path) throws IOException {
    var targetPathFragment = path.readSymbolicLink();
    if (targetPathFragment.isAbsolute()) {
      return path.getFileSystem().getPath(targetPathFragment);
    } else {
      return checkNotNull(path.getParentDirectory()).getRelative(targetPathFragment);
    }
  }

  private Path maybeResolveSymlink(Path path) throws IOException {
    // Potentially resolves a symlink to its target path. This differs from
    // Path#resolveSymbolicLinks() that:
    //   1. Path#resolveSymbolicLinks() checks each segment of the path, but we assume there is no
    //      intermediate symlink because they should've been already normalized for outputs.
    //   2. In case of dangling symlink, we return the target path instead of throwing
    //      FileNotFoundException because we want to download output to that target path.
    var maxAttempt = 32;
    while (path.isSymbolicLink() && maxAttempt-- > 0) {
      var resolvedPath = resolveOneSymlink(path);
      if (resolvedPath.asFragment().equals(path.asFragment())) {
        throw new FileSymlinkLoopException(path.asFragment());
      }
      path = resolvedPath;
    }
    if (maxAttempt <= 0) {
      throw new FileSymlinkLoopException(path.asFragment());
    }
    return path;
  }

  private Completable downloadFileNoCheckRx(
      ActionExecutionMetadata action,
      DirectoryContext dirCtx,
      Path path,
      @Nullable Path treeRoot,
      ActionInput actionInput,
      FileArtifactValue metadata,
      Priority priority) {
    // If the path to be prefetched is a non-dangling symlink, prefetch its target path instead.
    // Note that this only applies to symlinks created by spawns (or, currently, with the internal
    // version of BwoB); symlinks created in-process through an ActionFileSystem should have already
    // been resolved into their materializationExecPath in maybeGetSymlink.
    try {
      if (treeRoot != null) {
        var treeRootRelativePath = path.relativeTo(treeRoot);
        treeRoot = maybeResolveSymlink(treeRoot);
        path = treeRoot.getRelative(treeRootRelativePath);
      } else {
        path = maybeResolveSymlink(path);
      }
    } catch (IOException e) {
      return Completable.error(e);
    }

    Path finalTreeRoot = treeRoot;
    Path finalPath = path;

    AtomicBoolean completed = new AtomicBoolean(false);
    Completable download =
        Completable.using(
                tempPathGenerator::generateTempPath,
                tempPath ->
                    toCompletable(
                            () ->
                                doDownloadFile(
                                    action,
                                    reporter,
                                    tempPath,
                                    finalPath.relativeTo(execRoot),
                                    metadata,
                                    priority),
                            directExecutor())
                        .doOnComplete(
                            () -> {
                              finalizeDownload(dirCtx, finalTreeRoot, tempPath, finalPath);
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
                  if (error instanceof CacheNotFoundException) {
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

  private void finalizeDownload(
      DirectoryContext dirCtx, @Nullable Path treeRoot, Path tmpPath, Path finalPath)
      throws IOException {
    Path parentDir = checkNotNull(finalPath.getParentDirectory());

    if (treeRoot != null) {
      checkState(parentDir.startsWith(treeRoot));

      // Create intermediate tree artifact directories.
      // In order to minimize filesystem calls when prefetching multiple files into the same tree,
      // find the closest existing ancestor directory and only create its descendants.
      Deque<Path> dirs = new ArrayDeque<>();
      for (Path dir = parentDir; ; dir = dir.getParentDirectory()) {
        dirs.push(dir);
        // The very last pushed directory already exists, but we still need to make it writable
        // in case we previously prefetched into it and made it nonwritable.
        if (dir.equals(treeRoot) || dir.exists()) {
          break;
        }
      }
      while (!dirs.isEmpty()) {
        Path dir = dirs.pop();
        // Create directory or make existing directory writable.
        // We know with certainty that the directory belongs to a tree artifact.
        dirCtx.createOrSetWritable(dir, /* isDefinitelyTreeDir= */ true);
      }
    } else {
      // Temporarily make the parent directory writable if needed.
      // We don't know with certainty that the directory does not belong to a tree artifact; it
      // could if the fetched file is a non-tree artifact nested inside a tree artifact, or a
      // tree artifact inside a fileset (see b/254844173 for the latter).
      dirCtx.createOrSetWritable(parentDir, /* isDefinitelyTreeDir= */ false);
    }

    // Set output permissions on files (tree subdirectories are handled in DirectoryContext#close),
    // matching the behavior of SkyframeActionExecutor#checkOutputs for artifacts produced by local
    // actions.
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
        Completable.defer(
            () -> {
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

  public void finalizeAction(Action action, OutputMetadataStore outputMetadataStore)
      throws IOException, InterruptedException {
    List<Artifact> outputsToDownload = new ArrayList<>();
    for (Artifact output : action.getOutputs()) {
      if (outputMetadataStore.artifactOmitted(output)) {
        continue;
      }

      var metadata = outputMetadataStore.getOutputMetadata(output);
      if (!metadata.isRemote()) {
        continue;
      }

      if (output.isTreeArtifact()) {
        var children = outputMetadataStore.getTreeArtifactChildren((SpecialArtifact) output);
        for (var file : children) {
          if (remoteOutputChecker.shouldDownloadOutput(file)) {
            outputsToDownload.add(file);
          }
        }
      } else {
        if (remoteOutputChecker.shouldDownloadOutput(output)) {
          outputsToDownload.add(output);
        }
      }
    }

    if (!outputsToDownload.isEmpty()) {
      try (var s = Profiler.instance().profile(ProfilerTask.REMOTE_DOWNLOAD, "Download outputs")) {
        getFromFuture(
            prefetchFiles(
                action, outputsToDownload, outputMetadataStore::getOutputMetadata, Priority.HIGH));
      }
    }
  }

  public void flushOutputTree() throws InterruptedException {
    downloadCache.awaitInProgressTasks();
  }

  public ImmutableSet<ActionInput> getMissingActionInputs() {
    return ImmutableSet.copyOf(missingActionInputs);
  }

  public RemoteOutputChecker getRemoteOutputChecker() {
    return remoteOutputChecker;
  }
}
