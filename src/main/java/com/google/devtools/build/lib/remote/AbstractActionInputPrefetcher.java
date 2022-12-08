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

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.remote.util.RxFutures.toCompletable;
import static com.google.devtools.build.lib.remote.util.RxFutures.toListenableFuture;
import static com.google.devtools.build.lib.remote.util.RxUtils.mergeBulkTransfer;
import static com.google.devtools.build.lib.remote.util.RxUtils.toTransferResult;
import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
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
import com.google.devtools.build.lib.remote.util.AsyncTaskCache;
import com.google.devtools.build.lib.remote.util.RxUtils.TransferResult;
import com.google.devtools.build.lib.remote.util.TempPathGenerator;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import io.reactivex.rxjava3.core.Completable;
import io.reactivex.rxjava3.core.Flowable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.regex.Pattern;

/**
 * Abstract implementation of {@link ActionInputPrefetcher} which implements the orchestration of
 * prefeching multiple inputs so subclasses can focus on prefetching / downloading single input.
 */
public abstract class AbstractActionInputPrefetcher implements ActionInputPrefetcher {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final AsyncTaskCache.NoResult<Path> downloadCache = AsyncTaskCache.NoResult.create();
  private final TempPathGenerator tempPathGenerator;
  protected final Set<Artifact> outputsAreInputs = Sets.newConcurrentHashSet();

  protected final Path execRoot;
  protected final ImmutableList<Pattern> patternsToDownload;

  private static class Context {
    private final Set<Path> nonWritableDirs = Sets.newConcurrentHashSet();

    public void addNonWritableDir(Path dir) {
      nonWritableDirs.add(dir);
    }

    public void finalizeContext() throws IOException {
      for (Path path : nonWritableDirs) {
        path.setWritable(false);
      }
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
      Path execRoot,
      TempPathGenerator tempPathGenerator,
      ImmutableList<Pattern> patternsToDownload) {
    this.execRoot = execRoot;
    this.tempPathGenerator = tempPathGenerator;
    this.patternsToDownload = patternsToDownload;
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
    Map<SpecialArtifact, List<TreeFileArtifact>> trees = new HashMap<>();
    List<ActionInput> files = new ArrayList<>();
    for (ActionInput input : inputs) {
      if (input instanceof Artifact && ((Artifact) input).isSourceArtifact()) {
        continue;
      }

      if (input instanceof TreeFileArtifact) {
        TreeFileArtifact treeFile = (TreeFileArtifact) input;
        SpecialArtifact treeArtifact = treeFile.getParent();
        trees.computeIfAbsent(treeArtifact, unusedKey -> new ArrayList<>()).add(treeFile);
        continue;
      }

      files.add(input);
    }

    Context context = new Context();

    Flowable<TransferResult> treeDownloads =
        Flowable.fromIterable(trees.entrySet())
            .flatMapSingle(
                entry ->
                    toTransferResult(
                        prefetchInputTreeOrSymlink(
                            context,
                            metadataProvider,
                            entry.getKey(),
                            entry.getValue(),
                            priority)));

    Flowable<TransferResult> fileDownloads =
        Flowable.fromIterable(files)
            .flatMapSingle(
                input ->
                    toTransferResult(
                        prefetchInputFileOrSymlink(context, metadataProvider, input, priority)));

    Flowable<TransferResult> transfers = Flowable.merge(treeDownloads, fileDownloads);
    Completable prefetch =
        Completable.using(
                () -> context, ctx -> mergeBulkTransfer(transfers), Context::finalizeContext)
            .onErrorResumeNext(this::onErrorResumeNext);
    return toListenableFuture(prefetch);
  }

  private Completable prefetchInputTreeOrSymlink(
      Context context,
      MetadataProvider provider,
      SpecialArtifact tree,
      List<TreeFileArtifact> treeFiles,
      Priority priority)
      throws IOException {

    PathFragment execPath = tree.getExecPath();

    FileArtifactValue treeMetadata = provider.getMetadata(tree);
    // TODO(tjgq): Only download individual files that were requested within the tree.
    // This isn't straightforward because multiple tree artifacts may share the same output tree
    // when a ctx.actions.symlink is involved.
    if (treeMetadata == null || !canDownloadAnyTreeFiles(treeFiles, treeMetadata)) {
      return Completable.complete();
    }

    PathFragment prefetchExecPath = treeMetadata.getMaterializationExecPath().orElse(execPath);

    Completable prefetch =
        prefetchInputTree(context, provider, prefetchExecPath, treeFiles, treeMetadata, priority);

    // If prefetching to a different path, plant a symlink into it.
    if (!prefetchExecPath.equals(execPath)) {
      Completable prefetchAndSymlink =
          prefetch.doOnComplete(() -> createSymlink(execPath, prefetchExecPath));
      return downloadCache.executeIfNot(execRoot.getRelative(execPath), prefetchAndSymlink);
    }

    return prefetch;
  }

  private boolean canDownloadAnyTreeFiles(
      Iterable<TreeFileArtifact> treeFiles, FileArtifactValue metadata) {
    for (TreeFileArtifact treeFile : treeFiles) {
      if (canDownloadFile(treeFile.getPath(), metadata)) {
        return true;
      }
    }
    return false;
  }

  private boolean shouldDownloadAnyTreeFiles(
      Iterable<TreeFileArtifact> treeFiles, FileArtifactValue metadata) {
    for (TreeFileArtifact treeFile : treeFiles) {
      if (shouldDownloadFile(treeFile.getPath(), metadata)) {
        return true;
      }
    }
    return false;
  }

  private Completable prefetchInputTree(
      Context context,
      MetadataProvider provider,
      PathFragment execPath,
      List<TreeFileArtifact> treeFiles,
      FileArtifactValue treeMetadata,
      Priority priority) {
    Path treeRoot = execRoot.getRelative(execPath);
    HashMap<TreeFileArtifact, Path> treeFileTmpPathMap = new HashMap<>();

    Flowable<TransferResult> transfers =
        Flowable.fromIterable(treeFiles)
            .flatMapSingle(
                treeFile -> {
                  FileArtifactValue metadata = provider.getMetadata(treeFile);

                  Path tempPath = tempPathGenerator.generateTempPath();
                  treeFileTmpPathMap.put(treeFile, tempPath);

                  return toTransferResult(
                      toCompletable(
                          () ->
                              doDownloadFile(tempPath, treeFile.getExecPath(), metadata, priority),
                          directExecutor()));
                });

    AtomicBoolean completed = new AtomicBoolean();
    Completable download =
        mergeBulkTransfer(transfers)
            .doOnComplete(
                () -> {
                  HashSet<Path> dirs = new HashSet<>();

                  // Even though the root directory for a tree artifact is created prior to action
                  // execution, we might be prefetching to a different directory that doesn't yet
                  // exist (when FileArtifactValue#getMaterializationExecPath() is present).
                  // In any case, we need to make it writable to move files into it.
                  createWritableDirectory(treeRoot);
                  dirs.add(treeRoot);

                  for (Map.Entry<TreeFileArtifact, Path> entry : treeFileTmpPathMap.entrySet()) {
                    TreeFileArtifact treeFile = entry.getKey();
                    Path tempPath = entry.getValue();

                    Path path = treeRoot.getRelative(treeFile.getParentRelativePath());
                    Path dir = treeRoot;
                    for (String segment : treeFile.getParentRelativePath().segments()) {
                      dir = dir.getRelative(segment);
                      if (dir.equals(path)) {
                        break;
                      }
                      if (dirs.add(dir)) {
                        createWritableDirectory(dir);
                      }
                    }
                    checkState(dir.equals(path));
                    finalizeDownload(context, tempPath, path);
                  }

                  for (Path dir : dirs) {
                    // Change permission of all directories of a tree artifact to 0555 (files are
                    // changed inside {@code finalizeDownload}) in order to match the behaviour when
                    // the tree artifact is generated locally. In that case, permission of all files
                    // and directories inside a tree artifact is changed to 0555 within {@code
                    // checkOutputs()}.
                    dir.chmod(0555);
                  }

                  completed.set(true);
                })
            .doFinally(
                () -> {
                  if (!completed.get()) {
                    for (Map.Entry<TreeFileArtifact, Path> entry : treeFileTmpPathMap.entrySet()) {
                      deletePartialDownload(entry.getValue());
                    }
                  }
                });
    return downloadCache.executeIfNot(
        treeRoot,
        Completable.defer(
            () -> {
              if (shouldDownloadAnyTreeFiles(treeFiles, treeMetadata)) {
                return download;
              }
              return Completable.complete();
            }));
  }

  private Completable prefetchInputFileOrSymlink(
      Context context, MetadataProvider metadataProvider, ActionInput input, Priority priority)
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

    PathFragment prefetchExecPath = metadata.getMaterializationExecPath().orElse(execPath);

    Completable prefetch =
        downloadFileNoCheckRx(context, execRoot.getRelative(prefetchExecPath), metadata, priority);

    // If prefetching to a different path, plant a symlink into it.
    if (!prefetchExecPath.equals(execPath)) {
      Completable prefetchAndSymlink =
          prefetch.doOnComplete(() -> createSymlink(execPath, prefetchExecPath));
      return downloadCache.executeIfNot(execRoot.getRelative(execPath), prefetchAndSymlink);
    }

    return prefetch;
  }

  /**
   * Downloads file into the {@code path} with its metadata.
   *
   * <p>The file will be written into a temporary file and moved to the final destination after the
   * download finished.
   */
  private Completable downloadFileRx(
      Context context, Path path, FileArtifactValue metadata, Priority priority) {
    if (!canDownloadFile(path, metadata)) {
      return Completable.complete();
    }
    return downloadFileNoCheckRx(context, path, metadata, priority);
  }

  private Completable downloadFileNoCheckRx(
      Context context, Path path, FileArtifactValue metadata, Priority priority) {
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
                          finalizeDownload(context, tempPath, finalPath);
                          completed.set(true);
                        }),
            tempPath -> {
              if (!completed.get()) {
                deletePartialDownload(tempPath);
              }
            },
            // Set eager=false here because we want cleanup the download *after* upstream is
            // disposed.
            /* eager= */ false);

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
  public void downloadFile(Path path, FileArtifactValue metadata)
      throws IOException, InterruptedException {
    getFromFuture(downloadFileAsync(path.asFragment(), metadata, Priority.CRITICAL));
  }

  protected ListenableFuture<Void> downloadFileAsync(
      PathFragment path, FileArtifactValue metadata, Priority priority) {
    Context context = new Context();
    return toListenableFuture(
        Completable.using(
            () -> context,
            ctx ->
                downloadFileRx(context, execRoot.getFileSystem().getPath(path), metadata, priority),
            Context::finalizeContext));
  }

  private void finalizeDownload(Context context, Path tmpPath, Path path) throws IOException {
    Path parentDir = path.getParentDirectory();
    // In case the parent directory of the destination is not writable, temporarily change it to
    // writable. b/254844173.
    if (parentDir != null && !parentDir.isWritable()) {
      context.addNonWritableDir(parentDir);
      parentDir.setWritable(true);
    }

    // The permission of output file is changed to 0555 after action execution. We manually change
    // the permission here for the downloaded file to keep this behaviour consistent.
    tmpPath.chmod(0555);
    FileSystemUtils.moveFile(tmpPath, path);
  }

  private void deletePartialDownload(Path path) {
    try {
      path.delete();
    } catch (IOException e) {
      logger.atWarning().withCause(e).log(
          "Failed to delete output file after incomplete download: %s", path);
    }
  }

  private void createWritableDirectory(Path dir) throws IOException {
    dir.createDirectory();
    dir.setWritable(true);
  }

  private void createSymlink(PathFragment linkPath, PathFragment targetPath) throws IOException {
    Path link = execRoot.getRelative(linkPath);
    Path target = execRoot.getRelative(targetPath);
    // Delete the link path if it already exists.
    // This will happen for output directories, which get created before the action runs.
    link.delete();
    link.createSymbolicLink(target);
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

  @SuppressWarnings({"CheckReturnValue", "FutureReturnValueIgnored"})
  public void finalizeAction(Action action, MetadataHandler metadataHandler) {
    List<Artifact> inputsToDownload = new ArrayList<>();
    List<Artifact> outputsToDownload = new ArrayList<>();

    for (Artifact output : action.getOutputs()) {
      if (outputsAreInputs.remove(output)) {
        inputsToDownload.add(output);
      }

      if (output.isTreeArtifact()) {
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
      prefetchFiles(inputsToDownload, metadataHandler, Priority.HIGH);
    }

    if (!outputsToDownload.isEmpty()) {
      prefetchFiles(outputsToDownload, metadataHandler, Priority.LOW);
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
}
