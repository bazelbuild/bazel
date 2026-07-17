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
import static com.google.common.util.concurrent.Futures.immediateFailedFuture;
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.remote.util.RxFutures.toCompletable;
import static com.google.devtools.build.lib.remote.util.RxFutures.toListenableFuture;
import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;
import static com.google.devtools.build.lib.remote.util.Utils.mergeBulkTransfer;
import static io.reactivex.rxjava3.core.Completable.concat;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.ActionOutputDirectoryHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValueWithMaterializationData;
import com.google.devtools.build.lib.actions.FileContentsProxy;
import com.google.devtools.build.lib.actions.FileStateType;
import com.google.devtools.build.lib.actions.cache.OutputMetadataStore;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.remote.util.AsyncTaskCache;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.util.TempPathGenerator;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSymlinkLoopException;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.OutputPermissions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import io.reactivex.rxjava3.core.Completable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
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

  // The exec root isn't known when the prefetcher is created as its path depends on the name set in
  // WORKSPACE.
  private final Supplier<Path> execRootSupplier;
  protected final Path outputBase;
  protected final RemoteOutputChecker remoteOutputChecker;

  @Nullable protected final ActionOutputDirectoryHelper outputDirectoryHelper;

  /** The state of a directory tracked by {@link DirectoryTracker}, as explained below. */
  enum DirectoryState {
    PERMANENTLY_WRITABLE,
    TEMPORARILY_WRITABLE,
    OUTPUT_PERMISSIONS
  }

  /**
   * Tracks directory permissions to minimize filesystem operations.
   *
   * <p>Throughout the prefetcher, {@link Path#setWritable} and {@link Path#chmod} calls on output
   * directories must go through the methods in this class.
   */
  private final class DirectoryTracker {
    private final ConcurrentHashMap<Path, DirectoryState> directoryStateMap =
        new ConcurrentHashMap<>();

    /**
     * Marks a directory as temporarily writable.
     *
     * <p>A temporarily writable directory may have its output permissions set by a later call to
     * {@link #setOutputPermissions}, unless {@link #setPermanentlyWritable} is called in the
     * interim.
     */
    void setTemporarilyWritable(Path dir) throws IOException {
      setWritable(dir, DirectoryState.TEMPORARILY_WRITABLE);
    }

    /**
     * Marks a directory as permanently writable.
     *
     * <p>A permanently writable directory will never have its output permissions set by a later
     * call to {@link #setOutputPermissions}.
     */
    void setPermanentlyWritable(Path dir) throws IOException {
      setWritable(dir, DirectoryState.PERMANENTLY_WRITABLE);
    }

    private void setWritable(Path dir, DirectoryState newState) throws IOException {
      // External repo paths (which live directly under the output base) are not build outputs and
      // don't need output permission management. Check this first (comparing fragments, since the
      // dir may be on the host file system while the output base is on an overlay) so that the exec
      // root, which is only resolvable during the loading phase and later, is not resolved during
      // external repo materialization.
      if (dir.asFragment()
              .startsWith(
                  outputBase.getRelative(LabelConstants.EXTERNAL_REPOSITORY_LOCATION).asFragment())
          || !dir.startsWith(execRoot())) {
        return;
      }
      AtomicReference<IOException> caughtException = new AtomicReference<>();

      directoryStateMap.compute(
          dir,
          (unusedKey, oldState) -> {
            if (!forceRefetch(dir)
                && (oldState == DirectoryState.TEMPORARILY_WRITABLE
                    || oldState == DirectoryState.PERMANENTLY_WRITABLE)) {
              // Already writable, but must potentially upgrade from temporary to permanent.
              return newState == DirectoryState.PERMANENTLY_WRITABLE ? newState : oldState;
            }
            try {
              if (outputDirectoryHelper != null) {
                outputDirectoryHelper.createOutputDirectory(dir, execRoot());
              } else {
                dir.createDirectoryAndParents();
              }
              dir.setWritable(true);
            } catch (IOException e) {
              caughtException.set(e);
              return oldState;
            }
            return newState;
          });

      if (caughtException.get() != null) {
        throw caughtException.get();
      }
    }

    /**
     * Sets the output permissions on a directory.
     *
     * <p>If {@link #setPermanentlyWritable} has been previously called on this directory, or if no
     * {@link #setTemporarilyWritable} call has intervened since the last call to {@link
     * #setOutputPermissions}, this is a no-op. Otherwise, the output permissions are set.
     */
    void setOutputPermissions(Path dir) throws IOException {
      AtomicReference<IOException> caughtException = new AtomicReference<>();

      directoryStateMap.compute(
          dir,
          (unusedKey, oldState) -> {
            if (!forceRefetch(dir)
                && (oldState == DirectoryState.OUTPUT_PERMISSIONS
                    || oldState == DirectoryState.PERMANENTLY_WRITABLE)) {
              // Either the output permissions have already been set, or we're not changing the
              // permissions ever again.
              return oldState;
            }
            try {
              dir.chmod(outputPermissions.getPermissionsMode());
            } catch (IOException e) {
              caughtException.set(e);
              return oldState;
            }
            return DirectoryState.OUTPUT_PERMISSIONS;
          });

      if (caughtException.get() != null) {
        throw caughtException.get();
      }
    }
  }

  private final DirectoryTracker directoryTracker = new DirectoryTracker();

  /**
   * A symlink in the output tree that either points to another artifact's absolute path or
   * represents an unresolved symlink with its target path preserved verbatim.
   */
  record Symlink(Path linkPath, PathFragment targetPath) {
    Symlink {
      checkNotNull(linkPath, "linkPath");
      checkNotNull(targetPath, "targetPath");
      checkArgument(
          !linkPath.asFragment().equals(targetPath), "linkPath and targetPath must differ");
    }

    Path resolveOne() throws IOException {
      return resolveOneSymlink(linkPath, targetPath);
    }
  }

  protected AbstractActionInputPrefetcher(
      Reporter reporter,
      Supplier<Path> execRootSupplier,
      Path outputBase,
      TempPathGenerator tempPathGenerator,
      RemoteOutputChecker remoteOutputChecker,
      @Nullable ActionOutputDirectoryHelper outputDirectoryHelper,
      OutputPermissions outputPermissions) {
    this.reporter = reporter;
    this.execRootSupplier = Suppliers.memoize(execRootSupplier::get);
    this.outputBase = outputBase;
    this.tempPathGenerator = tempPathGenerator;
    this.remoteOutputChecker = remoteOutputChecker;
    this.outputDirectoryHelper = outputDirectoryHelper;
    this.outputPermissions = outputPermissions;
  }

  /**
   * Returns the exec root. May only be called once the workspace name is known, i.e. not before the
   * loading phase. Code paths reached during external repository materialization must use {@link
   * #outputBase} instead.
   */
  protected Path execRoot() {
    return execRootSupplier.get();
  }

  /**
   * Returns whether the given path lives under the external repository root. Unlike {@link
   * #execRoot()}, this can be evaluated during external repository materialization, where the exec
   * root isn't available yet.
   */
  private boolean isUnderExternalRepoRoot(Path path) {
    return path.asFragment()
        .startsWith(
            outputBase.getRelative(LabelConstants.EXTERNAL_REPOSITORY_LOCATION).asFragment());
  }

  /**
   * Resolves an exec path to an absolute path, avoiding evaluation of execRoot() if possible as it
   * isn't available during server startup. This logic is unique to Bazel 8.x as it still names the
   * exec root based on the name set in WORKSPACE, which is gone from HEAD and Bazel 9.x.
   */
  private Path resolveExecPath(PathFragment execPath) {
    if (execPath.isAbsolute()) {
      return outputBase.getFileSystem().getPath(execPath);
    }
    if (execPath.startsWith(LabelConstants.EXTERNAL_REPOSITORY_LOCATION)) {
      return outputBase.getRelative(execPath);
    }
    return execRoot().getRelative(execPath);
  }

  private boolean shouldDownloadFile(Path path, FileArtifactValue metadata) throws IOException {
    var stat = path.statIfFound();
    if (stat == null) {
      return true;
    }

    // If an action output is stale, Skyframe will delete it prior to action execution. However,
    // this doesn't apply to spawn outputs that aren't action outputs, or to files in external repos
    // that are remote repo contents cache hits. To avoid incorrectly reusing one such stale file,
    // check for its up-to-dateness here.
    if (stat.getSize() != metadata.getSize()) {
      return true;
    }
    FileContentsProxy contentsProxy;
    try {
      contentsProxy = metadata.getContentsProxy();
    } catch (UnsupportedOperationException e) {
      contentsProxy = null;
    }
    if (contentsProxy != null && contentsProxy.equals(FileContentsProxy.create(stat))) {
      return false;
    }

    byte[] digest = path.getFastDigest();
    if (digest == null) {
      digest = path.getDigest();
    }
    return !Arrays.equals(digest, metadata.getDigest());
  }

  protected abstract boolean canDownloadFile(Path path, FileArtifactValue metadata);

  /**
   * If true, then all previously acquired knowledge of the file system state of this path (e.g. the
   * existence of tree artifact directories or previously downloaded files) must be discarded.
   */
  protected abstract boolean forceRefetch(Path path);

  /**
   * Downloads file to the given path via its metadata.
   *
   * @param tempPath the temporary path which the input should be written to.
   */
  protected abstract ListenableFuture<Void> doDownloadFile(
      @Nullable ActionExecutionMetadata action,
      Reporter reporter,
      Path tempPath,
      PathFragment execPath,
      FileArtifactValue metadata,
      Priority priority,
      Reason reason)
      throws IOException;

  protected void prefetchVirtualActionInput(VirtualActionInput input) throws IOException {}

  /**
   * Fetches remotely stored action outputs and stores them under their path in the output base.
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
      @Nullable ActionExecutionMetadata action,
      Iterable<? extends ActionInput> inputs,
      MetadataSupplier metadataSupplier,
      Priority priority,
      Reason reason) {
    List<ActionInput> files = new ArrayList<>();

    for (ActionInput input : inputs) {
      // Source artifacts in the main repo don't need to be fetched.
      if (input instanceof Artifact artifact
          && artifact.isSourceArtifact()
          && artifact.getArtifactOwner().getLabel().getRepository().isMain()) {
        continue;
      }

      // Skip empty tree artifacts (non-empty tree artifacts should have already been expanded).
      if (input.isDirectory()) {
        continue;
      }

      files.add(input);
    }

    if (files.isEmpty()) {
      return immediateVoidFuture();
    }

    // Collect the set of directories whose output permissions must be set at the end of this call.
    // This responsibility cannot lie with the downloading of an individual file, because multiple
    // files may be concurrently downloaded into the same directory within a single call to
    // prefetchFiles, and two concurrent calls to prefetchFiles may prefetch the same file. In the
    // latter case, the second call will have its downloads deduplicated against the first call, but
    // it must still synchronize on the output permissions having been set.
    Set<Path> dirsWithOutputPermissions = Sets.newConcurrentHashSet();

    // Using plain futures to avoid RxJava overheads.
    List<ListenableFuture<Void>> transfers = new ArrayList<>(files.size());
    try (var s = Profiler.instance().profile("compose prefetches")) {
      for (var file : files) {
        transfers.add(
            prefetchFile(
                action, dirsWithOutputPermissions, metadataSupplier, file, priority, reason));
      }
    }

    ListenableFuture<Void> mergedTransfer;
    try (var s = Profiler.instance().profile("mergeBulkTransfer")) {
      mergedTransfer = mergeBulkTransfer(transfers);
    }

    return Futures.transformAsync(
        mergedTransfer,
        unused -> {
          try {
            // Set output permissions on tree artifact subdirectories, matching the behavior of
            // SkyframeActionExecutor#checkOutputs for artifacts produced by local actions.
            for (Path dir : dirsWithOutputPermissions) {
              directoryTracker.setOutputPermissions(dir);
            }
          } catch (IOException e) {
            return immediateFailedFuture(e);
          }
          return immediateVoidFuture();
        },
        directExecutor());
  }

  private ListenableFuture<Void> prefetchFile(
      @Nullable ActionExecutionMetadata action,
      Set<Path> dirsWithOutputPermissions,
      MetadataSupplier metadataSupplier,
      ActionInput input,
      Priority priority,
      Reason reason) {
    try {
      if (input instanceof VirtualActionInput virtualActionInput) {
        prefetchVirtualActionInput(virtualActionInput);
        return immediateVoidFuture();
      }

      PathFragment execPath = input.getExecPath();

      FileArtifactValue metadata = metadataSupplier.getMetadata(input);
      if (metadata == null) {
        return immediateVoidFuture();
      }
      Path inputPath = resolveExecPath(execPath);

      if (metadata.getType() == FileStateType.DIRECTORY) {
        // Tree artifacts have already been expanded into their children, so this is a source
        // directory. If it lies in an external repo backed by the remote repo contents cache, its
        // contents may only be available in memory and must be materialized to the local file
        // system for local actions to access them.
        if (inputPath.getFileSystem() instanceof SubtreeMaterializer subtreeMaterializer) {
          subtreeMaterializer.ensureSubtreeMaterialized(inputPath.asFragment());
        }
        return immediateVoidFuture();
      }

      var symlinks = getSymlinks(input, inputPath, metadata, metadataSupplier);
      // On Windows, the type of symlink depends on the target file and the target may have to
      // exist, so we plant symlinks in reverse order and only after any download has completed.
      var plantSymlinks = concat(Lists.transform(symlinks.reverse(), this::plantSymlink));
      if (!canDownloadFile(inputPath, metadata)) {
        // If the artifact is a declared ("unresolved") symlink, it can't be "downloaded", but the
        // symlink logic above creates it.
        return toListenableFuture(plantSymlinks);
      }

      if (!symlinks.isEmpty()) {
        // Symlink may track the parent of a TreeFileArtifact, so the parent relative path has to be
        // translated relative to it.
        var parentRelativePath = inputPath.relativeTo(symlinks.getFirst().linkPath());
        inputPath =
            inputPath
                .getFileSystem()
                .getPath(
                    symlinks.getLast().resolveOne().asFragment().getRelative(parentRelativePath));
      }

      @Nullable PathFragment treeRootExecPath = maybeGetTreeRoot(action, input, metadataSupplier);

      Completable result =
          downloadFileNoCheckRx(
                  action,
                  inputPath,
                  treeRootExecPath != null ? resolveExecPath(treeRootExecPath) : null,
                  dirsWithOutputPermissions,
                  input,
                  metadata,
                  priority,
                  reason)
              .onErrorResumeNext(
                  t -> {
                    if (t instanceof CacheNotFoundException cacheNotFoundException) {
                      // Only the symlink itself is guaranteed to be an input to the action, so
                      // report its path for rewinding.
                      cacheNotFoundException.setExecPath(input.getExecPath());
                      return Completable.error(cacheNotFoundException);
                    }
                    return Completable.error(t);
                  })
              .andThen(plantSymlinks);

      return toListenableFuture(result);
    } catch (IOException | InterruptedException e) {
      return immediateFailedFuture(e);
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
  private PathFragment maybeGetTreeRoot(
      ActionExecutionMetadata action, ActionInput input, MetadataSupplier metadataSupplier)
      throws IOException, InterruptedException {
    if (!(input instanceof TreeFileArtifact treeFile)) {
      return null;
    }
    SpecialArtifact treeArtifact = treeFile.getParent();
    FileArtifactValue treeMetadata = metadataSupplier.getMetadata(treeArtifact);
    if (treeMetadata == null) {
      if (!treeFile.isChildOfDeclaredDirectory() && action.getOutputs().contains(treeFile)) {
        // If this file is produced by an action template, the full tree artifact metadata might
        // not be available yet. However, we know with certainty that the file is not materialized
        // as a symlink.
        return null;
      }
      throw new IllegalStateException(
          String.format("input %s belongs to a tree artifact whose metadata is missing", treeFile));
    }
    return treeMetadata.getMaterializationExecPath().orElse(treeArtifact.getExecPath());
  }

  /**
   * Returns the symlinks to be planted on disk for inputs that are prefetched into a different
   * location, ordered from the input path to the final target.
   *
   * <p>Some artifacts (notably, those created by {@code ctx.actions.symlink}) are materialized in
   * the output tree as a symlink to another artifact, as indicated by the {@link
   * FileArtifactValue#getMaterializationExecPath()} field in their (or their parent tree artifact's)
   * metadata.
   *
   * <p>Paths in external repos backed by the remote repo contents cache may be (part of) a chain of
   * symlinks created by the repo rule, which has to be reproduced verbatim on disk.
   */
  private ImmutableList<Symlink> getSymlinks(
      ActionInput input,
      Path inputPath,
      FileArtifactValue metadata,
      MetadataSupplier metadataSupplier)
      throws IOException, InterruptedException {
    if (input instanceof TreeFileArtifact treeFile) {
      SpecialArtifact treeArtifact = treeFile.getParent();
      FileArtifactValue treeMetadata = metadataSupplier.getMetadata(treeArtifact);
      if (treeMetadata == null) {
        // There are two cases where tree metadata is legitimately not available:
        // (1) If the file is the output of an action expanded from an action template. In this
        //     case, the symlink optimization is intentionally not supported.
        // (2) If the file is part of an input fileset. In this case, a symlink has already been
        //     created, but we're currently unable to prefetch the file(s) it points to.
        // TODO: b/401575099 - Treating fileset more like runfiles could make the tree metadata
        //  available for case (2).
        return ImmutableList.of();
      }
      return getSymlinks(treeArtifact, treeArtifact.getPath(), treeMetadata, metadataSupplier);
    }
    if ((metadata.isRemote() || metadata.getType() == FileStateType.SYMLINK)
        && isUnderExternalRepoRoot(inputPath)) {
      // A path in an external repo, e.g. a source artifact consumed by an action or a file
      // prefetched during the materialization of an external repo. It may be (part of) a chain of
      // symlinks created by the repo rule, which has to be reproduced verbatim on disk.
      var symlinkChain = ImmutableList.<Symlink>builder();
      FileStatus stat;
      Path currentPath = inputPath;
      var maxAttempt = 32;
      while ((stat = currentPath.statIfFound(Symlinks.NOFOLLOW)) != null && stat.isSymbolicLink()) {
        if (maxAttempt-- == 0) {
          throw new FileSymlinkLoopException(inputPath.asFragment());
        }
        var symlink = new Symlink(currentPath, currentPath.readSymbolicLink());
        symlinkChain.add(symlink);
        currentPath = symlink.resolveOne();
      }
      return symlinkChain.build();
    }
    PathFragment execPath = input.getExecPath();
    PathFragment materializationExecPath = metadata.getMaterializationExecPath().orElse(execPath);
    if (materializationExecPath.equals(execPath)) {
      return ImmutableList.of();
    }
    return ImmutableList.of(
        new Symlink(inputPath, resolveExecPath(materializationExecPath).asFragment()));
  }

  private static Path resolveOneSymlink(Path path, @Nullable PathFragment targetPathFragment)
      throws IOException {
    if (targetPathFragment == null) {
      targetPathFragment = path.readSymbolicLink();
    }
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
      var resolvedPath = resolveOneSymlink(path, /* targetPathFragment= */ null);
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
      @Nullable ActionExecutionMetadata action,
      Path path,
      @Nullable Path treeRoot,
      Set<Path> dirsWithOutputPermissions,
      ActionInput actionInput,
      FileArtifactValue metadata,
      Priority priority,
      Reason reason) {
    // If the path to be prefetched is a non-dangling symlink, prefetch its target path instead.
    // Note that this only applies to symlinks created by spawns (or, currently, with the internal
    // version of BwoB); symlinks created in-process through an ActionFileSystem should have already
    // been resolved into their materializationExecPath in getSymlinks.
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

    if (actionInput instanceof Artifact && ((Artifact) actionInput).isChildOfDeclaredDirectory()) {
      // Arrange for the output permissions to be set on every directory inside the tree artifact.
      // This must be done at assembly time to ensure that the permissions are set before the
      // prefetchFiles call returns, even when the actual downloads are deduplicated against a
      // concurrent call. See finalizeDownload for why we don't do so in other cases.
      checkNotNull(treeRoot);
      for (Path dir = path.getParentDirectory();
          dir.startsWith(treeRoot);
          dir = dir.getParentDirectory()) {
        if (!dirsWithOutputPermissions.add(dir)) {
          break;
        }
      }
    }

    // Downloads should always be written to the "actual" host file system, not any overlays.
    // See the comment on resolveExecPath for the rationale behind the branching below.
    Path finalPath = path.forHostFileSystem();
    PathFragment execPath;
    if (finalPath
        .asFragment()
        .startsWith(
            outputBase.getRelative(LabelConstants.EXTERNAL_REPOSITORY_LOCATION).asFragment())) {
      execPath = finalPath.asFragment().relativeTo(outputBase.asFragment());
    } else {
      execPath = finalPath.asFragment().relativeTo(execRoot().asFragment());
    }

    Completable download =
        usingTempPath(
            (tempPath, alreadyDeleted) ->
                toCompletable(
                        () ->
                            doDownloadFile(
                                action,
                                reporter,
                                tempPath.forHostFileSystem(),
                                execPath,
                                metadata,
                                priority,
                                reason),
                        directExecutor())
                    .doOnComplete(
                        () -> {
                          finalizeDownload(
                              metadata,
                              tempPath.forHostFileSystem(),
                              finalPath,
                              dirsWithOutputPermissions);
                          alreadyDeleted.set(true);
                        })
                    .onErrorResumeNext(
                        error -> {
                          if (error instanceof CacheNotFoundException) {
                            return Completable.error(error);
                          }

                          // Treat other download error as CacheNotFoundException so that Bazel can
                          // correctly rewind the action/build.
                          var digest =
                              DigestUtil.buildDigest(metadata.getDigest(), metadata.getSize());
                          return Completable.error(new CacheNotFoundException(digest, execPath));
                        }));

    return downloadCache.execute(
        finalPath,
        Completable.defer(
            () -> {
              if (shouldDownloadFile(finalPath, metadata)) {
                return download;
              }
              return Completable.complete();
            }),
        forceRefetch(finalPath));
  }

  private void finalizeDownload(
      FileArtifactValue metadata, Path tmpPath, Path finalPath, Set<Path> dirsWithOutputPermissions)
      throws IOException {
    Path parentDir = checkNotNull(finalPath.getParentDirectory());

    // External repo paths live directly under the output base, not the exec root. Classify by the
    // external directory (rather than the exec root, which isn't resolvable during external repo
    // materialization) and compare as fragments since the exec root may be located on a file system
    // overlaying the host file system where the download is written to.
    if (!finalPath
        .asFragment()
        .startsWith(
            outputBase.getRelative(LabelConstants.EXTERNAL_REPOSITORY_LOCATION).asFragment())) {
      // Ensure the parent directory exists and is writable. We cannot rely on this precondition to
      // have been established by the execution of the owning action in a previous invocation, since
      // the output tree may have been externally modified in between invocations.
      if (dirsWithOutputPermissions.contains(parentDir)) {
        // The file belongs to a tree artifact created by an action that declared an output
        // directory (as opposed to an action template expansion). The output permissions should be
        // set on the parent directory after prefetching.
        directoryTracker.setTemporarilyWritable(parentDir);
      } else {
        // One of the following must apply:
        //   (1) The file does not belong to a tree artifact.
        //   (2) The file belongs to a tree artifact created by an action template expansion.
        // In case (1), the parent directory is a package or a subdirectory of a package, and should
        // remain writable. In case (2), even though we arguably ought to set the output permissions
        // on the parent directory to match local execution, we choose not to do it and avoid the
        // additional implementation complexity required to detect a race condition between
        // concurrent calls touching the same directory.
        directoryTracker.setPermanentlyWritable(parentDir);
      }
    } else {
      parentDir.createDirectoryAndParents();
    }

    // Set output permissions on files, matching the behavior of SkyframeActionExecutor#checkOutputs
    // for artifacts produced by local actions.
    tmpPath.chmod(outputPermissions.getPermissionsMode());
    FileSystemUtils.moveFile(tmpPath, finalPath);
    if (metadata instanceof RemoteFileArtifactValueWithMaterializationData remote) {
      remote.setContentsProxy(FileContentsProxy.create(finalPath.stat()));
    }
  }

  private interface TaskWithTempPath {
    Completable run(Path tempPath, AtomicBoolean alreadyDeleted);
  }

  /**
   * Runs a task with a temporary path.
   *
   * <p>The temporary path will be deleted once the task is done. Set {@code alreadyDeleted} to
   * signal that deletion is no longer needed.
   */
  private Completable usingTempPath(TaskWithTempPath task) {
    AtomicBoolean alreadyDeleted = new AtomicBoolean(false);
    return Completable.using(
        tempPathGenerator::generateTempPath,
        (tempPath) -> task.run(tempPath, alreadyDeleted),
        tempPath -> {
          if (!alreadyDeleted.get()) {
            deletePartialDownload(tempPath);
          }
        },
        // Clean up after the upstream is disposed to ensure tempPath won't be touched further.
        /* eager= */ false);
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
    Path linkPath = symlink.linkPath().forHostFileSystem();
    return downloadCache.execute(
        linkPath,
        Completable.defer(
            () -> {
              if (isUnderExternalRepoRoot(symlink.linkPath())) {
                // If the symlink is a source file in an external repo, its parent directory may not
                // exist yet.
                checkNotNull(linkPath.getParentDirectory()).createDirectoryAndParents();
              }
              // Delete the link path if it already exists. This is the case for tree artifacts,
              // whose root directory is created before the action runs.
              linkPath.delete();
              linkPath.createSymbolicLink(symlink.targetPath());
              return Completable.complete();
            }),
        forceRefetch(linkPath));
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
      if (!canDownloadFile(output.getPath(), metadata)) {
        continue;
      }

      if (output.isTreeArtifact()) {
        outputMetadataStore
            .getTreeArtifactValue((SpecialArtifact) output)
            .getChildValues()
            .forEach(
                (child, childMetadata) -> {
                  if (remoteOutputChecker.shouldDownloadOutput(
                      child, (RemoteFileArtifactValue) childMetadata)) {
                    outputsToDownload.add(child);
                  }
                });
      } else {
        if (remoteOutputChecker.shouldDownloadOutput(output, (RemoteFileArtifactValue) metadata)) {
          outputsToDownload.add(output);
        }
      }
    }

    if (!outputsToDownload.isEmpty()) {
      try (var s = Profiler.instance().profile(ProfilerTask.REMOTE_DOWNLOAD, "Download outputs")) {
        getFromFuture(
            prefetchFiles(
                action,
                outputsToDownload,
                outputMetadataStore::getOutputMetadata,
                Priority.HIGH,
                Reason.OUTPUTS));
      }
    }
  }

  public void flushOutputTree() throws InterruptedException {
    downloadCache.awaitInProgressTasks();
  }

  public RemoteOutputChecker getRemoteOutputChecker() {
    return remoteOutputChecker;
  }
}
