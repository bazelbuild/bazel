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

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.common.util.concurrent.Futures.immediateCancelledFuture;
import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;
import static com.google.devtools.build.lib.remote.util.Utils.waitForBulkTransfer;
import static com.google.devtools.build.lib.util.StringEncoding.unicodeToInternal;
import static com.google.devtools.build.lib.util.StringUtilities.bytesCountToDisplayString;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.remote.common.BulkTransferException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.vfs.DetailedIOException;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.OverlayFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SymlinkTargetType;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyFunctionException;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InterruptedIOException;
import java.io.OutputStream;
import java.nio.channels.SeekableByteChannel;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/**
 * A file system that overlays the native file system with an in-memory file system for the
 * "external" directory, which contains the contents of external repositories.
 *
 * <p>Each external repository can either be materialized to the native file system or kept in
 * memory in a {@link RemoteInMemoryFileSystem}, with file contents downloaded from the remote
 * cache on access. Symlinks may point from either backing into the other (e.g. from an in-memory
 * repo into a materialized repo or the workspace, or vice versa); resolving such chains correctly
 * is handled by {@link OverlayFileSystem}, which canonicalizes paths against the combined view
 * before dispatching to whichever backing owns the canonical path.
 *
 * <p>Paths outside the external directory are delegated wholesale to the native file system.
 */
public final class RemoteExternalOverlayFileSystem extends OverlayFileSystem {
  private final PathFragment externalDirectory;
  private final int externalDirectorySegmentCount;
  private final FileSystem nativeFs;
  private final RemoteInMemoryFileSystem externalFs;
  private final ConcurrentHashMap<String, Future<Void>> materializations =
      new ConcurrentHashMap<>();
  // As long as a repo name appears as a key in this map, the repo contents are available in
  // externalFs.
  private final ConcurrentHashMap<String, String> markerFileContents = new ConcurrentHashMap<>();
  private final Set<String> reposWithLostFiles = ConcurrentHashMap.newKeySet();

  // Per-build information that is set in beforeCommand and cleared in afterCommand.
  @Nullable private CombinedCache cache;
  @Nullable private AbstractActionInputPrefetcher inputPrefetcher;
  @Nullable private Reporter reporter;
  @Nullable private String buildRequestId;
  @Nullable private String commandId;
  @Nullable private MemoizingEvaluator evaluator;
  @Nullable private Duration remoteCacheTtl;
  @Nullable private ExecutorService materializationExecutor;

  public RemoteExternalOverlayFileSystem(PathFragment externalDirectory, FileSystem nativeFs) {
    super(nativeFs.getDigestFunction());
    this.externalDirectory = externalDirectory;
    this.externalDirectorySegmentCount = externalDirectory.segmentCount();
    this.nativeFs = nativeFs;
    this.externalFs = new RemoteInMemoryFileSystem(nativeFs.getDigestFunction());
  }

  public void beforeCommand(
      CombinedCache cache,
      AbstractActionInputPrefetcher inputPrefetcher,
      Reporter reporter,
      String buildRequestId,
      String commandId,
      MemoizingEvaluator evaluator,
      Duration remoteCacheTtl) {
    checkState(
        this.cache == null
            && this.inputPrefetcher == null
            && this.reporter == null
            && this.buildRequestId == null
            && this.commandId == null
            && this.evaluator == null
            && this.remoteCacheTtl == null
            && this.materializationExecutor == null);
    this.cache = cache;
    this.inputPrefetcher = inputPrefetcher;
    this.reporter = reporter;
    this.buildRequestId = buildRequestId;
    this.commandId = commandId;
    this.evaluator = evaluator;
    this.remoteCacheTtl = remoteCacheTtl;
    this.materializationExecutor =
        Executors.newThreadPerTaskExecutor(
            Thread.ofVirtual().name("remote-repo-materialization-", 0).factory());
  }

  public void afterCommand() {
    // While mayCacheResolution restricts the canonicalization cache to in-memory repo contents
    // that can't go stale within a command, this file system lives as long as the server. Clear
    // the cache between commands to reclaim memory and as a defense against staleness bugs.
    clearCanonicalizationCache();
    if (cache == null) {
      // Not all commands cause beforeCommand to be called, but afterCommand is called
      // unconditionally.
      return;
    }
    this.cache = null;
    this.inputPrefetcher = null;
    this.reporter = null;
    this.buildRequestId = null;
    this.commandId = null;
    this.remoteCacheTtl = null;
    // Materializations happen synchronously and upon request by other repo rules, so there is no
    // reason to await their orderly completion in afterCommand.
    materializationExecutor.shutdownNow();
    materializationExecutor = null;
    // Clean up the in-memory contents of materialized repos to save memory, or those that need to
    // be refetched to recover files that the remote cache has lost. This wouldn't be safe to do
    // eagerly as ongoing repo rule evaluations may still refer to the in-memory content and
    // refetching is not atomic.
    materializations.forEach(
        1,
        (repoName, materializationState) ->
            materializationState.state() == Future.State.SUCCESS
                    || reposWithLostFiles.contains(repoName)
                ? repoName
                : null,
        this::evictInMemoryRepo);
    invalidateRepoDirectories(evaluator, reposWithLostFiles);
    reposWithLostFiles.clear();
    this.evaluator = null;
  }

  /** Removes the contents of the given repo from the in-memory overlay file system. */
  private void evictInMemoryRepo(String repoName) {
    var repoDir = externalDirectory.getChild(repoName);
    // A repo that is evicted due to lost files has no native mirror, so cached canonicalization
    // results under it are stale (for materialized repos this is merely conservative).
    invalidatePrefix(repoDir);
    try {
      externalFs.deleteTree(repoDir);
    } catch (IOException e) {
      throw new IllegalStateException("In-memory file system is not expected to throw", e);
    }
    materializations.remove(repoName);
    markerFileContents.remove(repoName);
  }

  /** Invalidates the {@link SkyFunctions#REPOSITORY_DIRECTORY} nodes of the given repos. */
  private static void invalidateRepoDirectories(
      MemoizingEvaluator evaluator, Set<String> repoNames) {
    if (repoNames.isEmpty()) {
      return;
    }
    evaluator.delete(
        k ->
            k.functionName().equals(SkyFunctions.REPOSITORY_DIRECTORY)
                && repoNames.contains(((RepositoryName) k.argument()).getName()));
  }

  /**
   * Injects the given remote contents, possibly prefetching some files, and returns true on
   * success.
   */
  public boolean injectRemoteRepo(RepositoryName repo, Tree remoteContents, String markerFile)
      throws IOException, InterruptedException {
    var repoDir = externalDirectory.getChild(repo.getName());
    deleteTree(repoDir);
    var unused = delete(externalDirectory.getChild(repo.getMarkerFileName()));
    var childMap =
        remoteContents.getChildrenList().stream()
            .collect(
                toImmutableMap(cache.digestUtil::compute, directory -> directory, (a, b) -> a));
    var filesToPrefetch = new ArrayList<PathFragment>();
    injectRecursively(
        externalFs,
        repoDir,
        remoteContents.getRoot(),
        childMap,
        filesToPrefetch::add,
        Instant.now().plus(remoteCacheTtl));
    try {
      // TODO: This prefetches a large number of small files. Investigate whether BatchReadBlobs
      // would be more efficient.
      prefetch(filesToPrefetch);
    } catch (BulkTransferException e) {
      if (e.allCausedByCacheNotFoundException()) {
        // The cache has lost the .bzl files, which should be treated just like a cache miss.
        invalidatePrefix(repoDir);
        externalFs.deleteTree(repoDir);
        return false;
      }
      throw e;
    }
    // Create the repo directory on disk so that readdir reflects the overlaid state of the external
    // directory.
    nativeFs.createDirectoryAndParents(repoDir);
    // Keep the marker file contents in memory so that it can be written out when the repo is
    // materialized. This doubles as a presence marker for the in-memory repo contents.
    markerFileContents.put(repo.getName(), markerFile);
    return true;
  }

  private static void injectRecursively(
      RemoteInMemoryFileSystem fs,
      PathFragment path,
      Directory dir,
      ImmutableMap<Digest, Directory> childMap,
      Consumer<PathFragment> filesToPrefetch,
      Instant expirationTime)
      throws IOException {
    fs.createDirectoryAndParents(path);
    for (var file : dir.getFilesList()) {
      var filePath = path.getRelative(unicodeToInternal(file.getName()));
      if (shouldPrefetch(filePath)) {
        filesToPrefetch.accept(filePath);
      }
      fs.injectFile(
          filePath,
          // Using the *WithMaterializationData variant ensures that the file benefits from the
          // FileContentsProxy optimization to avoid widespread invalidation when it is
          // materialized later, even if expiration times aren't relevant (depends on the usage
          // of the lease extension).
          FileArtifactValue.createForRemoteFileWithMaterializationData(
              DigestUtil.toBinaryDigest(file.getDigest()),
              file.getDigest().getSizeBytes(),
              /* locationIndex= */ 1,
              expirationTime,
              /* inMemoryOutput= */ false));
      fs.setExecutable(filePath, file.getIsExecutable());
      // The RE API does not track whether a file is readable or writable. We choose to make all
      // files readable and not writable to ensure that other repo rules can't accidentally modify
      // the cached repo.
      fs.setWritable(filePath, false);
    }
    for (var symlink : dir.getSymlinksList()) {
      fs.createSymbolicLink(
          path.getRelative(unicodeToInternal(symlink.getName())),
          PathFragment.create(unicodeToInternal(symlink.getTarget())));
    }
    for (var subdirNode : dir.getDirectoriesList()) {
      var subdirPath = path.getRelative(unicodeToInternal(subdirNode.getName()));
      var subdir = childMap.get(subdirNode.getDigest());
      if (subdir == null) {
        throw new IOException(
            "Directory %s with digest %s not found in tree"
                .formatted(subdirPath, subdirNode.getDigest().getHash()));
      }
      injectRecursively(fs, subdirPath, subdir, childMap, filesToPrefetch, expirationTime);
    }
  }

  /**
   * Materializes the given external repository to the native file system if it hasn't been
   * materialized yet. This method blocks until the materialization is complete.
   *
   * <p>This should only be used for cases in which the given repo is accessed non-hermetically,
   * such as when another repo rule that depends on its files executes a command. Selective reads by
   * Bazel or local actions are handled automatically by the file system or {@link
   * AbstractActionInputPrefetcher}.
   */
  public void ensureMaterialized(RepositoryName repo, ExtendedEventHandler reporter)
      throws IOException, InterruptedException {
    if (!markerFileContents.containsKey(repo.getName())) {
      // The repo has not been injected into the in-memory file system.
      return;
    }
    var unused =
        getFromFuture(
            materializations.computeIfAbsent(
                repo.getName(),
                unusedRepoName ->
                    materializationExecutor.submit(
                        () -> {
                          doMaterialize(repo, reporter);
                          return null;
                        })));
  }

  private void doMaterialize(RepositoryName repo, ExtendedEventHandler reporter)
      throws IOException, InterruptedException {
    reporter.handle(Event.debug("Materializing remote repo %s".formatted(repo)));
    var repoPath = externalDirectory.getChild(repo.getName());
    var remoteRepo = externalFs.getPath(repoPath);
    var walkResult = walk(remoteRepo);
    for (var directory : walkResult.directories()) {
      nativeFs.getPath(directory).createDirectory();
    }
    prefetch(walkResult.files());
    // Create symlinks last as some platforms don't allow creating a symlink to a non-existent
    // target.
    prefetch(walkResult.symlinks());

    // After the repo has been copied, atomically materialize the marker file. This ensures that the
    // repo doesn't have to be refetched after the next server restart.
    var markerFile = nativeFs.getPath(externalDirectory.getChild(repo.getMarkerFileName()));
    var markerFileSibling =
        nativeFs.getPath(externalDirectory.getChild(repo.getMarkerFileName() + ".tmp"));
    FileSystemUtils.writeContentAsLatin1(
        markerFileSibling, markerFileContents.remove(repo.getName()));
    markerFileSibling.renameTo(markerFile);
  }

  private void prefetch(List<PathFragment> paths) throws IOException, InterruptedException {
    var unused =
        getFromFuture(
            inputPrefetcher.prefetchFilesInterruptibly(
                /* action= */ null,
                Lists.transform(paths, ActionInputHelper::fromPath),
                actionInput -> getInMemoryMetadata(actionInput.getExecPath()),
                ActionInputPrefetcher.Priority.CRITICAL,
                ActionInputPrefetcher.Reason.INPUTS));
  }

  /** Returns the metadata of a file or symlink in the in-memory file system. */
  private FileArtifactValue getInMemoryMetadata(PathFragment path) throws IOException {
    var status = externalFs.stat(path, /* followSymlinks= */ false);
    if (!status.isSymbolicLink()) {
      return ((RemoteInMemoryFileSystem.RemoteInMemoryFileInfo) status).getMetadata();
    }
    return FileArtifactValue.createForUnresolvedSymlink(externalFs.getPath(path));
  }

  /**
   * Informs the FS that no cache is available and in-memory repos can no longer be used.
   *
   * <p>Must not be called while accessing external repos.
   */
  public void notifyNoCacheAvailable(MemoizingEvaluator evaluator) {
    checkState(materializationExecutor == null, "must not be called when active");
    var reposToDiscard = ImmutableSet.copyOf(markerFileContents.keySet());
    reposToDiscard.forEach(this::evictInMemoryRepo);
    invalidateRepoDirectories(evaluator, reposToDiscard);
    clearCanonicalizationCache();
  }

  private record WalkResult(
      List<PathFragment> files, List<PathFragment> symlinks, List<PathFragment> directories) {}

  private static WalkResult walk(Path root) throws IOException {
    var result = new WalkResult(new ArrayList<>(), new ArrayList<>(), new ArrayList<>());
    walk(root, result);
    return result;
  }

  private static void walk(Path root, WalkResult result) throws IOException {
    for (var dirent : root.readdir(Symlinks.NOFOLLOW)) {
      var fromChild = root.getChild(dirent.getName());
      switch (dirent.getType()) {
        case FILE -> result.files.add(fromChild.asFragment());
        case SYMLINK -> result.symlinks.add(fromChild.asFragment());
        case DIRECTORY -> {
          result.directories.add(fromChild.asFragment());
          walk(fromChild, result);
        }
        default -> throw new IOException("Unsupported file type: " + dirent);
      }
    }
  }

  /** Whether the file with the given path should be materialized eagerly when injecting a repo. */
  private static boolean shouldPrefetch(PathFragment path) {
    // .bzl files are typically small and the loads between them can form complex DAGs that can only
    // be discovered layer by layer, so prefetching is worthwhile to reduce the number of sequential
    // cache requests.
    // The REPO.bazel file, if present, is a dependency of any package and will thus have to be
    // fetched anyway.
    return path.getFileExtension().equals("bzl") || path.getBaseName().equals("REPO.bazel");
  }

  @Override
  public FileSystem getHostFileSystem() {
    return nativeFs.getHostFileSystem();
  }

  // Only paths strictly below the external directory are overlaid; everything else is delegated
  // wholesale to the native file system.

  @Override
  @Nullable
  protected FileSystem wholesaleDelegate(PathFragment path) {
    if (path.startsWith(externalDirectory) && !path.equals(externalDirectory)) {
      return null;
    }
    return nativeFs;
  }

  @Override
  protected boolean mayCacheResolution(PathFragment path) {
    // Only the contents of unmaterialized in-memory repos are guaranteed not to change through
    // channels that bypass this file system (native repo directories are mutated by processes run
    // by repo rules, and workspace paths reached through symlinks by the user). The external
    // directory and its ancestors are managed by Bazel and stable, and caching them provides the
    // anchor for the in-memory repo entries below them.
    return externalDirectory.startsWith(path) || fsForPath(path) == externalFs;
  }

  // Always mirror tree deletions to the underlying native file system to support bazel clean and
  // repository refetching.

  @Override
  public void deleteTree(PathFragment path) throws IOException {
    invalidatePrefix(path);
    nativeFs.deleteTree(path);
    externalFs.deleteTree(path);
  }

  @Override
  public void deleteTreesBelow(PathFragment dir) throws IOException {
    invalidatePrefix(dir);
    nativeFs.deleteTreesBelow(dir);
    externalFs.deleteTreesBelow(dir);
  }

  /**
   * Returns the file system that owns the given path.
   *
   * <p>The path must be canonical, except that the last segment (for operations that operate on
   * the symlink itself) or the segments below the repo directory (for operations that create new
   * paths, which never target in-memory repos) may not be. Canonical paths that a symlink chain
   * caused to escape the external directory are owned by the native file system.
   */
  private FileSystem fsForPath(PathFragment path) {
    if (path.startsWith(externalDirectory) && !path.equals(externalDirectory)) {
      String repoName = path.getSegment(externalDirectorySegmentCount);
      var hasBeenInjected = markerFileContents.containsKey(repoName);
      var hasBeenMaterialized =
          materializations.getOrDefault(repoName, immediateCancelledFuture()).state()
              == Future.State.SUCCESS;
      if (hasBeenInjected && !hasBeenMaterialized) {
        // The repo may have been deleted due to refetching. Clean up in-memory state if that is the
        // case.
        if (externalFs.getPath(externalDirectory.getChild(repoName)).exists()) {
          return externalFs;
        }
        materializations.remove(repoName);
        markerFileContents.remove(repoName);
      }
      // Fall back to the native file system if the repo has been materialized, deleted, or never
      // injected.
    }
    return nativeFs;
  }

  @Override
  @Nullable
  protected FileStatus statNofollow(PathFragment path) throws IOException {
    return fsForPath(path).statIfFound(path, /* followSymlinks= */ false);
  }

  @Override
  protected PathFragment readSymlinkNofollow(PathFragment path) throws IOException {
    return fsForPath(path).readSymbolicLink(path);
  }

  @Override
  protected Collection<Dirent> readdirNofollow(PathFragment path) throws IOException {
    return fsForPath(path).readdir(path, /* followSymlinks= */ false);
  }

  @Override
  protected byte[] getDigestNofollow(PathFragment path) throws IOException {
    var fs = fsForPath(path);
    if (fs == externalFs) {
      // The in-memory file system stores digests as metadata; its inherited getDigest would
      // attempt to hash the file contents, which it doesn't have.
      return externalFs.getFastDigest(path);
    }
    return fs.getDigest(path);
  }

  @Override
  @Nullable
  protected byte[] getFastDigestNofollow(PathFragment path) throws IOException {
    return fsForPath(path).getFastDigest(path);
  }

  @Override
  protected boolean deleteNofollow(PathFragment path) throws IOException {
    return fsForPath(path).delete(path);
  }

  @Override
  protected void renameToNofollow(PathFragment sourcePath, PathFragment targetPath)
      throws IOException {
    var sourceFs = fsForPath(sourcePath);
    if (sourceFs != fsForPath(targetPath)) {
      // Renaming between the in-memory and native backing is not supported, just like renaming
      // across file system boundaries isn't in general.
      throw new IOException(
          "%s -> %s (Cross-device link)".formatted(sourcePath, targetPath));
    }
    sourceFs.renameTo(sourcePath, targetPath);
  }

  @Override
  protected void createSymbolicLinkNofollow(
      PathFragment linkPath, PathFragment targetFragment, SymlinkTargetType type)
      throws IOException {
    fsForPath(linkPath).createSymbolicLink(linkPath, targetFragment, type);
  }

  @Override
  protected void setLastModifiedTimeNofollow(PathFragment path, long newTime) throws IOException {
    fsForPath(path).setLastModifiedTime(path, newTime);
  }

  @Override
  protected boolean isReadableNofollow(PathFragment path) throws IOException {
    return fsForPath(path).isReadable(path);
  }

  @Override
  protected boolean isWritableNofollow(PathFragment path) throws IOException {
    return fsForPath(path).isWritable(path);
  }

  @Override
  protected boolean isExecutableNofollow(PathFragment path) throws IOException {
    return fsForPath(path).isExecutable(path);
  }

  @Override
  protected void setReadableNofollow(PathFragment path, boolean readable) throws IOException {
    fsForPath(path).setReadable(path, readable);
  }

  @Override
  protected void setWritableNofollow(PathFragment path, boolean writable) throws IOException {
    fsForPath(path).setWritable(path, writable);
  }

  @Override
  protected void setExecutableNofollow(PathFragment path, boolean executable) throws IOException {
    fsForPath(path).setExecutable(path, executable);
  }

  @Override
  protected void chmodNofollow(PathFragment path, int mode) throws IOException {
    fsForPath(path).chmod(path, mode);
  }

  @Override
  public boolean supportsModifications(PathFragment path) {
    return fsForPath(path).supportsModifications(path);
  }

  @Override
  public boolean supportsSymbolicLinksNatively(PathFragment path) {
    return fsForPath(path).supportsSymbolicLinksNatively(path);
  }

  @Override
  public boolean supportsHardLinksNatively(PathFragment path) {
    return fsForPath(path).supportsHardLinksNatively(path);
  }

  @Override
  public boolean mayBeCaseOrNormalizationInsensitive() {
    return nativeFs.mayBeCaseOrNormalizationInsensitive();
  }

  @Override
  public boolean createDirectory(PathFragment path) throws IOException {
    return fsForPath(path).createDirectory(path);
  }

  @Override
  public void createDirectoryAndParents(PathFragment path) throws IOException {
    fsForPath(path).createDirectoryAndParents(path);
  }

  @Override
  public InputStream getInputStream(PathFragment path) throws IOException {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      return delegate.getInputStream(path);
    }
    var resolvedPath = canonicalize(path);
    if (fsForPath(resolvedPath) != externalFs) {
      return nativeFs.getInputStream(resolvedPath);
    }
    if (shouldPrefetch(resolvedPath)) {
      // .bzl and REPO.bazel files are prefetched to the native file system during injection, but
      // only if they are regular files; a symlink with such a name is kept in the in-memory
      // overlay only. Since the path is canonical at this point, it refers to a prefetched file.
      return nativeFs.getInputStream(resolvedPath);
    }
    return downloadInputStream(resolvedPath);
  }

  private RemoteActionExecutionContext makeRemoteContext(PathFragment relativePath) {
    String repoName = relativePath.subFragment(0, 1).getBaseName();
    var metadata = TracingMetadataUtils.buildMetadata(buildRequestId, commandId, repoName);
    // Files in the remote external repo that Bazel reads are worth writing through to the
    // disk cache, as they are likely to be read again on future cold builds.
    return RemoteActionExecutionContext.create(metadata)
        .withReadCachePolicy(RemoteActionExecutionContext.CachePolicy.ANY_CACHE)
        .withWriteCachePolicy(RemoteActionExecutionContext.CachePolicy.ANY_CACHE);
  }

  /**
   * Downloads the contents of the in-memory file with the given canonical path from the remote
   * cache.
   */
  private InputStream downloadInputStream(PathFragment path) throws IOException {
    var relativePath = path.relativeTo(externalDirectory);
    var status = externalFs.stat(path, /* followSymlinks= */ false);
    if (!(status instanceof RemoteInMemoryFileSystem.RemoteInMemoryFileInfo info)) {
      // The canonical path denotes a directory.
      throw new IOException(path.getPathString() + " (Is a directory)");
    }
    reporter.post(
        new ExtendedEventHandler.FetchProgress() {
          @Override
          public String getResourceIdentifier() {
            return relativePath.getPathString();
          }

          @Override
          public String getProgress() {
            return "(%s)".formatted(bytesCountToDisplayString(info.getSize()));
          }

          @Override
          public boolean isFinished() {
            return false;
          }
        });
    var digest = DigestUtil.buildDigest(info.getMetadata().getDigest(), info.getSize());
    try {
      var contentFuture =
          cache.downloadBlob(
              makeRemoteContext(relativePath), path.getPathString(), /* execPath= */ null, digest);
      waitForBulkTransfer(ImmutableList.of(contentFuture));
      return new ByteArrayInputStream(contentFuture.get());
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new InterruptedIOException("interrupted while waiting for remote file transfer");
    } catch (BulkTransferException e) {
      if (e.allCausedByCacheNotFoundException()) {
        reposWithLostFiles.add(relativePath.getSegment(0));
        throw new DetailedIOException(
            "%s/%s with digest %s is no longer available in the remote cache"
                .formatted(
                    externalDirectory.getBaseName(), relativePath, DigestUtil.toString(digest)),
            e,
            FailureDetails.Filesystem.Code.REMOTE_FILE_EVICTED,
            SkyFunctionException.Transience.TRANSIENT);
      }
      throw e;
    } catch (ExecutionException e) {
      throw new IllegalStateException("waitForBulkTransfer should have thrown", e);
    } finally {
      reporter.post(
          new ExtendedEventHandler.FetchProgress() {
            @Override
            public String getResourceIdentifier() {
              return relativePath.getPathString();
            }

            @Override
            public String getProgress() {
              return "";
            }

            @Override
            public boolean isFinished() {
              return true;
            }
          });
    }
  }

  @Override
  public SeekableByteChannel createReadWriteByteChannel(PathFragment path) throws IOException {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      return delegate.createReadWriteByteChannel(path);
    }
    var resolvedPath = canonicalizeParent(path);
    return fsForPath(resolvedPath).createReadWriteByteChannel(resolvedPath);
  }

  @Override
  public OutputStream getOutputStream(PathFragment path, boolean append, boolean internal)
      throws IOException {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      return delegate.getOutputStream(path, append, internal);
    }
    var resolvedPath = canonicalizeParent(path);
    return fsForPath(resolvedPath).getOutputStream(resolvedPath, append, internal);
  }

  @Override
  public void createFSDependentHardLink(PathFragment linkPath, PathFragment originalPath)
      throws IOException {
    fsForPath(originalPath).createFSDependentHardLink(linkPath, originalPath);
  }

  @Override
  public File getIoFile(PathFragment path) {
    return fsForPath(path).getIoFile(path);
  }

  @Override
  public java.nio.file.Path getNioPath(PathFragment path) {
    return fsForPath(path).getNioPath(path);
  }

  @Override
  public String getFileSystemType(PathFragment path) {
    return fsForPath(path).getFileSystemType(path);
  }

  @Override
  public byte[] getxattr(PathFragment path, String name, boolean followSymlinks)
      throws IOException {
    var delegate = wholesaleDelegate(path);
    if (delegate != null) {
      return delegate.getxattr(path, name, followSymlinks);
    }
    var resolvedPath = followSymlinks ? canonicalize(path) : canonicalizeParent(path);
    var fs = fsForPath(resolvedPath);
    if (fs == externalFs) {
      // In-memory files don't have extended attributes.
      return null;
    }
    return fs.getxattr(resolvedPath, name, followSymlinks);
  }

  @Override
  public void createHardLink(PathFragment linkPath, PathFragment originalPath) throws IOException {
    fsForPath(linkPath).createHardLink(linkPath, originalPath);
  }

  @Override
  public void prefetchPackageAsync(PathFragment path, int maxDirs) {
    fsForPath(path).prefetchPackageAsync(path, maxDirs);
  }

  @Override
  public PathFragment createTempDirectory(PathFragment parent, String prefix) throws IOException {
    return fsForPath(parent).createTempDirectory(parent, prefix);
  }
}
