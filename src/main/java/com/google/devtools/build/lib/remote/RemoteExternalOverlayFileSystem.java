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
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;
import static com.google.devtools.build.lib.remote.util.BulkTransfers.waitForBulkTransfer;
import static com.google.devtools.build.lib.remote.util.Futures.getFromFuture;
import static com.google.devtools.build.lib.util.StringEncoding.unicodeToInternal;
import static com.google.devtools.build.lib.util.StringUtilities.bytesCountToDisplayString;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.TaskDeduplicator;
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
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSymlinkLoopException;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SymlinkTargetType;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.SkyFunctionException;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InterruptedIOException;
import java.io.OutputStream;
import java.nio.channels.SeekableByteChannel;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/**
 * A file system that overlays the native file system with a {@link RemoteExternalFileSystem} for
 * the "external" directory, which contains the contents of external repositories.
 *
 * <p>Each external repository can either be materialized to the native file system or kept in
 * memory in the {@link RemoteExternalFileSystem}.
 */
public final class RemoteExternalOverlayFileSystem extends FileSystem implements LazyMaterializer {
  private final PathFragment externalDirectory;
  private final int externalDirectorySegmentCount;
  private final FileSystem nativeFs;
  private final RemoteExternalFileSystem externalFs;
  private final TaskDeduplicator<String, Void, Void> materializations = new TaskDeduplicator<>();
  // The names of the repos whose contents have been fully materialized to nativeFs.
  private final Set<String> materializedRepos = ConcurrentHashMap.newKeySet();
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
  @Nullable private ListeningExecutorService materializationExecutor;

  public RemoteExternalOverlayFileSystem(PathFragment externalDirectory, FileSystem nativeFs) {
    super(nativeFs.getDigestFunction());
    this.externalDirectory = externalDirectory;
    this.externalDirectorySegmentCount = externalDirectory.segmentCount();
    this.nativeFs = nativeFs;
    this.externalFs = new RemoteExternalFileSystem(nativeFs.getDigestFunction());
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
        MoreExecutors.listeningDecorator(
            Executors.newThreadPerTaskExecutor(
                Thread.ofVirtual().name("remote-repo-materialization-", 0).factory()));
  }

  public void afterCommand() {
    if (cache == null) {
      // Not all commands cause beforeCommand to be called, but afterCommand is called
      // unconditionally.
      return;
    }

    // Uninterruptibly await the termination of all ongoing materializations to prevent cleanup
    // below from interfering with a user's interrupt of the build.
    materializationExecutor.shutdownNow();
    materializationExecutor.close();

    this.cache = null;
    this.inputPrefetcher = null;
    this.reporter = null;
    this.buildRequestId = null;
    this.commandId = null;
    this.remoteCacheTtl = null;
    this.materializationExecutor = null;
    // Clean up the in-memory contents of materialized repos to save memory, or those that need to
    // be refetched to recover files that the remote cache has lost. This wouldn't be safe to do
    // eagerly as ongoing repo rule evaluations may still refer to the in-memory content and
    // refetching is not atomic.
    materializedRepos.forEach(this::evictInMemoryRepo);
    reposWithLostFiles.forEach(this::evictInMemoryRepo);
    invalidateRepoDirectories(evaluator, reposWithLostFiles);
    reposWithLostFiles.clear();
    this.evaluator = null;
  }

  /** Removes the contents of the given repo from the in-memory overlay file system. */
  private void evictInMemoryRepo(String repoName) {
    try {
      externalFs.deleteTree(externalDirectory.getChild(repoName));
    } catch (IOException e) {
      throw new IllegalStateException("In-memory file system is not expected to throw", e);
    }
    materializedRepos.remove(repoName);
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
    materializedRepos.remove(repo.getName());
    var unused = delete(externalDirectory.getChild(repo.getMarkerFileName()));
    var childMap =
        remoteContents.getChildrenList().stream()
            .collect(
                toImmutableMap(cache.digestUtil::compute, directory -> directory, (a, b) -> a));
    var filesToPrefetch = new LinkedHashSet<PathFragment>();
    var symlinksToPrefetch = new ArrayList<PathFragment>();
    externalFs.createDirectoryAndParents(repoDir.getParentDirectory());
    injectRecursively(
        externalFs,
        repoDir,
        repoDir,
        remoteContents.getRoot(),
        childMap,
        filesToPrefetch::add,
        symlinksToPrefetch::add,
        Instant.now().plus(remoteCacheTtl));
    addSymlinkTargetsToPrefetch(symlinksToPrefetch, filesToPrefetch);
    try {
      // TODO: This prefetches a large number of small files. Investigate whether BatchReadBlobs
      // would be more efficient.
      prefetch(filesToPrefetch);
    } catch (BulkTransferException e) {
      if (e.allCausedByCacheNotFoundException()) {
        // The cache has lost the prefetched files, which should be treated just like a cache miss.
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

  /**
   * Collects the targets of the given symlinks into {@code filesToPrefetch}.
   *
   * <p>Whether a read is served from the native file system is decided by the path it is made
   * through, which for a symlink is not the path of the file that would be prefetched. A symlink
   * that should be prefetched thus requires its target to be materialized, no matter whether the
   * target's own path calls for prefetching.
   *
   * <p>Symlink targets can only be resolved once the entire repo has been injected, which is why
   * this doesn't happen in {@link #injectRecursively}.
   */
  private void addSymlinkTargetsToPrefetch(
      List<PathFragment> symlinks, Set<PathFragment> filesToPrefetch) throws IOException {
    for (var symlink : symlinks) {
      Path target;
      try {
        target = externalFs.getPath(symlink).resolveSymbolicLinks();
      } catch (IOException e) {
        // Dangling symlinks and symlink loops are reproduced verbatim and only fail when read.
        continue;
      }
      if (target.isFile()) {
        filesToPrefetch.add(target.asFragment());
      }
    }
  }

  private static boolean isValidName(String name) {
    return !name.isEmpty()
        && !PathFragment.containsSeparator(name)
        && !PathFragment.containsUplevelReferences(name)
        && PathFragment.isNormalizedRelativePath(name);
  }

  private static void injectRecursively(
      RemoteExternalFileSystem fs,
      PathFragment repoDir,
      PathFragment path,
      Directory dir,
      ImmutableMap<Digest, Directory> childMap,
      Consumer<PathFragment> filesToPrefetch,
      Consumer<PathFragment> symlinksToPrefetch,
      Instant expirationTime)
      throws IOException {
    // The parent directory always exists at this point: the repo's parent is created by
    // injectRemoteRepo and subdirectories are only visited after their parent has been created.
    var unused =
        fs.createDirectory(
            path, dir.getFilesCount() + dir.getSymlinksCount() + dir.getDirectoriesCount());
    for (var file : dir.getFilesList()) {
      String name = unicodeToInternal(file.getName());
      if (!isValidName(name)) {
        throw new IOException("invalid remote repo tree node name: " + name);
      }
      var filePath = path.getRelative(name);
      if (!filePath.startsWith(repoDir)) {
        throw new IOException("Path traversal detected: " + filePath + " is outside " + repoDir);
      }
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
      String name = unicodeToInternal(symlink.getName());
      if (!isValidName(name)) {
        throw new IOException("invalid remote repo tree node name: " + name);
      }
      var linkPath = path.getRelative(name);
      if (!linkPath.startsWith(repoDir)) {
        throw new IOException("Path traversal detected: " + linkPath + " is outside " + repoDir);
      }
      if (shouldPrefetch(linkPath)) {
        symlinksToPrefetch.accept(linkPath);
      }
      String target = unicodeToInternal(symlink.getTarget());
      PathFragment targetFragment = PathFragment.create(target);
      PathFragment resolvedTarget;
      if (targetFragment.isAbsolute()) {
        resolvedTarget = targetFragment;
      } else {
        resolvedTarget = linkPath.getParentDirectory().getRelative(targetFragment);
      }
      if (!resolvedTarget.startsWith(repoDir)) {
        throw new IOException(
            "Path traversal detected: symlink target " + resolvedTarget + " is outside " + repoDir);
      }
      fs.createSymbolicLink(linkPath, targetFragment);
    }
    for (var subdirNode : dir.getDirectoriesList()) {
      String name = unicodeToInternal(subdirNode.getName());
      if (!isValidName(name)) {
        throw new IOException("invalid remote repo tree node name: " + name);
      }
      var subdirPath = path.getRelative(name);
      if (!subdirPath.startsWith(repoDir)) {
        throw new IOException("Path traversal detected: " + subdirPath + " is outside " + repoDir);
      }
      var subdir = childMap.get(subdirNode.getDigest());
      if (subdir == null) {
        throw new IOException(
            "Directory %s with digest %s not found in tree"
                .formatted(subdirPath, subdirNode.getDigest().getHash()));
      }
      injectRecursively(
          fs,
          repoDir,
          subdirPath,
          subdir,
          childMap,
          filesToPrefetch,
          symlinksToPrefetch,
          expirationTime);
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
  @Override
  public void ensureMaterialized(RepositoryName repo, ExtendedEventHandler reporter)
      throws IOException, InterruptedException {
    if (!markerFileContents.containsKey(repo.getName())) {
      // The repo has not been injected into the in-memory file system.
      return;
    }
    var unused =
        getFromFuture(
            materializations.execute(
                repo.getName(),
                /* attributes= */ null,
                /* canJoin= */ unusedAttributes -> true,
                () -> {
                  // Another caller may have finished since the presence check above.
                  if (!markerFileContents.containsKey(repo.getName())) {
                    return immediateVoidFuture();
                  }
                  return materializationExecutor.submit(
                      () -> {
                        doMaterialize(repo, reporter);
                        return null;
                      });
                }));
  }

  private void doMaterialize(RepositoryName repo, ExtendedEventHandler reporter)
      throws IOException, InterruptedException {
    reporter.handle(Event.debug("Materializing remote repo %s".formatted(repo)));
    materializeSubtree(externalDirectory.getChild(repo.getName()));
    materializedRepos.add(repo.getName());

    // After the repo has been copied, atomically materialize the marker file. This ensures that the
    // repo doesn't have to be refetched after the next server restart.
    var markerFile = nativeFs.getPath(externalDirectory.getChild(repo.getMarkerFileName()));
    var markerFileSibling =
        nativeFs.getPath(externalDirectory.getChild(repo.getMarkerFileName() + ".tmp"));
    FileSystemUtils.writeContentAsLatin1(
        markerFileSibling, markerFileContents.remove(repo.getName()));
    markerFileSibling.renameTo(markerFile);
  }

  private void prefetch(Iterable<PathFragment> paths) throws IOException, InterruptedException {
    // These paths may have been prefetched and then deleted again earlier in this invocation, e.g.
    // by an injection whose fetch was subsequently restarted due to memory pressure. The
    // prefetcher's download cache would otherwise consider them downloaded already and not even
    // verify they exist on the local file system.
    inputPrefetcher.invalidateDownloads(paths);
    var unused =
        getFromFuture(
            inputPrefetcher.prefetchFilesInterruptibly(
                /* action= */ null,
                Iterables.transform(paths, ActionInputHelper::fromPath),
                actionInput -> externalFs.getMetadata(actionInput.getExecPath()),
                ActionInputPrefetcher.Priority.CRITICAL,
                ActionInputPrefetcher.Reason.INPUTS));
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
  }

  /**
   * Materializes the subtree rooted at the given path to the native file system if it lies in a
   * repo whose contents are currently only available in memory.
   *
   * <p>This is used to make the files below a source directory action input available to local
   * actions, which access them through the native file system.
   */
  @Override
  public void ensureSubtreeMaterialized(PathFragment path)
      throws IOException, InterruptedException {
    if (fsForPath(path) != externalFs) {
      return;
    }
    materializeSubtree(path);
  }

  private void materializeSubtree(PathFragment path) throws IOException, InterruptedException {
    var files = new LinkedHashSet<PathFragment>();
    var symlinks = new LinkedHashSet<PathFragment>();
    // The path or any of the directories above it may be a symlink. Reproduce these symlinks on the
    // native file system and materialize the subtree at the path they resolve to, as creating the
    // directories along the given path instead would turn the symlinks into regular directories.
    var root = externalFs.getPath(path.subFragment(0, externalDirectorySegmentCount + 1));
    for (String segment : path.subFragment(externalDirectorySegmentCount + 1).segments()) {
      root = root.getChild(segment);
      if (root.isSymbolicLink()) {
        symlinks.add(root.asFragment());
        root = root.resolveSymbolicLinks();
      }
    }
    collectAndCreateDirectories(root, files, symlinks, new HashSet<>());
    prefetch(files);
    // Create symlinks last as some platforms don't allow creating a symlink to a non-existent
    // target.
    prefetch(symlinks);
  }

  private void collectAndCreateDirectories(
      Path dir, Set<PathFragment> files, Set<PathFragment> symlinks, Set<PathFragment> visitedDirs)
      throws IOException {
    if (!visitedDirs.add(dir.asFragment())) {
      return;
    }
    nativeFs.createDirectoryAndParents(dir.asFragment());
    for (var dirent : dir.readdir(Symlinks.NOFOLLOW)) {
      var child = dir.getChild(dirent.getName());
      switch (dirent.getType()) {
        case FILE -> files.add(child.asFragment());
        case SYMLINK -> {
          symlinks.add(child.asFragment());
          // The symlink chain is reproduced verbatim on the native file system, but its target may
          // lie outside the materialized subtree and has to be materialized as well so that the
          // chain doesn't dangle.
          Path target;
          try {
            target = child.resolveSymbolicLinks();
          } catch (FileNotFoundException | FileSymlinkLoopException e) {
            // Dangling symlinks and symlink loops are reproduced verbatim.
            continue;
          }
          // TODO(#30160): RepositoryUtils.replantSymlinks currently ensures that all symlinks
          // within a remotely cacheable external repo stay within that repo. If that changes, new
          // logic has to be added here to prefetch such files correctly.
          if (target.isDirectory(Symlinks.NOFOLLOW)) {
            collectAndCreateDirectories(target, files, symlinks, visitedDirs);
          } else {
            files.add(target.asFragment());
          }
        }
        case DIRECTORY -> collectAndCreateDirectories(child, files, symlinks, visitedDirs);
        default -> throw new IOException("Unsupported file type: " + dirent);
      }
    }
  }

  /**
   * Whether reads of the given path should be served from the native file system, which requires
   * its contents to be materialized eagerly when injecting a repo.
   *
   * <p>This is decided by the path a read is made through, which for a symlink is not the path of
   * the file that ends up being materialized.
   */
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

  // Always mirror tree deletions to the underlying native file system to support bazel clean and
  // repository refetching.

  @Override
  public void deleteTree(PathFragment path) throws IOException {
    nativeFs.deleteTree(path);
    externalFs.deleteTree(path);
  }

  @Override
  public void deleteTreesBelow(PathFragment dir) throws IOException {
    nativeFs.deleteTreesBelow(dir);
    externalFs.deleteTreesBelow(dir);
  }

  // All other methods delegate to the file system given by this method. It is important to override
  // each non-final FileSystem method to benefit from optimizations implemented in the respective
  // underlying file systems.
  /** A read of a path that is performed on the file system backing the path. */
  private interface Read<T> {
    T apply(FileSystem fs, PathFragment path) throws IOException;
  }

  /**
   * Performs the given read on the backing file system of the path it resolves to.
   *
   * <p>A natively fetched repo can contain symlinks into a repo that is served from memory, e.g.
   * because its repo rule created them from labels of that repo, which also excludes the native repo
   * itself from the cache. The native file system finds such a symlink dangling or, if some files of
   * the target have been prefetched, pointing to an incomplete directory, so a native path in the
   * external directory is read where its symlinks lead. Paths that resolve natively are read as
   * given, since the native file system may resolve raw symlink targets differently.
   *
   * @param followLast whether the read follows a symlink at the end of the path
   */
  private <T> T read(PathFragment path, boolean followLast, Read<T> read) throws IOException {
    FileSystem fs = fsForPath(path);
    if (fs == externalFs || markerFileContents.isEmpty() || !path.startsWith(externalDirectory)) {
      return read.apply(fs, path);
    }
    PathFragment resolved = resolveIntoMemory(path, followLast);
    return resolved != null ? read.apply(externalFs, resolved) : read.apply(fs, path);
  }

  /**
   * Returns the path in a repo served from memory that the given native path below the external
   * directory resolves to through symlinks, or null if it doesn't resolve into memory.
   *
   * <p>Symlinks are followed by their normalized targets, which is how Bazel resolves them
   * everywhere else, but the native file system resolves a target such as {@code dirlink/../file}
   * against the directory {@code dirlink} points to rather than the one it lies in. A symlink that
   * resolves natively is thus only followed if its normalized target is the same file.
   */
  @Nullable
  private PathFragment resolveIntoMemory(PathFragment path, boolean followLast) {
    try {
      PathFragment current = externalDirectory;
      PathFragment remaining = path.relativeTo(externalDirectory);
      int symlinksFollowed = 0;
      while (!remaining.isEmpty()) {
        current = current.getChild(remaining.getSegment(0));
        remaining = remaining.subFragment(1);
        if (fsForPath(current) == externalFs) {
          return current.getRelative(remaining);
        }
        if (remaining.isEmpty() && !followLast) {
          return null;
        }
        FileStatus status = nativeFs.statIfFound(current, /* followSymlinks= */ false);
        if (status == null) {
          return null;
        }
        if (!status.isSymbolicLink()) {
          continue;
        }
        if (++symlinksFollowed > MAX_SYMLINKS) {
          return null;
        }
        PathFragment target = nativeFs.readSymbolicLink(current);
        PathFragment resolvedTarget =
            target.isAbsolute() ? target : current.getParentDirectory().getRelative(target);
        FileStatus nativeStatus = nativeFs.statIfFound(current, /* followSymlinks= */ true);
        if (nativeStatus != null) {
          FileStatus targetStatus = nativeFs.statIfFound(resolvedTarget, /* followSymlinks= */ true);
          if (targetStatus == null || targetStatus.getNodeId() != nativeStatus.getNodeId()) {
            return null;
          }
        }
        if (!resolvedTarget.startsWith(externalDirectory)) {
          return null;
        }
        current = externalDirectory;
        remaining = resolvedTarget.relativeTo(externalDirectory).getRelative(remaining);
      }
      return null;
    } catch (IOException e) {
      // The path doesn't resolve at all, which its own backing file system reports.
      return null;
    }
  }

  private FileSystem fsForPath(PathFragment path) {
    if (path.startsWith(externalDirectory) && !path.equals(externalDirectory)) {
      String repoName = path.getSegment(externalDirectorySegmentCount);
      var hasBeenInjected = markerFileContents.containsKey(repoName);
      var hasBeenMaterialized = materializedRepos.contains(repoName);
      if (hasBeenInjected && !hasBeenMaterialized) {
        // The repo may have been deleted due to refetching. Clean up in-memory state if that is the
        // case.
        boolean exists;
        try {
          exists = externalFs.getPath(externalDirectory.getChild(repoName)).exists();
        } catch (IOException e) {
          // Ignore and treat as if the repo does not exist.
          exists = false;
        }
        if (exists) {
          return externalFs;
        }
        materializedRepos.remove(repoName);
        markerFileContents.remove(repoName);
      }
      // Fall back to the native file system if the repo has been materialized, deleted, or never
      // injected.
    }
    return nativeFs;
  }

  @Override
  public boolean delete(PathFragment path) throws IOException {
    return fsForPath(path).delete(path);
  }

  @Override
  public byte[] getDigest(PathFragment path) throws IOException {
    return read(path, /* followLast= */ true, (fs, p) -> fs.getDigest(p));
  }

  @Nullable
  @Override
  public byte[] getFastDigest(PathFragment path) throws IOException {
    return read(path, /* followLast= */ true, (fs, p) -> fs.getFastDigest(p));
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
    return fsForPath(externalDirectory).mayBeCaseOrNormalizationInsensitive();
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
  public long getFileSize(PathFragment path, boolean followSymlinks) throws IOException {
    return read(path, followSymlinks, (fs, p) -> fs.getFileSize(p, followSymlinks));
  }

  @Override
  public long getLastModifiedTime(PathFragment path, boolean followSymlinks) throws IOException {
    return read(path, followSymlinks, (fs, p) -> fs.getLastModifiedTime(p, followSymlinks));
  }

  @Override
  public void setLastModifiedTime(PathFragment path, long newTime) throws IOException {
    fsForPath(path).setLastModifiedTime(path, newTime);
  }

  @Override
  public FileStatus stat(PathFragment path, boolean followSymlinks) throws IOException {
    return read(path, followSymlinks, (fs, p) -> fs.stat(p, followSymlinks));
  }

  @Override
  public void createSymbolicLink(
      PathFragment linkPath, PathFragment targetFragment, SymlinkTargetType hint)
      throws IOException {
    fsForPath(linkPath).createSymbolicLink(linkPath, targetFragment, hint);
  }

  @Override
  public PathFragment readSymbolicLink(PathFragment path) throws IOException {
    return read(path, /* followLast= */ false, (fs, p) -> fs.readSymbolicLink(p));
  }

  @Override
  public boolean exists(PathFragment path, boolean followSymlinks) throws IOException {
    return read(path, followSymlinks, (fs, p) -> fs.exists(p, followSymlinks));
  }

  @Override
  public boolean exists(PathFragment path) throws IOException {
    return read(path, /* followLast= */ true, (fs, p) -> fs.exists(p));
  }

  @Override
  public Collection<String> getDirectoryEntries(PathFragment path) throws IOException {
    return read(path, /* followLast= */ true, (fs, p) -> fs.getDirectoryEntries(p));
  }

  @Override
  public boolean isReadable(PathFragment path) throws IOException {
    return read(path, /* followLast= */ true, (fs, p) -> fs.isReadable(p));
  }

  @Override
  public void setReadable(PathFragment path, boolean readable) throws IOException {
    fsForPath(path).setReadable(path, readable);
  }

  @Override
  public boolean isWritable(PathFragment path) throws IOException {
    return read(path, /* followLast= */ true, (fs, p) -> fs.isWritable(p));
  }

  @Override
  public void setWritable(PathFragment path, boolean writable) throws IOException {
    fsForPath(path).setWritable(path, writable);
  }

  @Override
  public boolean isExecutable(PathFragment path) throws IOException {
    return read(path, /* followLast= */ true, (fs, p) -> fs.isExecutable(p));
  }

  @Override
  public void setExecutable(PathFragment path, boolean executable) throws IOException {
    fsForPath(path).setExecutable(path, executable);
  }

  @Override
  public InputStream getInputStream(PathFragment path) throws IOException {
    return read(path, /* followLast= */ true, (fs, p) -> fs.getInputStream(p));
  }

  @Override
  public SeekableByteChannel createReadWriteByteChannel(PathFragment path) throws IOException {
    return fsForPath(path).createReadWriteByteChannel(path);
  }

  @Override
  public OutputStream getOutputStream(PathFragment path, boolean append, boolean internal)
      throws IOException {
    return fsForPath(path).getOutputStream(path, append, internal);
  }

  @Override
  public void renameTo(PathFragment sourcePath, PathFragment targetPath) throws IOException {
    fsForPath(sourcePath).renameTo(sourcePath, targetPath);
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
    return read(path, followSymlinks, (fs, p) -> fs.getxattr(p, name, followSymlinks));
  }

  @Nullable
  @Override
  public PathFragment resolveOneLink(PathFragment path) throws IOException {
    return fsForPath(path).resolveOneLink(path);
  }

  @Override
  public Path resolveSymbolicLinks(PathFragment path) throws IOException {
    PathFragment resolved =
        read(path, /* followLast= */ true, (fs, p) -> fs.resolveSymbolicLinks(p).asFragment());
    // Ensure that the return value doesn't leave the overlay file system.
    return getPath(resolved);
  }

  @Nullable
  @Override
  public FileStatus statIfFound(PathFragment path, boolean followSymlinks) throws IOException {
    return read(path, followSymlinks, (fs, p) -> fs.statIfFound(p, followSymlinks));
  }

  @Override
  public boolean isFile(PathFragment path, boolean followSymlinks) throws IOException {
    return read(path, followSymlinks, (fs, p) -> fs.isFile(p, followSymlinks));
  }

  @Override
  public boolean isSpecialFile(PathFragment path, boolean followSymlinks) throws IOException {
    return read(path, followSymlinks, (fs, p) -> fs.isSpecialFile(p, followSymlinks));
  }

  @Override
  public boolean isSymbolicLink(PathFragment path) throws IOException {
    return read(path, /* followLast= */ false, (fs, p) -> fs.isSymbolicLink(p));
  }

  @Override
  public boolean isDirectory(PathFragment path, boolean followSymlinks) throws IOException {
    return read(path, followSymlinks, (fs, p) -> fs.isDirectory(p, followSymlinks));
  }

  @Override
  public PathFragment readSymbolicLinkUnchecked(PathFragment path) throws IOException {
    return read(path, /* followLast= */ false, (fs, p) -> fs.readSymbolicLinkUnchecked(p));
  }

  @Override
  public Collection<Dirent> readdir(PathFragment path, boolean followSymlinks) throws IOException {
    // The directory is always followed, the flag only applies to the entries, whose symlinks may
    // lead into the other backing file system and are thus followed through this one.
    return read(
        path,
        /* followLast= */ true,
        (fs, p) -> {
          Collection<Dirent> entries = fs.readdir(p, /* followSymlinks= */ false);
          if (!followSymlinks || fs == externalFs) {
            return followSymlinks ? fs.readdir(p, /* followSymlinks= */ true) : entries;
          }
          ImmutableList.Builder<Dirent> followed =
              ImmutableList.builderWithExpectedSize(entries.size());
          for (Dirent entry : entries) {
            followed.add(followDirent(p, entry));
          }
          return followed.build();
        });
  }

  /** Returns the given entry of the given directory with the type of what its symlink points to. */
  private Dirent followDirent(PathFragment dir, Dirent entry) throws IOException {
    if (entry.getType() != Dirent.Type.SYMLINK) {
      return entry;
    }
    FileStatus status;
    try {
      status = statIfFound(dir.getChild(entry.getName()), /* followSymlinks= */ true);
    } catch (FileSymlinkLoopException e) {
      status = null;
    }
    Dirent.Type type;
    if (status == null) {
      type = Dirent.Type.UNKNOWN;
    } else if (status.isFile()) {
      type = Dirent.Type.FILE;
    } else if (status.isDirectory()) {
      type = Dirent.Type.DIRECTORY;
    } else {
      type = Dirent.Type.UNKNOWN;
    }
    return new Dirent(entry.getName(), type);
  }

  @Override
  public void chmod(PathFragment path, int mode) throws IOException {
    fsForPath(path).chmod(path, mode);
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

  private final class RemoteExternalFileSystem
      extends RemoteActionFileSystem.RemoteInMemoryFileSystem {

    RemoteExternalFileSystem(DigestHashFunction hashFunction) {
      super(hashFunction);
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

    private FileArtifactValue getMetadata(PathFragment path) throws IOException {
      var status = stat(path, /* followSymlinks= */ false);
      if (!status.isSymbolicLink()) {
        return ((RemoteActionFileSystem.RemoteInMemoryFileInfo) status).getMetadata();
      }
      return FileArtifactValue.createForUnresolvedSymlink(externalFs.getPath(path));
    }

    @Override
    public InputStream getInputStream(PathFragment path) throws IOException {
      // Symlinks are never prefetched to the native file system themselves, only the regular file
      // they resolve to, so follow them before reading a prefetched file. Either end of the chain
      // can be what makes the read eligible: a symlink named `helper.bzl` pointing at `helper.txt`
      // as well as one named `helper.txt` pointing at `helper.bzl`.
      boolean prefetched = shouldPrefetch(path);
      path = resolveSymbolicLinks(path).asFragment();
      if (prefetched || shouldPrefetch(path)) {
        return nativeFs.getInputStream(path);
      }
      var relativePath = path.relativeTo(externalDirectory);
      if (!(stat(path, /* followSymlinks= */ true)
          instanceof RemoteActionFileSystem.RemoteInMemoryFileInfo info)) {
        throw Errno.EISDIR.exception(path);
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
                makeRemoteContext(relativePath),
                path.getPathString(),
                /* execPath= */ null,
                digest);
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
    public byte[] getDigest(PathFragment path) throws IOException {
      // All regular files in this file system are remote files, whose digest is known in advance
      // and returned by the base implementation of getFastDigest, which also correctly reports
      // errors such as EISDIR for paths that don't resolve to regular files. The base
      // implementation of getDigest would instead download the file contents to hash them.
      return getFastDigest(path);
    }
  }
}
