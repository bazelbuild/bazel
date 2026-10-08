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
import static com.google.common.collect.ImmutableSet.toImmutableSet;
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
import com.google.common.collect.Sets;
import com.google.common.hash.HashCode;
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
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.rewinding.LostRemoteRepoFileException;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSymlinkLoopException;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RewindableRepoFileSystem;
import com.google.devtools.build.lib.vfs.SymlinkTargetType;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
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
import java.util.TreeMap;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.function.Consumer;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * A file system that overlays the native file system with a {@link RemoteExternalFileSystem} for
 * the "external" directory, which contains the contents of external repositories.
 *
 * <p>Each external repository can either be materialized to the native file system or kept in
 * memory in the {@link RemoteExternalFileSystem}.
 */
public final class RemoteExternalOverlayFileSystem extends FileSystem
    implements LazyMaterializer, RewindableRepoFileSystem {
  private final PathFragment externalDirectory;
  private final int externalDirectorySegmentCount;
  private final FileSystem nativeFs;
  private final RemoteExternalFileSystem externalFs;
  private final TaskDeduplicator<String, Void, Void> materializations = new TaskDeduplicator<>();
  // The states of the repos that have been retrieved from the remote cache, by repo name. A repo
  // without a state is served from nativeFs and is not known to have lost any files. All changes of
  // the state of a repo are atomic.
  private final ConcurrentHashMap<String, RepoState> repoStates = new ConcurrentHashMap<>();

  // Per-build information that is set in beforeCommand and cleared in afterCommand.
  @Nullable private CombinedCache cache;
  @Nullable private AbstractActionInputPrefetcher inputPrefetcher;
  @Nullable private Reporter reporter;
  @Nullable private String buildRequestId;
  @Nullable private String commandId;
  @Nullable private Supplier<MemoizingEvaluator> evaluator;
  @Nullable private Duration remoteCacheTtl;
  @Nullable private ListeningExecutorService materializationExecutor;

  /**
   * A repo whose contents have been injected into externalFs.
   *
   * @param markerFile the contents of the marker file to write when the repo is materialized
   * @param rootDigest the digest of the root directory of the injected contents
   */
  private record InjectedRepo(String markerFile, Digest rootDigest) {}

  private sealed interface RepoState {}

  /**
   * The repo is served from externalFs.
   *
   * @param hasLostFiles whether the remote cache has lost files of the repo, which then have to be
   *     restored via {@link #materializeFrom}
   */
  private record InMemory(InjectedRepo contents, boolean hasLostFiles) implements RepoState {}

  /**
   * The repo has been fully materialized to nativeFs and is served from there. Its contents remain
   * available in externalFs until the end of the command for those who are still reading them.
   */
  private record Materialized(InjectedRepo contents) implements RepoState {}

  /**
   * The remote cache has lost files of the repo and its contents are no longer available in
   * externalFs, so that it has to be fetched again rather than looked up in the cache.
   *
   * @param markerFile the contents of the marker file of the cache entry that references the lost
   *     files, which identifies the entry; entries for other inputs of the repo rule are unaffected
   */
  private record AwaitingRefetch(String markerFile) implements RepoState {}

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
      Supplier<MemoizingEvaluator> evaluator,
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
    var reposToRefetch = new HashSet<String>();
    for (String repoName : ImmutableSet.copyOf(repoStates.keySet())) {
      repoStates.computeIfPresent(
          repoName,
          (unused, state) ->
              switch (state) {
                case Materialized materialized -> evict(repoName, materialized);
                // A repo with lost files that is still served from memory has not been restored
                // during this command, e.g. because the command failed first. Drop its contents
                // and invalidate its fetch so that the next command fetches it again.
                case InMemory inMemory when inMemory.hasLostFiles() -> {
                  reposToRefetch.add(repoName);
                  yield evict(repoName, inMemory);
                }
                default -> state;
              });
    }
    if (!reposToRefetch.isEmpty()) {
      // The evaluator may have been replaced during the command.
      invalidateRepoDirectories(evaluator.get(), reposToRefetch);
    }
    this.evaluator = null;
  }

  private void evictInMemoryRepo(String repoName) {
    repoStates.computeIfPresent(repoName, (unused, state) -> evict(repoName, state));
  }

  @Nullable
  private RepoState evict(String repoName, RepoState state) {
    try {
      externalFs.deleteTree(externalDirectory.getChild(repoName));
    } catch (IOException e) {
      throw new IllegalStateException("In-memory file system is not expected to throw", e);
    }
    return withoutContents(state);
  }

  @Nullable
  private static RepoState withoutContents(RepoState state) {
    return switch (state) {
      case InMemory inMemory when inMemory.hasLostFiles() ->
          new AwaitingRefetch(inMemory.contents().markerFile());
      case InMemory unused -> null;
      case Materialized unused -> null;
      case AwaitingRefetch awaitingRefetch -> awaitingRefetch;
    };
  }

  @Nullable
  private InjectedRepo getInjectedRepo(String repoName) {
    return switch (repoStates.get(repoName)) {
      case InMemory inMemory -> inMemory.contents();
      case Materialized materialized -> materialized.contents();
      case AwaitingRefetch unused -> null;
      case null -> null;
    };
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
    evictInMemoryRepo(repo.getName());
    deleteTree(repoDir);
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
    // materialized.
    repoStates.put(
        repo.getName(),
        new InMemory(
            new InjectedRepo(markerFile, cache.digestUtil.compute(remoteContents.getRoot())),
            /* hasLostFiles= */ false));
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

  @Override
  public boolean isRepoPath(PathFragment path) {
    return path.startsWith(externalDirectory)
        && path.segmentCount() > externalDirectorySegmentCount;
  }

  @Override
  public RepositoryName repoContaining(PathFragment path) {
    return RepositoryName.createUnvalidated(path.getSegment(externalDirectorySegmentCount));
  }

  @Override
  public void markLostRepoFile(RepositoryName repo) {
    // A loss reported after the repo has been materialized or evicted is ignored: recording it
    // would keep the repo from being looked up in the cache by later commands.
    repoStates.computeIfPresent(
        repo.getName(),
        (unused, state) ->
            state instanceof InMemory inMemory
                ? new InMemory(inMemory.contents(), /* hasLostFiles= */ true)
                : state);
  }

  /**
   * Returns whether the remote cache has lost files referenced by the cache entry of the given repo
   * with the given marker file, in which case the entry has to be treated as a miss. Other entries
   * of the repo are unaffected.
   *
   * <p>Doesn't clear the state: a lookup is not a promise that the repo will actually be fetched.
   */
  public boolean shouldRefetch(RepositoryName repo, String markerFile) {
    return switch (repoStates.get(repo.getName())) {
      case InMemory inMemory ->
          inMemory.hasLostFiles() && inMemory.contents().markerFile().equals(markerFile);
      case AwaitingRefetch awaitingRefetch -> awaitingRefetch.markerFile().equals(markerFile);
      case Materialized unused -> false;
      case null -> false;
    };
  }

  @Override
  public boolean isServedFromCache(RepositoryName repo) {
    return getInjectedRepo(repo.getName()) != null;
  }

  @Override
  public void repoRefetched(RepositoryName repo) {
    // The fetched contents are served from disk and no longer reference the lost files. Whether
    // they have also been uploaded doesn't matter here: a cache entry that still references lost
    // files is only consulted once the repo's marker file has become stale and is recovered from
    // like the first time.
    forgetLostFiles(repo.getName());
  }

  @Override
  public void repoRefetchFailed(RepositoryName repo) {
    // A file is also reported as lost if it couldn't be downloaded for any other reason, so the
    // cached contents may be usable after all. As long as the repo can't be fetched, they are the
    // only way to get it, and if they still reference lost files, that is noticed again.
    forgetLostFiles(repo.getName());
  }

  private void forgetLostFiles(String repoName) {
    repoStates.computeIfPresent(
        repoName,
        (unused, state) ->
            switch (state) {
              case InMemory inMemory -> new InMemory(inMemory.contents(), /* hasLostFiles= */ false);
              case Materialized materialized -> materialized;
              case AwaitingRefetch awaitingRefetch -> null;
            });
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
    if (!isServedFromMemory(repo.getName())) {
      // The repo has not been injected into the in-memory file system or has already been
      // materialized.
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
                  if (!isServedFromMemory(repo.getName())) {
                    return immediateVoidFuture();
                  }
                  return materializationExecutor.submit(
                      () -> {
                        doMaterialize(repo, reporter, /* filesInPlace= */ false);
                        return null;
                      });
                }));
  }

  /**
   * @param filesInPlace whether all regular files of the repo are already present on the native
   *     file system. They are then not prefetched, which would join any download of a lost file
   *     that is still in flight and fail along with it.
   */
  private void doMaterialize(
      RepositoryName repo, ExtendedEventHandler reporter, boolean filesInPlace)
      throws IOException, InterruptedException {
    var injectedRepo = getInjectedRepo(repo.getName());
    if (injectedRepo == null) {
      return;
    }
    reporter.handle(Event.debug("Materializing remote repo %s".formatted(repo)));
    materializeSubtree(externalDirectory.getChild(repo.getName()), filesInPlace);

    // The repo may be materialized multiple times concurrently, so every attempt writes to its own
    // temporary file, whose name doesn't include that of the repo as it could become too long.
    var markerFile = nativeFs.getPath(externalDirectory.getChild(repo.getMarkerFileName()));
    var markerFileSibling =
        nativeFs.getPath(
            externalDirectory.getChild("@%s.marker.tmp".formatted(UUID.randomUUID())));
    try {
      FileSystemUtils.writeContentAsLatin1(markerFileSibling, injectedRepo.markerFile());
      markerFileSibling.renameTo(markerFile);
    } catch (IOException e) {
      try {
        var unused = markerFileSibling.delete();
      } catch (IOException e2) {
        e.addSuppressed(e2);
      }
      throw e;
    }
    // Only serve the repo from the native file system once it is complete there, including its
    // marker file, which keeps a later fetch of the repo from replacing its contents.
    repoStates.computeIfPresent(
        repo.getName(),
        (unused, state) ->
            state instanceof InMemory inMemory ? new Materialized(inMemory.contents()) : state);
  }

  private boolean isServedFromMemory(String repoName) {
    return repoStates.get(repoName) instanceof InMemory;
  }

  /**
   * Returns the contents of the marker file of the cache entry that the given repo has been
   * retrieved from if the remote cache has lost files of it while its contents are served from
   * memory, otherwise null. Such a repo has to be restored via {@link #materializeFrom}.
   */
  @Nullable
  public String getLostFilesMarkerFile(RepositoryName repo) {
    return repoStates.get(repo.getName()) instanceof InMemory inMemory && inMemory.hasLostFiles()
        ? inMemory.contents().markerFile()
        : null;
  }

  /**
   * Returns the digest of the root directory of the contents of the given repo that have been
   * retrieved from the remote cache, or null if there are none.
   */
  @Nullable
  public Digest getInjectedRootDigest(RepositoryName repo) {
    var injectedRepo = getInjectedRepo(repo.getName());
    return injectedRepo != null ? injectedRepo.rootDigest() : null;
  }

  /**
   * Returns the contents of the marker file of the given repo that have been retrieved from the
   * remote cache, or null if there are none.
   */
  @Nullable
  public String getInjectedMarkerFile(RepositoryName repo) {
    var injectedRepo = getInjectedRepo(repo.getName());
    return injectedRepo != null ? injectedRepo.markerFile() : null;
  }

  /**
   * Materializes the given repo to the native file system, taking the contents of its files from
   * the given directory rather than the remote cache.
   *
   * <p>The directory must have the same contents as the repo, which the caller has to verify by
   * comparing the digest of its root with {@link #getInjectedRootDigest}. Its files are moved into
   * the repo unless they are already present there, so that files that others may be reading are
   * left alone. Does nothing if the repo has been materialized in the meantime.
   */
  public void materializeFrom(RepositoryName repo, Path contents, ExtendedEventHandler reporter)
      throws IOException, InterruptedException {
    if (!isServedFromMemory(repo.getName())) {
      return;
    }
    installFiles(
        nativeFs.getPath(contents.asFragment()), externalDirectory.getChild(repo.getName()));
    doMaterialize(repo, reporter, /* filesInPlace= */ true);
  }

  private void installFiles(Path sourceDir, PathFragment targetDir) throws IOException {
    // Moving a file out of a directory requires the directory to be writable, which a repo rule
    // doesn't have to leave it as.
    sourceDir.setWritable(true);
    for (var dirent : sourceDir.readdir(Symlinks.NOFOLLOW)) {
      var source = sourceDir.getChild(dirent.getName());
      var target = targetDir.getChild(dirent.getName());
      switch (dirent.getType()) {
        case FILE ->
            inputPrefetcher.installFile(source, getPath(target), externalFs.getMetadata(target));
        case DIRECTORY -> installFiles(source, target);
        default -> {}
      }
    }
  }

  /**
   * Describes the first difference between the contents of the given repo that have been retrieved
   * from the remote cache and those of the given directory.
   */
  public String describeFirstDifference(RepositoryName repo, Path contents) throws IOException {
    var difference =
        findFirstDifference(
            externalFs.getPath(externalDirectory.getChild(repo.getName())),
            nativeFs.getPath(contents.asFragment()),
            PathFragment.EMPTY_FRAGMENT);
    return difference != null ? difference : "no difference found";
  }

  @Nullable
  private static String findFirstDifference(
      Path cachedDir, Path fetchedDir, PathFragment relativePath) throws IOException {
    var cachedEntries = new TreeMap<String, Dirent.Type>();
    for (var dirent : cachedDir.readdir(Symlinks.NOFOLLOW)) {
      cachedEntries.put(dirent.getName(), dirent.getType());
    }
    var fetchedEntries = new TreeMap<String, Dirent.Type>();
    for (var dirent : fetchedDir.readdir(Symlinks.NOFOLLOW)) {
      fetchedEntries.put(dirent.getName(), dirent.getType());
    }
    for (var name : Sets.union(cachedEntries.keySet(), fetchedEntries.keySet())) {
      var path = relativePath.getChild(name);
      var cachedType = cachedEntries.get(name);
      var fetchedType = fetchedEntries.get(name);
      if (cachedType == null) {
        return "%s only exists in the fetched contents".formatted(path);
      }
      if (fetchedType == null) {
        return "%s only exists in the cached contents".formatted(path);
      }
      if (cachedType != fetchedType) {
        return "%s is a %s in the cached contents, but a %s in the fetched contents"
            .formatted(path, describe(cachedType), describe(fetchedType));
      }
      var cached = cachedDir.getChild(name);
      var fetched = fetchedDir.getChild(name);
      String difference =
          switch (cachedType) {
            case DIRECTORY -> findFirstDifference(cached, fetched, path);
            case SYMLINK -> {
              var cachedTarget = cached.readSymbolicLink();
              var fetchedTarget = fetched.readSymbolicLink();
              yield cachedTarget.equals(fetchedTarget)
                  ? null
                  : "%s points to %s in the cached contents, but to %s in the fetched contents"
                      .formatted(path, cachedTarget, fetchedTarget);
            }
            case FILE -> {
              var cachedDigest = HashCode.fromBytes(cached.getDigest());
              var fetchedDigest = HashCode.fromBytes(fetched.getDigest());
              if (!cachedDigest.equals(fetchedDigest)) {
                yield "%s has digest %s in the cached contents, but %s in the fetched contents"
                    .formatted(path, cachedDigest, fetchedDigest);
              }
              yield cached.isExecutable() == fetched.isExecutable()
                  ? null
                  : "%s is %s in the cached contents, but %s in the fetched contents"
                      .formatted(
                          path,
                          cached.isExecutable() ? "executable" : "not executable",
                          fetched.isExecutable() ? "executable" : "not executable");
            }
            default -> null;
          };
      if (difference != null) {
        return difference;
      }
    }
    return null;
  }

  private static String describe(Dirent.Type type) {
    return switch (type) {
      case DIRECTORY -> "directory";
      case SYMLINK -> "symlink";
      case FILE -> "file";
      case UNKNOWN -> "special file";
    };
  }

  /**
   * Records that the given file in a repo has been lost from the remote cache and returns the
   * exception to fail the read with.
   */
  private LostRemoteRepoFileException lostRemoteFile(
      PathFragment relativePath, Digest digest, BulkTransferException cause) {
    String repoName = relativePath.getSegment(0);
    markLostRepoFile(RepositoryName.createUnvalidated(repoName));
    return new LostRemoteRepoFileException(
        "%s/%s with digest %s is no longer available in the remote cache"
            .formatted(externalDirectory.getBaseName(), relativePath, DigestUtil.toString(digest)),
        cause,
        RepositoryName.createUnvalidated(repoName),
        DigestUtil.toString(digest));
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
    var reposToDiscard =
        repoStates.keySet().stream()
            .filter(repoName -> getInjectedRepo(repoName) != null)
            .collect(toImmutableSet());
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
    materializeSubtree(path, /* filesInPlace= */ false);
  }

  private void materializeSubtree(PathFragment path, boolean filesInPlace)
      throws IOException, InterruptedException {
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
    try {
      if (!filesInPlace) {
        prefetch(files);
      }
      // Create symlinks last as some platforms don't allow creating a symlink to a non-existent
      // target.
      prefetch(symlinks);
    } catch (BulkTransferException e) {
      var lostArtifacts = e.getLostArtifacts(ActionInputHelper::fromPath);
      if (!lostArtifacts.isEmpty()) {
        // We don't track the particular lost artifacts since the repo needs to be fetched again,
        // which restores all of them anyway.
        var anyLostArtifact = lostArtifacts.byDigest().entries().iterator().next();
        var relativePath = anyLostArtifact.getValue().getExecPath().relativeTo(externalDirectory);
        throw lostRemoteFile(relativePath, DigestUtil.fromString(anyLostArtifact.getKey()), e);
      }
      throw e;
    }
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
    // .bzl and .scl files are typically small and the loads between them can form complex DAGs that
    // can only be discovered layer by layer, so prefetching is worthwhile to reduce the number of
    // sequential cache requests.
    // None of these files are read by nodes that can rewind the fetch of the repo when a file has
    // been lost, so prefetching turns such a loss into a cache miss instead.
    String extension = path.getFileExtension();
    String baseName = path.getBaseName();
    return extension.equals("bzl")
        || extension.equals("scl")
        || baseName.equals("REPO.bazel")
        || baseName.equals(".bazelignore")
        || baseName.equals("MODULE.bazel")
        || baseName.endsWith(".MODULE.bazel");
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
    if (fs == externalFs || repoStates.isEmpty() || !path.startsWith(externalDirectory)) {
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
      if (isServedFromMemory(repoName)) {
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
        repoStates.computeIfPresent(
            repoName,
            (unused, state) -> state instanceof InMemory ? withoutContents(state) : state);
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
      if (inputPrefetcher.isAvailable(nativeFs.getPath(path), info.getMetadata())) {
        // The file has been downloaded before, e.g. as an input of an action.
        return nativeFs.getInputStream(path);
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
          throw lostRemoteFile(relativePath, digest, e);
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
