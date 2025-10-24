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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;
import static com.google.devtools.build.lib.remote.util.Utils.waitForBulkTransfer;
import static com.google.devtools.build.lib.util.StringEncoding.unicodeToInternal;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.remote.common.BulkTransferException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.vfs.DetailedIOException;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileStatus;
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
import java.io.IOException;
import java.io.InputStream;
import java.io.InterruptedIOException;
import java.io.OutputStream;
import java.nio.channels.SeekableByteChannel;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import javax.annotation.Nullable;

/**
 * A file system that overlays the native file system with a {@link RemoteExternalFileSystem} for
 * the "external" directory, which contains the contents of external repositories.
 *
 * <p>Each external repository can either be materialized to the native file system or kept in
 * memory in the {@link RemoteExternalFileSystem}.
 */
public final class RemoteExternalOverlayFileSystem extends FileSystem {
  private static final int MATERIALIZATION_FINISHED = 0;
  private static final int MATERIALIZATION_IN_PROGRESS = 1;
  private static final int MATERIALIZATION_NOT_STARTED = 2;

  private final PathFragment externalDirectory;
  private final int externalDirectorySegmentCount;
  private final FileSystem nativeFs;
  private final RemoteExternalFileSystem externalFs;
  // The count of the latch represents the current materialization state (see constants above).
  private final ConcurrentHashMap<String, CountDownLatch> remoteRepoMaterializationState =
      new ConcurrentHashMap<>();
  private final ConcurrentHashMap<String, String> markerFileContents = new ConcurrentHashMap<>();
  private final Set<String> reposWithLostFiles = ConcurrentHashMap.newKeySet();

  // Per-build information that is set in beforeCommand and cleared in afterCommand.
  @Nullable private CombinedCache cache;
  @Nullable private AbstractActionInputPrefetcher inputPrefetcher;
  @Nullable private Reporter reporter;
  @Nullable private String buildRequestId;
  @Nullable private String commandId;
  @Nullable private MemoizingEvaluator evaluator;

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
      MemoizingEvaluator evaluator) {
    checkState(
        this.cache == null
            && this.inputPrefetcher == null
            && this.reporter == null
            && this.buildRequestId == null
            && this.commandId == null
            && this.evaluator == null);
    this.cache = cache;
    this.inputPrefetcher = inputPrefetcher;
    this.reporter = reporter;
    this.buildRequestId = buildRequestId;
    this.commandId = commandId;
    this.evaluator = evaluator;
  }

  public void afterCommand() {
    this.cache = null;
    this.inputPrefetcher = null;
    this.reporter = null;
    this.buildRequestId = null;
    this.commandId = null;
    // Clean up the in-memory contents of materialized repos or those that need to be refetched to
    // recover files that the remote cache has lost. This wouldn't be safe to do eagerly as ongoing
    // repo rule evaluations may still refer to the in-memory content and refetching is not atomic.
    remoteRepoMaterializationState.forEach(
        0,
        (repoName, materializationState) ->
            materializationState.getCount() == MATERIALIZATION_FINISHED
                    || reposWithLostFiles.contains(repoName)
                ? repoName
                : null,
        this::deleteRepo);
    if (!reposWithLostFiles.isEmpty()) {
      evaluator.delete(
          k ->
              k.functionName().equals(SkyFunctions.REPOSITORY_DIRECTORY)
                  && reposWithLostFiles.contains(((RepositoryName) k.argument()).getName()));
    }
    reposWithLostFiles.clear();
    this.evaluator = null;
  }

  public void injectRemoteRepo(RepositoryName repo, Tree remoteContents, String markerFile)
      throws IOException {
    var childMap =
        remoteContents.getChildrenList().stream()
            .collect(
                toImmutableMap(cache.digestUtil::compute, directory -> directory, (a, b) -> a));
    injectRecursively(
        externalFs, externalDirectory.getChild(repo.getName()), remoteContents.getRoot(), childMap);
    // Keep the marker file contents in memory so that it can be written out when the repo is
    // materialized.
    markerFileContents.put(repo.getName(), markerFile);
    // Create the repo directory on disk so that readdir reflects the overlaid state of the external
    // directory.
    nativeFs.createDirectoryAndParents(externalDirectory.getChild(repo.getName()));
    remoteRepoMaterializationState.put(repo.getName(), new CountDownLatch(2));
  }

  // Must not be called concurrently with any other method.
  private void deleteRepo(String repoName) {
    try {
      externalFs.deleteTree(externalDirectory.getChild(repoName));
    } catch (IOException e) {
      throw new IllegalStateException("In-memory file system is not expected to throw", e);
    }
    remoteRepoMaterializationState.remove(repoName);
  }

  private static void injectRecursively(
      RemoteExternalFileSystem fs,
      PathFragment path,
      Directory dir,
      ImmutableMap<Digest, Directory> childMap)
      throws IOException {
    fs.createDirectoryAndParents(path);
    for (var file : dir.getFilesList()) {
      var filePath = path.getRelative(unicodeToInternal(file.getName()));
      fs.injectFile(
          filePath,
          FileArtifactValue.createForRemoteFile(
              DigestUtil.toBinaryDigest(file.getDigest()),
              file.getDigest().getSizeBytes(),
              /* locationIndex= */ 1));
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
      injectRecursively(fs, subdirPath, subdir, childMap);
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
  public void ensureMaterialized(RepositoryName repo, Runnable report)
      throws IOException, InterruptedException {
    String repoName = repo.getName();
    // Fast path that avoids write locking and allocations.
    var state = remoteRepoMaterializationState.get(repoName);
    if (state == null || state.getCount() == MATERIALIZATION_FINISHED) {
      // Never injected or already materialized.
      return;
    }
    var newLatch = new CountDownLatch(MATERIALIZATION_IN_PROGRESS);
    var materializationLatch =
        checkNotNull(
            remoteRepoMaterializationState.computeIfPresent(
                repoName,
                (key, currentValue) -> {
                  if (currentValue.getCount() == MATERIALIZATION_NOT_STARTED) {
                    // Claim this materialization.
                    return newLatch;
                  }
                  return currentValue;
                }));
    if (materializationLatch != newLatch) {
      // Another caller won the race, wait for it to complete materialization.
      materializationLatch.await();
      return;
    }

    report.run();
    var repoPath = externalDirectory.getChild(repoName);
    var remoteRepo = externalFs.getPath(repoPath);
    var walkResult = walk(remoteRepo);
    getFromFuture(
        inputPrefetcher.prefetchFilesInterruptibly(
            /* action= */ null,
            Iterables.transform(
                walkResult.files(), path -> ActionInputHelper.fromPath(path.asFragment())),
            actionInput -> externalFs.getMetadata(actionInput.getExecPath()),
            ActionInputPrefetcher.Priority.CRITICAL,
            ActionInputPrefetcher.Reason.INPUTS));
    // Create symlinks last as some platforms don't allow creating a symlink to a non-existent
    // target.
    for (var remoteSymlink : walkResult.symlinks()) {
      var nativeSymlink = nativeFs.getPath(remoteSymlink.asFragment());
      nativeSymlink.getParentDirectory().createDirectoryAndParents();
      nativeSymlink.createSymbolicLink(remoteSymlink.readSymbolicLink());
    }

    // After the repo has been copied, atomically materialize the marker file. This ensures that the
    // repo doesn't have to be refetched after the next server restart.
    var markerFile = nativeFs.getPath(externalDirectory.getChild(repo.getMarkerFileName()));
    var markerFileSibling =
        nativeFs.getPath(externalDirectory.getChild(repo.getMarkerFileName() + ".tmp"));
    FileSystemUtils.writeContentAsLatin1(markerFileSibling, markerFileContents.remove(repoName));
    markerFileSibling.renameTo(markerFile);

    materializationLatch.countDown();
  }

  private record WalkResult(List<Path> files, List<Path> symlinks) {}

  private static WalkResult walk(Path root) throws IOException {
    var result = new WalkResult(new ArrayList<>(), new ArrayList<>());
    walk(root, result);
    return result;
  }

  private static void walk(Path root, WalkResult result) throws IOException {
    for (var dirent : root.readdir(Symlinks.NOFOLLOW)) {
      var fromChild = root.getChild(dirent.getName());
      switch (dirent.getType()) {
        case FILE -> result.files.add(fromChild);
        case SYMLINK -> result.symlinks.add(fromChild);
        case DIRECTORY -> walk(fromChild, result);
        default -> throw new IOException("Unsupported file type: " + dirent);
      }
    }
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
  private FileSystem fsForPath(PathFragment path) {
    if (path.startsWith(externalDirectory) && !path.equals(externalDirectory)) {
      String repoName = path.getSegment(externalDirectorySegmentCount);
      var materializationState = remoteRepoMaterializationState.get(repoName);
      if (materializationState != null
          && materializationState.getCount() != MATERIALIZATION_FINISHED) {
        // The repo may have been deleted due to refetching. Clean up in-memory state if that is the
        // case.
        if (externalFs.getPath(externalDirectory.getChild(repoName)).exists()) {
          return externalFs;
        }
        remoteRepoMaterializationState.remove(repoName);
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
    return fsForPath(path).getDigest(path);
  }

  @Nullable
  @Override
  public byte[] getFastDigest(PathFragment path) throws IOException {
    return fsForPath(path).getFastDigest(path);
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
    return fsForPath(path).getFileSize(path, followSymlinks);
  }

  @Override
  public long getLastModifiedTime(PathFragment path, boolean followSymlinks) throws IOException {
    return fsForPath(path).getLastModifiedTime(path, followSymlinks);
  }

  @Override
  public void setLastModifiedTime(PathFragment path, long newTime) throws IOException {
    fsForPath(path).setLastModifiedTime(path, newTime);
  }

  @Override
  public FileStatus stat(PathFragment path, boolean followSymlinks) throws IOException {
    return fsForPath(path).stat(path, followSymlinks);
  }

  @Override
  public void createSymbolicLink(
      PathFragment linkPath, PathFragment targetFragment, SymlinkTargetType hint)
      throws IOException {
    fsForPath(linkPath).createSymbolicLink(linkPath, targetFragment, hint);
  }

  @Override
  public PathFragment readSymbolicLink(PathFragment path) throws IOException {
    return fsForPath(path).readSymbolicLink(path);
  }

  @Override
  public boolean exists(PathFragment path, boolean followSymlinks) {
    return fsForPath(path).exists(path, followSymlinks);
  }

  @Override
  public Collection<String> getDirectoryEntries(PathFragment path) throws IOException {
    return fsForPath(path).getDirectoryEntries(path);
  }

  @Override
  public boolean isReadable(PathFragment path) throws IOException {
    return fsForPath(path).isReadable(path);
  }

  @Override
  public void setReadable(PathFragment path, boolean readable) throws IOException {
    fsForPath(path).setReadable(path, readable);
  }

  @Override
  public boolean isWritable(PathFragment path) throws IOException {
    return fsForPath(path).isWritable(path);
  }

  @Override
  public void setWritable(PathFragment path, boolean writable) throws IOException {
    fsForPath(path).setWritable(path, writable);
  }

  @Override
  public boolean isExecutable(PathFragment path) throws IOException {
    return fsForPath(path).isExecutable(path);
  }

  @Override
  public void setExecutable(PathFragment path, boolean executable) throws IOException {
    fsForPath(path).setExecutable(path, executable);
  }

  @Override
  public InputStream getInputStream(PathFragment path) throws IOException {
    return fsForPath(path).getInputStream(path);
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
    return fsForPath(path).getxattr(path, name, followSymlinks);
  }

  @Nullable
  @Override
  public PathFragment resolveOneLink(PathFragment path) throws IOException {
    return fsForPath(path).resolveOneLink(path);
  }

  @Override
  public Path resolveSymbolicLinks(PathFragment path) throws IOException {
    return fsForPath(path).resolveSymbolicLinks(path);
  }

  @Nullable
  @Override
  public FileStatus statNullable(PathFragment path, boolean followSymlinks) {
    return fsForPath(path).statNullable(path, followSymlinks);
  }

  @Nullable
  @Override
  public FileStatus statIfFound(PathFragment path, boolean followSymlinks) throws IOException {
    return fsForPath(path).statIfFound(path, followSymlinks);
  }

  @Override
  public boolean isFile(PathFragment path, boolean followSymlinks) {
    return fsForPath(path).isFile(path, followSymlinks);
  }

  @Override
  public boolean isSpecialFile(PathFragment path, boolean followSymlinks) {
    return fsForPath(path).isSpecialFile(path, followSymlinks);
  }

  @Override
  public boolean isSymbolicLink(PathFragment path) {
    return fsForPath(path).isSymbolicLink(path);
  }

  @Override
  public boolean isDirectory(PathFragment path, boolean followSymlinks) {
    return fsForPath(path).isDirectory(path, followSymlinks);
  }

  @Override
  public PathFragment readSymbolicLinkUnchecked(PathFragment path) throws IOException {
    return fsForPath(path).readSymbolicLinkUnchecked(path);
  }

  @Override
  public boolean exists(PathFragment path) {
    return fsForPath(path).exists(path);
  }

  @Override
  public Collection<Dirent> readdir(PathFragment path, boolean followSymlinks) throws IOException {
    return fsForPath(path).readdir(path, followSymlinks);
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
      var metadata =
          TracingMetadataUtils.buildMetadata(
              buildRequestId, commandId, repoName, /* actionExecutionMetadata= */ null);
      // Files in the remote external repo that Bazel reads are worth writing through to the
      // disk cache, as they are likely to be read again on future cold builds.
      return RemoteActionExecutionContext.create(metadata)
          .withReadCachePolicy(RemoteActionExecutionContext.CachePolicy.ANY_CACHE)
          .withWriteCachePolicy(RemoteActionExecutionContext.CachePolicy.ANY_CACHE);
    }

    private FileArtifactValue getMetadata(PathFragment path) throws IOException {
      var info =
          (RemoteActionFileSystem.RemoteInMemoryFileInfo) stat(path, /* followSymlinks= */ true);
      return info.getMetadata();
    }

    // TODO: Support rewinding to allow incremental builds to recover from a lost remote BUILD or
    //  .bzl file.
    @Override
    public InputStream getInputStream(PathFragment path) throws IOException {
      var relativePath = path.relativeTo(externalDirectory);
      var info =
          (RemoteActionFileSystem.RemoteInMemoryFileInfo) stat(path, /* followSymlinks= */ true);
      reporter.post(
          new ExtendedEventHandler.FetchProgress() {
            @Override
            public String getResourceIdentifier() {
              return relativePath.getPathString();
            }

            @Override
            public String getProgress() {
              return "(%s)".formatted(Utils.bytesCountToDisplayString(info.getSize()));
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
      var info =
          (RemoteActionFileSystem.RemoteInMemoryFileInfo) stat(path, /* followSymlinks= */ true);
      return info.getMetadata().getDigest();
    }

    @Override
    public byte[] getFastDigest(PathFragment path) throws IOException {
      return getDigest(path);
    }
  }
}
