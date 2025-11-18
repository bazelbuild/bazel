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
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.remote.common.BulkTransferException;
import com.google.devtools.build.lib.remote.common.RemoteActionExecutionContext;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.TracingMetadataUtils;
import com.google.devtools.build.lib.remote.util.Utils;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
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
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import javax.annotation.Nullable;

/**
 * A file system that overlays the native file system with a {@link RemoteExternalFileSystem} for
 * the "external" directory, which contains the contents of external repositories.
 *
 * <p>Each external repository can either be materialized to the native file system or kept in
 * memory in the {@link RemoteExternalFileSystem}.
 */
public final class RemoteExternalOverlayFileSystem extends FileSystem {

  @Override
  public boolean isFilePathCaseSensitive() {
    return nativeFs.isFilePathCaseSensitive();
  }

  private final PathFragment externalDirectory;
  private final int externalDirectorySegmentCount;
  private final FileSystem nativeFs;
  private final RemoteExternalFileSystem externalFs;
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
  @Nullable private ExecutorService materializationExecutor;

  public RemoteExternalOverlayFileSystem(PathFragment externalDirectory, FileSystem nativeFs) {
    super(nativeFs.getDigestFunction());
    this.externalDirectory = externalDirectory;
    this.externalDirectorySegmentCount = externalDirectory.segmentCount();
    this.nativeFs = nativeFs;
    this.externalFs = new RemoteExternalFileSystem(nativeFs.getDigestFunction());
  }

  @SuppressWarnings("AllowVirtualThreads")
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
            && this.evaluator == null
            && this.materializationExecutor == null);
    this.cache = cache;
    this.inputPrefetcher = inputPrefetcher;
    this.reporter = reporter;
    this.buildRequestId = buildRequestId;
    this.commandId = commandId;
    this.evaluator = evaluator;
    this.materializationExecutor =
        Executors.newThreadPerTaskExecutor(
            Thread.ofVirtual().name("remote-repo-materialization-", 0).factory());
  }

  public void afterCommand() {
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
        repoName -> {
          try {
            externalFs.getPath(externalDirectory.getChild(repoName)).deleteTree();
          } catch (IOException e) {
            throw new IllegalStateException("In-memory file system is not expected to throw", e);
          }
          materializations.remove(repoName);
          markerFileContents.remove(repoName);
        });
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
    // Create the repo directory on disk so that readdir reflects the overlaid state of the external
    // directory.
    nativeFs.createDirectoryAndParents(externalDirectory.getChild(repo.getName()));
    // Keep the marker file contents in memory so that it can be written out when the repo is
    // materialized. This doubles as a presence marker for the in-memory repo contents.
    markerFileContents.put(repo.getName(), markerFile);
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
          FileArtifactValue.RemoteFileArtifactValue.create(
              DigestUtil.toBinaryDigest(file.getDigest()),
              file.getDigest().getSizeBytes(),
              /* locationIndex= */ 1));
      fs.getPath(filePath).setExecutable(file.getIsExecutable());
      // The RE API does not track whether a file is readable or writable. We choose to make all
      // files readable and not writable to ensure that other repo rules can't accidentally modify
      // the cached repo.
      fs.getPath(filePath).setWritable(false);
    }
    for (var symlink : dir.getSymlinksList()) {
      fs.getPath(path.getRelative(unicodeToInternal(symlink.getName())))
          .createSymbolicLink(PathFragment.create(unicodeToInternal(symlink.getTarget())));
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
    var unused =
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
    FileSystemUtils.writeContentAsLatin1(
        markerFileSibling, markerFileContents.remove(repo.getName()));
    markerFileSibling.renameTo(markerFile);
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

  // Always mirror tree deletions to the underlying native file system to support bazel clean and
  // repository refetching.

  @Override
  protected void deleteTree(PathFragment path) throws IOException {
    nativeFs.getPath(path).deleteTree();
    externalFs.getPath(path).deleteTree();
  }

  @Override
  protected void deleteTreesBelow(PathFragment dir) throws IOException {
    nativeFs.getPath(dir).deleteTreesBelow();
    externalFs.getPath(dir).deleteTreesBelow();
  }

  // All other methods delegate to the file system given by this method. It is important to override
  // each non-final FileSystem method to benefit from optimizations implemented in the respective
  // underlying file systems.
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

  /** Returns a {@link Path} on the delegate file system for the given path fragment. */
  private Path delegatePath(PathFragment path) {
    return fsForPath(path).getPath(path);
  }

  @Override
  protected boolean delete(PathFragment path) throws IOException {
    return delegatePath(path).delete();
  }

  @Override
  protected byte[] getDigest(PathFragment path) throws IOException {
    return delegatePath(path).getDigest();
  }

  @Nullable
  @Override
  protected byte[] getFastDigest(PathFragment path) throws IOException {
    return delegatePath(path).getFastDigest();
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
  public boolean createDirectory(PathFragment path) throws IOException {
    return fsForPath(path).createDirectory(path);
  }

  @Override
  public void createDirectoryAndParents(PathFragment path) throws IOException {
    fsForPath(path).createDirectoryAndParents(path);
  }

  @Override
  protected long getFileSize(PathFragment path, boolean followSymlinks) throws IOException {
    return delegatePath(path).getFileSize(followSymlinks ? Symlinks.FOLLOW : Symlinks.NOFOLLOW);
  }

  @Override
  protected long getLastModifiedTime(PathFragment path, boolean followSymlinks) throws IOException {
    return delegatePath(path).getLastModifiedTime(
        followSymlinks ? Symlinks.FOLLOW : Symlinks.NOFOLLOW);
  }

  @Override
  public void setLastModifiedTime(PathFragment path, long newTime) throws IOException {
    delegatePath(path).setLastModifiedTime(newTime);
  }

  @Override
  protected FileStatus stat(PathFragment path, boolean followSymlinks) throws IOException {
    return delegatePath(path).stat(followSymlinks ? Symlinks.FOLLOW : Symlinks.NOFOLLOW);
  }

  @Override
  protected void createSymbolicLink(
      PathFragment linkPath, PathFragment targetFragment, SymlinkTargetType hint)
      throws IOException {
    delegatePath(linkPath).createSymbolicLink(targetFragment);
  }

  @Override
  protected PathFragment readSymbolicLink(PathFragment path) throws IOException {
    return delegatePath(path).readSymbolicLink();
  }

  @Override
  protected boolean exists(PathFragment path, boolean followSymlinks) {
    return delegatePath(path).exists(followSymlinks ? Symlinks.FOLLOW : Symlinks.NOFOLLOW);
  }

  @Override
  public boolean exists(PathFragment path) {
    return delegatePath(path).exists();
  }

  @Override
  protected Collection<String> getDirectoryEntries(PathFragment path) throws IOException {
    return delegatePath(path).getDirectoryEntries().stream()
        .map(Path::getBaseName)
        .collect(ImmutableList.toImmutableList());
  }

  @Override
  protected boolean isReadable(PathFragment path) throws IOException {
    return delegatePath(path).isReadable();
  }

  @Override
  protected void setReadable(PathFragment path, boolean readable) throws IOException {
    delegatePath(path).setReadable(readable);
  }

  @Override
  protected boolean isWritable(PathFragment path) throws IOException {
    return delegatePath(path).isWritable();
  }

  @Override
  public void setWritable(PathFragment path, boolean writable) throws IOException {
    delegatePath(path).setWritable(writable);
  }

  @Override
  protected boolean isExecutable(PathFragment path) throws IOException {
    return delegatePath(path).isExecutable();
  }

  @Override
  protected void setExecutable(PathFragment path, boolean executable) throws IOException {
    delegatePath(path).setExecutable(executable);
  }

  @Override
  protected InputStream getInputStream(PathFragment path) throws IOException {
    return delegatePath(path).getInputStream();
  }

  @Override
  protected SeekableByteChannel createReadWriteByteChannel(PathFragment path) throws IOException {
    return delegatePath(path).createReadWriteByteChannel();
  }

  @Override
  protected OutputStream getOutputStream(PathFragment path, boolean append, boolean internal)
      throws IOException {
    return delegatePath(path).getOutputStream(append);
  }

  @Override
  public void renameTo(PathFragment sourcePath, PathFragment targetPath) throws IOException {
    delegatePath(sourcePath).renameTo(delegatePath(targetPath));
  }

  @Override
  protected void createFSDependentHardLink(PathFragment linkPath, PathFragment originalPath)
      throws IOException {
    // Fall back to creating a copy if hard links aren't supported across file systems.
    FileSystemUtils.copyFile(delegatePath(originalPath), delegatePath(linkPath));
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

  @Override
  protected Path resolveSymbolicLinks(PathFragment path) throws IOException {
    return delegatePath(path).resolveSymbolicLinks();
  }

  @Nullable
  @Override
  protected FileStatus statNullable(PathFragment path, boolean followSymlinks) {
    return delegatePath(path).statNullable(followSymlinks ? Symlinks.FOLLOW : Symlinks.NOFOLLOW);
  }

  @Nullable
  @Override
  protected FileStatus statIfFound(PathFragment path, boolean followSymlinks) throws IOException {
    return delegatePath(path).statIfFound(followSymlinks ? Symlinks.FOLLOW : Symlinks.NOFOLLOW);
  }

  @Override
  protected boolean isFile(PathFragment path, boolean followSymlinks) {
    return delegatePath(path).isFile(followSymlinks ? Symlinks.FOLLOW : Symlinks.NOFOLLOW);
  }

  @Override
  protected boolean isSpecialFile(PathFragment path, boolean followSymlinks) {
    return delegatePath(path).isSpecialFile(followSymlinks ? Symlinks.FOLLOW : Symlinks.NOFOLLOW);
  }

  @Override
  protected boolean isSymbolicLink(PathFragment path) {
    return delegatePath(path).isSymbolicLink();
  }

  @Override
  protected boolean isDirectory(PathFragment path, boolean followSymlinks) {
    return delegatePath(path).isDirectory(
        followSymlinks ? Symlinks.FOLLOW : Symlinks.NOFOLLOW);
  }

  @Override
  protected PathFragment readSymbolicLinkUnchecked(PathFragment path) throws IOException {
    return delegatePath(path).readSymbolicLinkUnchecked();
  }

  @Override
  protected Collection<Dirent> readdir(PathFragment path, boolean followSymlinks)
      throws IOException {
    return delegatePath(path).readdir(
        followSymlinks ? Symlinks.FOLLOW : Symlinks.NOFOLLOW);
  }

  @Override
  protected void chmod(PathFragment path, int mode) throws IOException {
    delegatePath(path).chmod(mode);
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
              buildRequestId, commandId, repoName, /* actionMetadata= */ null);
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

    @Override
    public synchronized InputStream getInputStream(PathFragment path) throws IOException {
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
          throw new IOException(
              "%s/%s with digest %s is no longer available in the remote cache"
                  .formatted(
                      externalDirectory.getBaseName(), relativePath, DigestUtil.toString(digest)),
              e);
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
    public synchronized byte[] getFastDigest(PathFragment path) throws IOException {
      return getDigest(path);
    }
  }
}
